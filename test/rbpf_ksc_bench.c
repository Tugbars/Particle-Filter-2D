/**
 * @file rbpf_ksc_bench.c
 * @brief Benchmark for RBPF-KSC: latency and accuracy vs true volatility
 *
 * Compile:
 *   gcc -O3 -march=native -fopenmp rbpf_ksc.c rbpf_ksc_bench.c -o rbpf_bench \
 *       -lmkl_rt -lm -DNDEBUG
 *
 * Or with Intel compiler:
 *   icx -O3 -xHost -qopenmp rbpf_ksc.c rbpf_ksc_bench.c -o rbpf_bench \
 *       -qmkl -DNDEBUG
 */

#include "rbpf_ksc.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * HIGH-RESOLUTION TIMER
 *───────────────────────────────────────────────────────────────────────────*/

static inline double get_time_us(void)
{
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)freq.QuadPart * 1e6;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e6 + ts.tv_nsec / 1e3;
#endif
}

/*─────────────────────────────────────────────────────────────────────────────
 * SYNTHETIC DATA GENERATOR
 *
 * Generates returns with known stochastic volatility process:
 *   log_vol[t] = (1-θ)*log_vol[t-1] + θ*μ + σ_v*ε_v
 *   return[t] = exp(log_vol[t]) * ε_r
 *
 * With regime switches at known points.
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    float *returns;      /* Simulated returns */
    float *true_vol;     /* True volatility (for accuracy measurement) */
    float *true_log_vol; /* True log-volatility */
    int *true_regime;    /* True regime at each timestep */
    int n;               /* Number of observations */
} SyntheticData;

static float randn(void)
{
    /* Box-Muller */
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-10f)
        u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185f * u2);
}

SyntheticData *generate_synthetic_data(int n, int seed)
{
    srand(seed);

    SyntheticData *data = (SyntheticData *)malloc(sizeof(SyntheticData));
    data->n = n;
    data->returns = (float *)malloc(n * sizeof(float));
    data->true_vol = (float *)malloc(n * sizeof(float));
    data->true_log_vol = (float *)malloc(n * sizeof(float));
    data->true_regime = (int *)malloc(n * sizeof(int));

    /* Regime parameters (matching RBPF setup) */
    /* Regime 0: Low vol, stable */
    /* Regime 1: Medium vol */
    /* Regime 2: High vol (crisis) */
    /* Regime 3: Very high vol (extreme) */

    float theta[4] = {0.05f, 0.08f, 0.15f, 0.20f};
    float mu_vol[4] = {logf(0.01f), logf(0.02f), logf(0.05f), logf(0.10f)};
    float sigma_vol[4] = {0.05f, 0.10f, 0.20f, 0.30f};

    /* Regime schedule: create interesting volatility patterns */
    int regime = 0;
    float log_vol = mu_vol[0];

    for (int t = 0; t < n; t++)
    {
        /* Regime switches at specific points */
        if (t == n / 5)
            regime = 1; /* 20%: enter medium vol */
        if (t == 2 * n / 5)
            regime = 2; /* 40%: crisis begins */
        if (t == n / 2)
            regime = 3; /* 50%: peak crisis */
        if (t == 3 * n / 5)
            regime = 2; /* 60%: crisis easing */
        if (t == 7 * n / 10)
            regime = 1; /* 70%: recovery */
        if (t == 4 * n / 5)
            regime = 0; /* 80%: back to calm */

        /* Evolve log-vol with OU process */
        float th = theta[regime];
        float mv = mu_vol[regime];
        float sv = sigma_vol[regime];

        log_vol = (1.0f - th) * log_vol + th * mv + sv * randn();

        /* Generate return */
        float vol = expf(log_vol);
        float ret = vol * randn();

        data->returns[t] = ret;
        data->true_vol[t] = vol;
        data->true_log_vol[t] = log_vol;
        data->true_regime[t] = regime;
    }

    return data;
}

void free_synthetic_data(SyntheticData *data)
{
    if (data)
    {
        free(data->returns);
        free(data->true_vol);
        free(data->true_log_vol);
        free(data->true_regime);
        free(data);
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * ACCURACY METRICS
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    float mae_vol;         /* Mean Absolute Error on volatility */
    float rmse_vol;        /* Root Mean Squared Error on volatility */
    float mae_log_vol;     /* MAE on log-volatility */
    float max_error_vol;   /* Maximum absolute error */
    float correlation;     /* Correlation with true vol */
    float tail_mae;        /* MAE when true_vol > 90th percentile */
    float regime_accuracy; /* % correct regime detection */
} AccuracyMetrics;

static int compare_float(const void *a, const void *b)
{
    float fa = *(const float *)a;
    float fb = *(const float *)b;
    return (fa > fb) - (fa < fb);
}

static int compare_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

AccuracyMetrics compute_accuracy(const float *est_vol, const float *est_log_vol,
                                 const int *est_regime, const SyntheticData *data)
{
    AccuracyMetrics m = {0};
    int n = data->n;

    /* Basic stats */
    float sum_ae = 0, sum_se = 0, sum_ae_lv = 0;
    float max_err = 0;
    int regime_correct = 0;

    /* For correlation */
    float sum_true = 0, sum_est = 0;
    float sum_true2 = 0, sum_est2 = 0, sum_te = 0;

    /* Find 90th percentile of true vol for tail analysis */
    float *sorted_vol = (float *)malloc(n * sizeof(float));
    memcpy(sorted_vol, data->true_vol, n * sizeof(float));
    qsort(sorted_vol, n, sizeof(float), compare_float);
    float p90_vol = sorted_vol[(int)(0.9f * n)];
    free(sorted_vol);

    int tail_count = 0;
    float tail_sum_ae = 0;

    for (int t = 0; t < n; t++)
    {
        float true_v = data->true_vol[t];
        float est_v = est_vol[t];
        float true_lv = data->true_log_vol[t];
        float est_lv = est_log_vol[t];

        float ae = fabsf(true_v - est_v);
        float se = (true_v - est_v) * (true_v - est_v);
        float ae_lv = fabsf(true_lv - est_lv);

        sum_ae += ae;
        sum_se += se;
        sum_ae_lv += ae_lv;
        if (ae > max_err)
            max_err = ae;

        sum_true += true_v;
        sum_est += est_v;
        sum_true2 += true_v * true_v;
        sum_est2 += est_v * est_v;
        sum_te += true_v * est_v;

        if (data->true_regime[t] == est_regime[t])
            regime_correct++;

        /* Tail analysis */
        if (true_v > p90_vol)
        {
            tail_sum_ae += ae;
            tail_count++;
        }
    }

    m.mae_vol = sum_ae / n;
    m.rmse_vol = sqrtf(sum_se / n);
    m.mae_log_vol = sum_ae_lv / n;
    m.max_error_vol = max_err;
    m.regime_accuracy = (float)regime_correct / n * 100.0f;
    m.tail_mae = (tail_count > 0) ? tail_sum_ae / tail_count : 0;

    /* Pearson correlation */
    float n_f = (float)n;
    float cov = sum_te - sum_true * sum_est / n_f;
    float var_true = sum_true2 - sum_true * sum_true / n_f;
    float var_est = sum_est2 - sum_est * sum_est / n_f;
    m.correlation = cov / (sqrtf(var_true * var_est) + 1e-10f);

    return m;
}

/*─────────────────────────────────────────────────────────────────────────────
 * BENCHMARK RUNNER
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    int n_particles;
    double latency_mean_us;
    double latency_p50_us;
    double latency_p99_us;
    double latency_max_us;
    AccuracyMetrics accuracy;
} BenchmarkResult;

BenchmarkResult run_benchmark(int n_particles, const SyntheticData *data,
                              int warmup_iters, int measure_iters)
{
    BenchmarkResult result = {0};
    result.n_particles = n_particles;

    int n_obs = data->n;

    /* Create filter */
    RBPF_KSC *rbpf = rbpf_ksc_create(n_particles, 4);
    if (!rbpf)
    {
        fprintf(stderr, "Failed to create RBPF with %d particles\n", n_particles);
        return result;
    }

    /* Configure regimes to match synthetic data */
    rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, logf(0.01f), 0.05f);
    rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, logf(0.02f), 0.10f);
    rbpf_ksc_set_regime_params(rbpf, 2, 0.15f, logf(0.05f), 0.20f);
    rbpf_ksc_set_regime_params(rbpf, 3, 0.20f, logf(0.10f), 0.30f);

    /* Build transition matrix (sticky regimes) */
    float trans[16] = {
        0.95f, 0.03f, 0.015f, 0.005f, /* From regime 0 */
        0.05f, 0.90f, 0.04f, 0.01f,   /* From regime 1 */
        0.02f, 0.08f, 0.85f, 0.05f,   /* From regime 2 */
        0.01f, 0.04f, 0.10f, 0.85f    /* From regime 3 */
    };
    rbpf_ksc_build_transition_lut(rbpf, trans);

    /* Regularization */
    rbpf_ksc_set_regularization(rbpf, 0.02f, 0.001f);

    /* Allocate output storage */
    float *est_vol = (float *)malloc(n_obs * sizeof(float));
    float *est_log_vol = (float *)malloc(n_obs * sizeof(float));
    int *est_regime = (int *)malloc(n_obs * sizeof(int));
    double *latencies = (double *)malloc(n_obs * sizeof(double));

    /* Warmup runs */
    for (int iter = 0; iter < warmup_iters; iter++)
    {
        rbpf_ksc_init(rbpf, logf(0.01f), 0.1f);
        rbpf_ksc_warmup(rbpf);

        RBPF_KSC_Output out;
        for (int t = 0; t < n_obs; t++)
        {
            rbpf_ksc_step(rbpf, data->returns[t], &out);
        }
    }

    /* Measurement runs */
    double total_latency = 0;

    for (int iter = 0; iter < measure_iters; iter++)
    {
        rbpf_ksc_init(rbpf, logf(0.01f), 0.1f);

        RBPF_KSC_Output out;
        for (int t = 0; t < n_obs; t++)
        {
            double t0 = get_time_us();
            rbpf_ksc_step(rbpf, data->returns[t], &out);
            double t1 = get_time_us();

            latencies[t] = t1 - t0;
            total_latency += latencies[t];

            /* Store last iteration's estimates for accuracy */
            if (iter == measure_iters - 1)
            {
                est_vol[t] = out.vol_mean;
                est_log_vol[t] = out.log_vol_mean;
                est_regime[t] = out.dominant_regime;
            }
        }
    }

    /* Latency statistics */
    result.latency_mean_us = total_latency / (measure_iters * n_obs);

    /* Sort for percentiles */
    qsort(latencies, n_obs, sizeof(double), compare_double);
    result.latency_p50_us = latencies[n_obs / 2];
    result.latency_p99_us = latencies[(int)(0.99 * n_obs)];
    result.latency_max_us = latencies[n_obs - 1];

    /* Accuracy */
    result.accuracy = compute_accuracy(est_vol, est_log_vol, est_regime, data);

    /* Cleanup */
    free(est_vol);
    free(est_log_vol);
    free(est_regime);
    free(latencies);
    rbpf_ksc_destroy(rbpf);

    return result;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DIAGNOSTIC OUTPUT
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct
{
    float *est_vol;
    float *est_log_vol;
    float *true_vol;
    float *true_log_vol;
    int *est_regime;
    int *true_regime;
    float *surprise;
    float *regime_entropy;
    float *vol_ratio;
    float *ess;
    int *regime_changed;
    int *change_type;
    int n;
} DiagnosticData;

DiagnosticData *run_diagnostic(int n_particles, const SyntheticData *data)
{
    int n_obs = data->n;

    DiagnosticData *diag = (DiagnosticData *)malloc(sizeof(DiagnosticData));
    diag->n = n_obs;
    diag->est_vol = (float *)malloc(n_obs * sizeof(float));
    diag->est_log_vol = (float *)malloc(n_obs * sizeof(float));
    diag->true_vol = (float *)malloc(n_obs * sizeof(float));
    diag->true_log_vol = (float *)malloc(n_obs * sizeof(float));
    diag->est_regime = (int *)malloc(n_obs * sizeof(int));
    diag->true_regime = (int *)malloc(n_obs * sizeof(int));
    diag->surprise = (float *)malloc(n_obs * sizeof(float));
    diag->regime_entropy = (float *)malloc(n_obs * sizeof(float));
    diag->vol_ratio = (float *)malloc(n_obs * sizeof(float));
    diag->ess = (float *)malloc(n_obs * sizeof(float));
    diag->regime_changed = (int *)malloc(n_obs * sizeof(int));
    diag->change_type = (int *)malloc(n_obs * sizeof(int));

    /* Copy true values */
    memcpy(diag->true_vol, data->true_vol, n_obs * sizeof(float));
    memcpy(diag->true_log_vol, data->true_log_vol, n_obs * sizeof(float));
    memcpy(diag->true_regime, data->true_regime, n_obs * sizeof(int));

    /* Create and configure filter */
    RBPF_KSC *rbpf = rbpf_ksc_create(n_particles, 4);
    if (!rbpf)
        return diag;

    rbpf_ksc_set_regime_params(rbpf, 0, 0.05f, logf(0.01f), 0.05f);
    rbpf_ksc_set_regime_params(rbpf, 1, 0.08f, logf(0.02f), 0.10f);
    rbpf_ksc_set_regime_params(rbpf, 2, 0.15f, logf(0.05f), 0.20f);
    rbpf_ksc_set_regime_params(rbpf, 3, 0.20f, logf(0.10f), 0.30f);

    float trans[16] = {
        0.95f, 0.03f, 0.015f, 0.005f,
        0.05f, 0.90f, 0.04f, 0.01f,
        0.02f, 0.08f, 0.85f, 0.05f,
        0.01f, 0.04f, 0.10f, 0.85f};
    rbpf_ksc_build_transition_lut(rbpf, trans);
    rbpf_ksc_set_regularization(rbpf, 0.02f, 0.001f);
    rbpf_ksc_init(rbpf, logf(0.01f), 0.1f);

    /* Run filter and collect diagnostics */
    RBPF_KSC_Output out;
    for (int t = 0; t < n_obs; t++)
    {
        rbpf_ksc_step(rbpf, data->returns[t], &out);

        diag->est_vol[t] = out.vol_mean;
        diag->est_log_vol[t] = out.log_vol_mean;
        diag->est_regime[t] = out.dominant_regime;
        diag->surprise[t] = out.surprise;
        diag->regime_entropy[t] = out.regime_entropy;
        diag->vol_ratio[t] = out.vol_ratio;
        diag->ess[t] = out.ess;
        diag->regime_changed[t] = out.regime_changed;
        diag->change_type[t] = out.change_type;
    }

    rbpf_ksc_destroy(rbpf);
    return diag;
}

void free_diagnostic(DiagnosticData *diag)
{
    if (diag)
    {
        free(diag->est_vol);
        free(diag->est_log_vol);
        free(diag->true_vol);
        free(diag->true_log_vol);
        free(diag->est_regime);
        free(diag->true_regime);
        free(diag->surprise);
        free(diag->regime_entropy);
        free(diag->vol_ratio);
        free(diag->ess);
        free(diag->regime_changed);
        free(diag->change_type);
        free(diag);
    }
}

void export_diagnostic_csv(const DiagnosticData *diag, const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f)
    {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    fprintf(f, "t,true_vol,est_vol,true_log_vol,est_log_vol,true_regime,est_regime,"
               "surprise,entropy,vol_ratio,ess,detected,change_type\n");

    for (int t = 0; t < diag->n; t++)
    {
        fprintf(f, "%d,%.6f,%.6f,%.4f,%.4f,%d,%d,%.4f,%.4f,%.4f,%.1f,%d,%d\n",
                t,
                diag->true_vol[t], diag->est_vol[t],
                diag->true_log_vol[t], diag->est_log_vol[t],
                diag->true_regime[t], diag->est_regime[t],
                diag->surprise[t], diag->regime_entropy[t],
                diag->vol_ratio[t], diag->ess[t],
                diag->regime_changed[t], diag->change_type[t]);
    }

    fclose(f);
    printf("Diagnostic data exported to: %s\n", filename);
}

void print_changepoint_analysis(const DiagnosticData *diag, const SyntheticData *data)
{
    printf("\n=== Changepoint Analysis ===\n\n");

    /* Find true changepoints */
    int n = data->n;
    int prev_regime = data->true_regime[0];

    printf("True Changepoints vs Filter Response:\n");
    printf("%-8s %-12s %-12s %-10s %-10s %-10s %-10s\n",
           "Time", "Regime", "Est Regime", "Surprise", "Entropy", "VolRatio", "Detected?");
    printf("────────────────────────────────────────────────────────────────────────────\n");

    for (int t = 1; t < n; t++)
    {
        if (data->true_regime[t] != prev_regime)
        {
            /* True changepoint - show window around it */
            int detected_nearby = 0;
            float max_surprise = 0;
            float max_entropy = 0;

            /* Look in window [t-5, t+10] for detection */
            for (int w = (t > 5 ? t - 5 : 0); w < (t + 10 < n ? t + 10 : n); w++)
            {
                if (diag->regime_changed[w])
                    detected_nearby = 1;
                if (diag->surprise[w] > max_surprise)
                    max_surprise = diag->surprise[w];
                if (diag->regime_entropy[w] > max_entropy)
                    max_entropy = diag->regime_entropy[w];
            }

            printf("%-8d %d → %-8d %-12d %-10.2f %-10.2f %-10.2f %-10s\n",
                   t,
                   prev_regime, data->true_regime[t],
                   diag->est_regime[t],
                   diag->surprise[t],
                   diag->regime_entropy[t],
                   diag->vol_ratio[t],
                   detected_nearby ? "✓" : "✗");

            prev_regime = data->true_regime[t];
        }
    }

    /* Count false positives */
    int false_positives = 0;
    int true_positives = 0;
    prev_regime = data->true_regime[0];

    for (int t = 1; t < n; t++)
    {
        if (diag->regime_changed[t])
        {
            /* Check if near a true changepoint */
            int near_true_cp = 0;
            for (int w = (t > 10 ? t - 10 : 0); w < (t + 10 < n ? t + 10 : n); w++)
            {
                if (w > 0 && data->true_regime[w] != data->true_regime[w - 1])
                {
                    near_true_cp = 1;
                    break;
                }
            }
            if (near_true_cp)
                true_positives++;
            else
                false_positives++;
        }
    }

    printf("\nDetection Summary:\n");
    printf("  True positives:  %d\n", true_positives);
    printf("  False positives: %d\n", false_positives);
}

void print_tracking_summary(const DiagnosticData *diag)
{
    printf("\n=== Tracking Quality by Regime ===\n\n");

    int n = diag->n;
    float mae_by_regime[4] = {0};
    int count_by_regime[4] = {0};

    for (int t = 0; t < n; t++)
    {
        int r = diag->true_regime[t];
        if (r >= 0 && r < 4)
        {
            mae_by_regime[r] += fabsf(diag->true_log_vol[t] - diag->est_log_vol[t]);
            count_by_regime[r]++;
        }
    }

    printf("%-10s %-10s %-12s %-12s\n", "Regime", "Count", "MAE(log-vol)", "Typical Vol");
    printf("────────────────────────────────────────────────\n");

    float typical_vol[4] = {0.01f, 0.02f, 0.05f, 0.10f};
    for (int r = 0; r < 4; r++)
    {
        if (count_by_regime[r] > 0)
        {
            mae_by_regime[r] /= count_by_regime[r];
        }
        printf("%-10d %-10d %-12.4f %-12.4f\n",
               r, count_by_regime[r], mae_by_regime[r], typical_vol[r]);
    }

    /* Surprise statistics */
    printf("\n=== Surprise Statistics ===\n\n");

    float surprise_sum = 0, surprise_max = 0;
    for (int t = 0; t < n; t++)
    {
        surprise_sum += diag->surprise[t];
        if (diag->surprise[t] > surprise_max)
            surprise_max = diag->surprise[t];
    }

    printf("Mean surprise: %.2f\n", surprise_sum / n);
    printf("Max surprise:  %.2f\n", surprise_max);

    /* ESS statistics */
    float ess_sum = 0, ess_min = diag->ess[0];
    for (int t = 0; t < n; t++)
    {
        ess_sum += diag->ess[t];
        if (diag->ess[t] < ess_min)
            ess_min = diag->ess[t];
    }

    printf("\nMean ESS: %.1f\n", ess_sum / n);
    printf("Min ESS:  %.1f\n", ess_min);
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN
 *───────────────────────────────────────────────────────────────────────────*/

int main(int argc, char **argv)
{
    printf("=== RBPF-KSC Benchmark ===\n\n");

    /* Configuration */
    int n_obs = 5000; /* Observations per run */
    int warmup = 3;   /* Warmup iterations */
    int measure = 5;  /* Measurement iterations */
    int seed = 42;

    /* Particle counts to test */
    int particle_counts[] = {50, 100, 200, 500, 1000, 2000};
    int n_configs = sizeof(particle_counts) / sizeof(particle_counts[0]);

    /* Generate synthetic data */
    printf("Generating synthetic data: %d observations, seed=%d\n", n_obs, seed);
    SyntheticData *data = generate_synthetic_data(n_obs, seed);

    /* Print data summary */
    float min_vol = data->true_vol[0], max_vol = data->true_vol[0];
    for (int t = 1; t < n_obs; t++)
    {
        if (data->true_vol[t] < min_vol)
            min_vol = data->true_vol[t];
        if (data->true_vol[t] > max_vol)
            max_vol = data->true_vol[t];
    }
    printf("True volatility range: [%.4f, %.4f] (%.1fx dynamic range)\n\n",
           min_vol, max_vol, max_vol / min_vol);

    /* Run benchmarks */
    printf("%-10s | %8s %8s %8s %8s | %8s %8s %8s %6s\n",
           "Particles", "Mean", "P50", "P99", "Max",
           "MAE_vol", "Tail_MAE", "Corr", "Regime%");
    printf("%-10s | %8s %8s %8s %8s | %8s %8s %8s %6s\n",
           "", "(μs)", "(μs)", "(μs)", "(μs)", "", "", "", "");
    printf("──────────────────────────────────────────────────────────────────────────\n");

    for (int i = 0; i < n_configs; i++)
    {
        int n_p = particle_counts[i];
        BenchmarkResult r = run_benchmark(n_p, data, warmup, measure);

        printf("%-10d | %8.2f %8.2f %8.2f %8.2f | %8.4f %8.4f %8.4f %5.1f%%\n",
               r.n_particles,
               r.latency_mean_us, r.latency_p50_us, r.latency_p99_us, r.latency_max_us,
               r.accuracy.mae_vol, r.accuracy.tail_mae,
               r.accuracy.correlation, r.accuracy.regime_accuracy);
    }

    printf("\n");

    /* Detailed report for recommended config */
    printf("=== Detailed Report (n=200 particles) ===\n");
    BenchmarkResult detailed = run_benchmark(200, data, warmup, measure);
    printf("Latency:\n");
    printf("  Mean:  %.2f μs\n", detailed.latency_mean_us);
    printf("  P50:   %.2f μs\n", detailed.latency_p50_us);
    printf("  P99:   %.2f μs\n", detailed.latency_p99_us);
    printf("  Max:   %.2f μs\n", detailed.latency_max_us);
    printf("\nAccuracy:\n");
    printf("  MAE (vol):      %.4f\n", detailed.accuracy.mae_vol);
    printf("  RMSE (vol):     %.4f\n", detailed.accuracy.rmse_vol);
    printf("  MAE (log-vol):  %.4f\n", detailed.accuracy.mae_log_vol);
    printf("  Max error:      %.4f\n", detailed.accuracy.max_error_vol);
    printf("  Tail MAE:       %.4f (90th percentile vol)\n", detailed.accuracy.tail_mae);
    printf("  Correlation:    %.4f\n", detailed.accuracy.correlation);
    printf("  Regime acc:     %.1f%%\n", detailed.accuracy.regime_accuracy);

    /* Budget check */
    printf("\n=== Latency Budget Check (200μs total) ===\n");
    float ssa_us = 140.0f;
    float rbpf_us = (float)detailed.latency_mean_us;
    float kelly_us = 0.5f;
    float total_us = ssa_us + rbpf_us + kelly_us;
    float headroom = 200.0f - total_us;

    printf("  SSA:      %.1f μs\n", ssa_us);
    printf("  RBPF:     %.1f μs (measured)\n", rbpf_us);
    printf("  Kelly:    %.1f μs\n", kelly_us);
    printf("  ─────────────────\n");
    printf("  Total:    %.1f μs\n", total_us);
    printf("  Headroom: %.1f μs %s\n", headroom,
           headroom > 0 ? "✓" : "✗ OVER BUDGET");

    /* Run diagnostics with 200 particles */
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");
    printf("                         DIAGNOSTIC ANALYSIS\n");
    printf("════════════════════════════════════════════════════════════════════════════\n");

    DiagnosticData *diag = run_diagnostic(200, data);

    print_changepoint_analysis(diag, data);
    print_tracking_summary(diag);

    /* Export CSV for plotting */
    export_diagnostic_csv(diag, "rbpf_diagnostic.csv");

    free_diagnostic(diag);

    /* Cleanup */
    free_synthetic_data(data);

    return 0;
}