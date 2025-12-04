/**
 * @file example_usage_2d.c
 * @brief Example usage of 2D Particle Filter with stochastic volatility
 *
 * Demonstrates:
 *   - MKL/OpenMP configuration for Intel hybrid CPUs (P-core pinning)
 *   - Proper warmup to eliminate first-call latency
 *   - High-resolution timing with latency percentiles
 *   - Regime-switching stochastic volatility tracking
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>

/* MKL configuration for hybrid CPUs - must be before particle_filter_2d.h */
#define SSA_USE_MKL
#include "mkl_config.h"

#include "particle_filter_2d.h"

/*============================================================================
 * CROSS-PLATFORM HIGH-RESOLUTION TIMER
 *============================================================================*/

#ifdef _WIN32
#include <windows.h>

static LARGE_INTEGER g_freq;
static int g_timer_initialized = 0;

static void timer_init(void)
{
    if (!g_timer_initialized)
    {
        QueryPerformanceFrequency(&g_freq);
        g_timer_initialized = 1;
    }
}

static double get_time_us(void)
{
    LARGE_INTEGER counter;
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1e6 / (double)g_freq.QuadPart;
}

#else
#include <sys/time.h>

static void timer_init(void) { /* No-op on Unix */ }

static double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1e6 + tv.tv_usec;
}
#endif

/*============================================================================
 * LATENCY STATISTICS
 *============================================================================*/

typedef struct
{
    double *samples;
    int capacity;
    int count;
} LatencyStats;

static void latency_init(LatencyStats *ls, int capacity)
{
    ls->samples = (double *)malloc(capacity * sizeof(double));
    ls->capacity = capacity;
    ls->count = 0;
}

static void latency_add(LatencyStats *ls, double us)
{
    if (ls->count < ls->capacity)
    {
        ls->samples[ls->count++] = us;
    }
}

static int cmp_double(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void latency_print(LatencyStats *ls)
{
    if (ls->count == 0)
        return;

    /* Sort for percentiles */
    qsort(ls->samples, ls->count, sizeof(double), cmp_double);

    /* Compute stats */
    double sum = 0, min_val = ls->samples[0], max_val = ls->samples[ls->count - 1];
    for (int i = 0; i < ls->count; i++)
    {
        sum += ls->samples[i];
    }
    double mean = sum / ls->count;

    int p50_idx = (int)(ls->count * 0.50);
    int p90_idx = (int)(ls->count * 0.90);
    int p99_idx = (int)(ls->count * 0.99);
    int p999_idx = (int)(ls->count * 0.999);

    printf("\n=== Latency Distribution ===\n");
    printf("Samples:    %d\n", ls->count);
    printf("Min:        %.2f us\n", min_val);
    printf("Mean:       %.2f us\n", mean);
    printf("P50:        %.2f us\n", ls->samples[p50_idx]);
    printf("P90:        %.2f us\n", ls->samples[p90_idx]);
    printf("P99:        %.2f us\n", ls->samples[p99_idx]);
    printf("P99.9:      %.2f us\n", ls->samples[p999_idx]);
    printf("Max:        %.2f us\n", max_val);
}

static void latency_free(LatencyStats *ls)
{
    free(ls->samples);
    ls->samples = NULL;
    ls->count = 0;
}

/*============================================================================
 * WARMUP FUNCTIONS
 *============================================================================*/

/**
 * Full warmup sequence with timing breakdown.
 * Separates OpenMP thread creation from MKL kernel warmup.
 */
static void warmup_full(PF2D *pf, PF2DRegimeProbs *rp, int n_warmup_ticks)
{
    double start, elapsed;

    printf("=== Warmup Phase ===\n");

    /* Phase 1: OpenMP thread pool creation (~1-2ms first time) */
    start = get_time_us();
#pragma omp parallel
    {
        volatile int tid = omp_get_thread_num();
        (void)tid;
    }
    elapsed = get_time_us() - start;
    printf("  OpenMP thread pool:     %.2f ms (%d threads)\n",
           elapsed / 1000.0, omp_get_max_threads());

    /* Phase 2: MKL kernel warmup via library function */
    start = get_time_us();
    pf2d_warmup(pf);
    elapsed = get_time_us() - start;
    printf("  MKL kernels (RNG/VML/BLAS): %.2f ms\n", elapsed / 1000.0);

    /* Phase 3: Filter warmup - real updates to warm instruction cache */
    start = get_time_us();

    pf2d_pcg32_t warmup_rng;
    pf2d_pcg32_seed(&warmup_rng, 99999, 0);

    pf2d_real price = 100.0;
    for (int t = 0; t < n_warmup_ticks; t++)
    {
        price += 0.01 * pf2d_pcg32_gaussian(&warmup_rng);
        pf2d_update(pf, price, rp);
    }

    elapsed = get_time_us() - start;
    printf("  Filter warmup (%d ticks): %.2f ms (%.1f us/tick)\n",
           n_warmup_ticks, elapsed / 1000.0, elapsed / n_warmup_ticks);

    printf("\nWarmup complete.\n");
}

/*============================================================================
 * SIMULATED MARKET DATA
 *============================================================================*/

typedef struct
{
    pf2d_real price;
    pf2d_real true_vol;
    int true_regime;
} MarketState;

static void simulate_tick(MarketState *m, pf2d_pcg32_t *rng)
{
    pf2d_real u = pf2d_pcg32_uniform(rng);
    if (u < 0.01)
    {
        m->true_regime = (m->true_regime + 1) % 4;
    }

    pf2d_real drift, theta_v, mu_v, sigma_v;
    switch (m->true_regime)
    {
    case 0: /* Trend */
        drift = 0.001;
        theta_v = 0.02;
        mu_v = log(0.01);
        sigma_v = 0.05;
        break;
    case 1: /* Mean-revert */
        drift = 0.0;
        theta_v = 0.05;
        mu_v = log(0.008);
        sigma_v = 0.03;
        break;
    case 2: /* High-vol */
        drift = 0.0;
        theta_v = 0.10;
        mu_v = log(0.03);
        sigma_v = 0.10;
        break;
    case 3: /* Jump */
        drift = 0.0;
        theta_v = 0.20;
        mu_v = log(0.05);
        sigma_v = 0.20;
        break;
    default:
        drift = 0.0;
        theta_v = 0.05;
        mu_v = log(0.01);
        sigma_v = 0.05;
    }

    pf2d_real lv = log(m->true_vol);
    lv = (1.0 - theta_v) * lv + theta_v * mu_v + sigma_v * pf2d_pcg32_gaussian(rng);
    m->true_vol = exp(lv);

    m->price = m->price + drift + m->true_vol * pf2d_pcg32_gaussian(rng);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char *argv[])
{
    int n_particles = 4000;
    int n_ticks = 10000;
    int n_warmup_ticks = 500;
    int verbose_mkl = 1;

    if (argc > 1)
        n_particles = atoi(argv[1]);
    if (argc > 2)
        n_ticks = atoi(argv[2]);
    if (argc > 3)
        n_warmup_ticks = atoi(argv[3]);

    /* Initialize high-resolution timer */
    timer_init();

    /*========================================================================
     * PHASE 0: MKL/OpenMP Configuration (MUST BE FIRST)
     *========================================================================*/
    printf("=== MKL/OpenMP Configuration ===\n");

    /* Pin to P-cores only on Intel hybrid CPUs (14900KF, 13900K, etc.)
     * This avoids slow E-cores and gives ~25-30% speedup.
     * Must be called BEFORE any MKL/OpenMP operations. */
    pf2d_mkl_config_14900kf(verbose_mkl);

    printf("\n=== 2D Particle Filter Benchmark ===\n");
    printf("Particles:      %d\n", n_particles);
    printf("Ticks:          %d\n", n_ticks);
    printf("Warmup ticks:   %d\n\n", n_warmup_ticks);

    /*========================================================================
     * PHASE 1: Create and Configure Filter
     *========================================================================*/
    PF2D *pf = pf2d_create(n_particles, 4);
    if (!pf)
    {
        fprintf(stderr, "Failed to create particle filter\n");
        return 1;
    }

    /* Configure regimes */
    pf2d_set_regime_params(pf, 0, 0.001, 0.02, log(0.01), 0.05, 0.0); /* Trend */
    pf2d_set_regime_params(pf, 1, 0.0, 0.05, log(0.008), 0.03, 0.0);  /* Mean-revert */
    pf2d_set_regime_params(pf, 2, 0.0, 0.10, log(0.03), 0.10, 0.0);   /* High-vol */
    pf2d_set_regime_params(pf, 3, 0.0, 0.20, log(0.05), 0.20, 0.0);   /* Jump */

    pf2d_real initial_price = 100.0;
    pf2d_real initial_log_vol = log(0.01);
    pf2d_initialize(pf, initial_price, 1.0, initial_log_vol, 0.1);

    pf2d_enable_pcg(pf, 0);
    pf2d_set_resample_adaptive(pf, 0.01);

    pf2d_print_config(pf);

    /* Setup regime probabilities */
    PF2DRegimeProbs rp;
    pf2d_real probs[4] = {0.4, 0.3, 0.2, 0.1};
    pf2d_set_regime_probs(&rp, probs, 4);
    pf2d_build_regime_lut(pf, &rp);

    /*========================================================================
     * PHASE 2: Warmup (Eliminate First-Call Latency)
     *========================================================================*/
    warmup_full(pf, &rp, n_warmup_ticks);

    /* Re-initialize filter state after warmup */
    pf2d_initialize(pf, initial_price, 1.0, initial_log_vol, 0.1);

    printf("\nStarting benchmark...\n\n");

    /*========================================================================
     * PHASE 3: Benchmark
     *========================================================================*/
    MarketState market = {
        .price = initial_price,
        .true_vol = 0.01,
        .true_regime = 0};
    pf2d_pcg32_t sim_rng;
    pf2d_pcg32_seed(&sim_rng, 12345, 0);

    /* Latency tracking */
    LatencyStats latency;
    latency_init(&latency, n_ticks);

    /* Accuracy tracking */
    double price_rmse = 0.0;
    double vol_rmse = 0.0;
    int resample_count = 0;

    /* Main benchmark loop */
    double total_start = get_time_us();

    for (int t = 0; t < n_ticks; t++)
    {
        simulate_tick(&market, &sim_rng);

        double tick_start = get_time_us();
        PF2DOutput out = pf2d_update(pf, market.price, &rp);
        double tick_end = get_time_us();

        latency_add(&latency, tick_end - tick_start);

        double price_err = out.price_mean - market.price;
        double vol_err = out.vol_mean - market.true_vol;
        price_rmse += price_err * price_err;
        vol_rmse += vol_err * vol_err;

        if (out.resampled)
            resample_count++;

        if ((t + 1) % 2000 == 0 || t == 0)
        {
            printf("Tick %5d: price=%.4f (est=%.4f, err=%+.4f) "
                   "vol=%.5f (est=%.5f) ESS=%.0f regime=%d\n",
                   t + 1, (double)market.price, (double)out.price_mean,
                   price_err, (double)market.true_vol, (double)out.vol_mean,
                   (double)out.ess, out.dominant_regime);
        }
    }

    double total_end = get_time_us();
    double total_time_sec = (total_end - total_start) / 1e6;

    /*========================================================================
     * PHASE 4: Results
     *========================================================================*/
    price_rmse = sqrt(price_rmse / n_ticks);
    vol_rmse = sqrt(vol_rmse / n_ticks);

    printf("\n=== Throughput ===\n");
    printf("Total time:       %.3f sec\n", total_time_sec);
    printf("Throughput:       %.0f ticks/sec\n", n_ticks / total_time_sec);
    printf("Mean latency:     %.2f us/tick\n", (total_end - total_start) / n_ticks);

    latency_print(&latency);

    printf("\n=== Accuracy ===\n");
    printf("Price RMSE:       %.6f\n", price_rmse);
    printf("Vol RMSE:         %.6f\n", vol_rmse);
    printf("Resample rate:    %.1f%%\n", 100.0 * resample_count / n_ticks);

    printf("\n=== Final State ===\n");
    printf("True:  price=%.4f  vol=%.6f  regime=%d\n",
           (double)market.price, (double)market.true_vol, market.true_regime);

    PF2DOutput final_out = pf2d_update(pf, market.price, &rp);
    printf("Est:   price=%.4f (%.4f)  vol=%.6f (log=%.4f +/- %.4f)\n",
           (double)final_out.price_mean, sqrt((double)final_out.price_variance),
           (double)final_out.vol_mean, (double)final_out.log_vol_mean,
           sqrt((double)final_out.log_vol_variance));
    printf("ESS:   %.0f / %d (%.1f%%)\n",
           (double)final_out.ess, n_particles,
           100.0 * final_out.ess / n_particles);

    /*========================================================================
     * Cleanup
     *========================================================================*/
    latency_free(&latency);
    pf2d_destroy(pf);

    printf("\nDone.\n");
    return 0;
}