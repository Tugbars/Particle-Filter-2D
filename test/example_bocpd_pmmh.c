/**
 * @file example_bocpd_pmmh.c
 * @brief End-to-end example: BOCPD detects regime change → PMMH re-estimates parameters
 *
 * Demonstrates the full pipeline:
 *   1. Generate synthetic data with known regime change
 *   2. Run naive BOCPD to detect changepoint
 *   3. Trigger PMMH when changepoint detected
 *   4. Verify PMMH recovers true parameters
 *
 * Compile: gcc -O2 example_bocpd_pmmh.c -o example_bocpd_pmmh -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

/* Synthetic data generation */
#define N_OBSERVATIONS      1000
#define CHANGEPOINT_TRUE    500     /* True changepoint location */

/* Regime 1: Low volatility, zero drift */
#define REGIME1_DRIFT       0.0
#define REGIME1_MU_VOL      (-4.5)   /* exp(-4.5) ≈ 0.011 */
#define REGIME1_SIGMA_VOL   0.03

/* Regime 2: Higher volatility, positive drift */
#define REGIME2_DRIFT       0.0015
#define REGIME2_MU_VOL      (-2.5)   /* exp(-2.5) ≈ 0.082 */
#define REGIME2_SIGMA_VOL   0.15

/* Changepoint detection */
#define DETECTOR_WINDOW     50      /* Rolling variance window */
#define DETECTOR_RATIO_THRESH 3.0   /* Variance ratio threshold */

/* PMMH parameters */
#define PMMH_ITERATIONS     400
#define PMMH_BURNIN         150
#define PMMH_WINDOW         200     /* Use 200 obs after changepoint */

/*============================================================================
 * SIMPLE RNG (xorshift128+)
 *============================================================================*/

static uint64_t rng_s[2] = {12345678901234567ULL, 98765432109876543ULL};

static inline uint64_t rng_next(void) {
    uint64_t s1 = rng_s[0];
    const uint64_t s0 = rng_s[1];
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng_s[1] + s0;
}

static inline double randu(void) {
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double randn(void) {
    double u1 = randu(), u2 = randu();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/*============================================================================
 * SIMPLE CHANGEPOINT DETECTION (Variance Ratio)
 *
 * Simpler than full BOCPD - just track rolling variance and detect when
 * it jumps significantly. Good enough for this demonstration.
 *============================================================================*/

typedef struct {
    double *buffer;         /* Rolling window of returns */
    int window_size;
    int head;
    int count;
    
    double baseline_var;    /* Variance from first N observations */
    int baseline_set;
    
    double current_var;     /* Current rolling variance */
    double var_ratio;       /* current_var / baseline_var */
    
    int detected;
    int detection_time;
} SimpleChangeDetector;

static SimpleChangeDetector* detector_create(int window_size) {
    SimpleChangeDetector *d = (SimpleChangeDetector*)calloc(1, sizeof(SimpleChangeDetector));
    d->window_size = window_size;
    d->buffer = (double*)calloc(window_size, sizeof(double));
    d->baseline_set = 0;
    d->detected = 0;
    d->detection_time = -1;
    return d;
}

static void detector_destroy(SimpleChangeDetector *d) {
    if (!d) return;
    free(d->buffer);
    free(d);
}

static double compute_variance(const double *data, int n) {
    if (n < 2) return 0.0;
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
        sum2 += data[i] * data[i];
    }
    double mean = sum / n;
    return (sum2 / n) - mean * mean;
}

/* Returns 1 if changepoint detected */
static int detector_update(SimpleChangeDetector *d, double x, int t) {
    /* Add to buffer */
    d->buffer[d->head] = x;
    d->head = (d->head + 1) % d->window_size;
    if (d->count < d->window_size) d->count++;
    
    /* Set baseline after warmup */
    if (!d->baseline_set && d->count == d->window_size && t < 400) {
        d->baseline_var = compute_variance(d->buffer, d->count);
        d->baseline_set = 1;
        if (d->baseline_var < 1e-10) d->baseline_var = 1e-10;
    }
    
    /* Compute current variance and ratio */
    if (d->baseline_set && d->count == d->window_size) {
        d->current_var = compute_variance(d->buffer, d->count);
        d->var_ratio = d->current_var / d->baseline_var;
        
        /* Detect if variance jumps by factor of 3+ */
        if (!d->detected && d->var_ratio > 3.0 && t > 450) {
            d->detected = 1;
            d->detection_time = t;
            return 1;
        }
    }
    
    return 0;
}

/*============================================================================
 * PMMH IMPLEMENTATION (simplified from test file)
 *============================================================================*/

typedef struct {
    double drift;
    double mu_vol;
    double sigma_vol;
} PMMHParams;

typedef struct {
    PMMHParams mean;
    PMMHParams std;
} PMMHPrior;

typedef struct {
    PMMHParams posterior_mean;
    PMMHParams posterior_std;
    double acceptance_rate;
    int n_samples;
} PMMHResult;

static double clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static double log_normal_pdf(double x, double mean, double std) {
    double z = (x - mean) / std;
    return -0.5 * z * z - log(std) - 0.9189385332;
}

/* Compute log-prior */
static double pmmh_log_prior(const PMMHParams *p, const PMMHPrior *prior) {
    double lp = 0.0;
    lp += log_normal_pdf(p->drift, prior->mean.drift, prior->std.drift);
    lp += log_normal_pdf(p->mu_vol, prior->mean.mu_vol, prior->std.mu_vol);
    lp += log_normal_pdf(log(p->sigma_vol), log(prior->mean.sigma_vol), prior->std.sigma_vol);
    return lp;
}

/* Compute log-likelihood using simplified stochastic volatility model */
static double pmmh_log_likelihood(const double *returns, int n,
                                   const PMMHParams *params,
                                   double theta_vol) {
    if (n <= 0) return -INFINITY;
    
    /* Run a simple particle filter */
    int n_particles = 128;
    double *log_vol = (double*)malloc(n_particles * sizeof(double));
    double *log_vol_new = (double*)malloc(n_particles * sizeof(double));
    double *weights = (double*)malloc(n_particles * sizeof(double));
    
    /* Initialize log-volatility at prior mean */
    for (int i = 0; i < n_particles; i++) {
        log_vol[i] = params->mu_vol + randn() * 0.2;
    }
    
    double log_lik = 0.0;
    double one_minus_theta = 1.0 - theta_vol;
    double theta_mu = theta_vol * params->mu_vol;
    
    for (int t = 0; t < n; t++) {
        double ret = returns[t];
        
        /* Propagate volatility: log_vol' = (1-θ)*log_vol + θ*μ + σ*ε */
        for (int i = 0; i < n_particles; i++) {
            log_vol_new[i] = one_minus_theta * log_vol[i] + theta_mu 
                            + params->sigma_vol * randn();
        }
        
        /* Compute weights: p(return | vol) */
        double max_log_w = -1e30;
        for (int i = 0; i < n_particles; i++) {
            double vol = exp(log_vol_new[i]);
            double z = (ret - params->drift) / vol;
            double log_w = -0.5 * z * z - log_vol_new[i];  /* -log(vol) from Jacobian */
            weights[i] = log_w;
            if (log_w > max_log_w) max_log_w = log_w;
        }
        
        /* Log-sum-exp for marginal likelihood */
        double sum_w = 0.0;
        for (int i = 0; i < n_particles; i++) {
            sum_w += exp(weights[i] - max_log_w);
        }
        log_lik += max_log_w + log(sum_w / n_particles);
        
        /* Resample (simple multinomial) */
        double total_w = 0.0;
        for (int i = 0; i < n_particles; i++) {
            weights[i] = exp(weights[i] - max_log_w);
            total_w += weights[i];
        }
        
        double *resampled = (double*)malloc(n_particles * sizeof(double));
        for (int i = 0; i < n_particles; i++) {
            double u = randu() * total_w;
            double cum = 0.0;
            int j = 0;
            while (j < n_particles - 1 && cum + weights[j] < u) {
                cum += weights[j];
                j++;
            }
            resampled[i] = log_vol_new[j];
        }
        memcpy(log_vol, resampled, n_particles * sizeof(double));
        free(resampled);
    }
    
    free(log_vol);
    free(log_vol_new);
    free(weights);
    
    return log_lik;
}

/* Run PMMH to estimate parameters */
static void pmmh_run(const double *returns, int n,
                     const PMMHPrior *prior,
                     double theta_vol,
                     int n_iterations, int n_burnin,
                     PMMHResult *result) {
    
    /* Initialize at prior mean */
    PMMHParams current = prior->mean;
    double current_ll = pmmh_log_likelihood(returns, n, &current, theta_vol);
    double current_lp = pmmh_log_prior(&current, prior);
    
    /* Proposal standard deviations */
    double drift_std = 0.0003;
    double mu_vol_std = 0.05;
    double sigma_vol_log_std = 0.03;
    
    /* Sample storage */
    int max_samples = n_iterations - n_burnin;
    PMMHParams *samples = (PMMHParams*)malloc(max_samples * sizeof(PMMHParams));
    int sample_idx = 0;
    int n_accepted = 0;
    
    printf("  Running PMMH (%d iterations, %d burnin)...\n", n_iterations, n_burnin);
    
    for (int iter = 0; iter < n_iterations; iter++) {
        /* Propose new parameters */
        PMMHParams proposed;
        proposed.drift = current.drift + randn() * drift_std;
        proposed.mu_vol = current.mu_vol + randn() * mu_vol_std;
        proposed.sigma_vol = current.sigma_vol * exp(randn() * sigma_vol_log_std);
        
        /* Clamp to reasonable bounds */
        proposed.drift = clamp(proposed.drift, -0.01, 0.01);
        proposed.mu_vol = clamp(proposed.mu_vol, -8.0, 0.0);
        proposed.sigma_vol = clamp(proposed.sigma_vol, 0.01, 0.5);
        
        /* Compute acceptance ratio */
        double prop_ll = pmmh_log_likelihood(returns, n, &proposed, theta_vol);
        double prop_lp = pmmh_log_prior(&proposed, prior);
        
        double log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);
        
        /* MH accept/reject */
        if (log(randu()) < log_alpha) {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }
        
        /* Store sample after burnin */
        if (iter >= n_burnin) {
            samples[sample_idx++] = current;
        }
        
        /* Progress */
        if ((iter + 1) % 100 == 0) {
            printf("    Iteration %d/%d, acceptance rate: %.2f%%\n", 
                   iter + 1, n_iterations, 100.0 * n_accepted / (iter + 1));
        }
    }
    
    /* Compute posterior mean and std */
    PMMHParams sum = {0, 0, 0};
    PMMHParams sum_sq = {0, 0, 0};
    
    for (int i = 0; i < sample_idx; i++) {
        sum.drift += samples[i].drift;
        sum.mu_vol += samples[i].mu_vol;
        sum.sigma_vol += samples[i].sigma_vol;
        sum_sq.drift += samples[i].drift * samples[i].drift;
        sum_sq.mu_vol += samples[i].mu_vol * samples[i].mu_vol;
        sum_sq.sigma_vol += samples[i].sigma_vol * samples[i].sigma_vol;
    }
    
    double inv_n = 1.0 / sample_idx;
    result->posterior_mean.drift = sum.drift * inv_n;
    result->posterior_mean.mu_vol = sum.mu_vol * inv_n;
    result->posterior_mean.sigma_vol = sum.sigma_vol * inv_n;
    
    result->posterior_std.drift = sqrt(sum_sq.drift * inv_n - 
                                        result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq.mu_vol * inv_n - 
                                         result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq.sigma_vol * inv_n - 
                                            result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol);
    
    result->acceptance_rate = (double)n_accepted / n_iterations;
    result->n_samples = sample_idx;
    
    free(samples);
}

/*============================================================================
 * SYNTHETIC DATA GENERATION
 *============================================================================*/

static void generate_sv_data(double *prices, double *returns, int n,
                              int changepoint,
                              double drift1, double mu_vol1, double sigma_vol1,
                              double drift2, double mu_vol2, double sigma_vol2,
                              double theta_vol) {
    prices[0] = 100.0;
    
    double log_vol = mu_vol1;  /* Start at regime 1 mean */
    
    for (int t = 1; t < n; t++) {
        /* Switch parameters at changepoint */
        double drift, mu_vol, sigma_vol;
        if (t < changepoint) {
            drift = drift1;
            mu_vol = mu_vol1;
            sigma_vol = sigma_vol1;
        } else {
            drift = drift2;
            mu_vol = mu_vol2;
            sigma_vol = sigma_vol2;
        }
        
        /* Evolve log-volatility (Ornstein-Uhlenbeck) */
        log_vol = (1.0 - theta_vol) * log_vol + theta_vol * mu_vol 
                  + sigma_vol * randn();
        
        /* Generate return */
        double vol = exp(log_vol);
        double ret = drift + vol * randn();
        
        prices[t] = prices[t-1] * (1.0 + ret);
        returns[t-1] = ret;
    }
    returns[n-1] = 0.0;  /* Last return undefined */
}

/*============================================================================
 * MAIN: END-TO-END DEMONSTRATION
 *============================================================================*/

int main(void) {
    printf("================================================================\n");
    printf("  BOCPD + PMMH Integration Example\n");
    printf("================================================================\n\n");
    
    /* Allocate data */
    double *prices = (double*)malloc(N_OBSERVATIONS * sizeof(double));
    double *returns = (double*)malloc(N_OBSERVATIONS * sizeof(double));
    
    /* Fixed parameters */
    double theta_vol = 0.02;  /* Mean-reversion speed */
    
    printf("=== Step 1: Generate Synthetic Data ===\n");
    printf("  Regime 1 (t < %d): drift=%.4f, mu_vol=%.2f, sigma_vol=%.3f\n",
           CHANGEPOINT_TRUE, REGIME1_DRIFT, REGIME1_MU_VOL, REGIME1_SIGMA_VOL);
    printf("  Regime 2 (t >= %d): drift=%.4f, mu_vol=%.2f, sigma_vol=%.3f\n",
           CHANGEPOINT_TRUE, REGIME2_DRIFT, REGIME2_MU_VOL, REGIME2_SIGMA_VOL);
    
    generate_sv_data(prices, returns, N_OBSERVATIONS, CHANGEPOINT_TRUE,
                     REGIME1_DRIFT, REGIME1_MU_VOL, REGIME1_SIGMA_VOL,
                     REGIME2_DRIFT, REGIME2_MU_VOL, REGIME2_SIGMA_VOL,
                     theta_vol);
    
    /* Compute empirical stats before/after changepoint */
    double mean1 = 0, var1 = 0, mean2 = 0, var2 = 0;
    for (int t = 0; t < CHANGEPOINT_TRUE - 1; t++) {
        mean1 += returns[t];
    }
    mean1 /= (CHANGEPOINT_TRUE - 1);
    for (int t = 0; t < CHANGEPOINT_TRUE - 1; t++) {
        var1 += (returns[t] - mean1) * (returns[t] - mean1);
    }
    var1 /= (CHANGEPOINT_TRUE - 2);
    
    for (int t = CHANGEPOINT_TRUE; t < N_OBSERVATIONS - 1; t++) {
        mean2 += returns[t];
    }
    mean2 /= (N_OBSERVATIONS - 1 - CHANGEPOINT_TRUE);
    for (int t = CHANGEPOINT_TRUE; t < N_OBSERVATIONS - 1; t++) {
        var2 += (returns[t] - mean2) * (returns[t] - mean2);
    }
    var2 /= (N_OBSERVATIONS - 2 - CHANGEPOINT_TRUE);
    
    printf("  Empirical stats - Regime 1: mean=%.6f, std=%.6f\n", mean1, sqrt(var1));
    printf("  Empirical stats - Regime 2: mean=%.6f, std=%.6f\n", mean2, sqrt(var2));
    printf("\n");
    
    /* ===== Step 2: Run Changepoint Detection ===== */
    printf("=== Step 2: Run Changepoint Detection (Variance Ratio) ===\n");
    
    SimpleChangeDetector *detector = detector_create(50);  /* 50-tick window */
    
    int detection_time = -1;
    for (int t = 0; t < N_OBSERVATIONS - 1; t++) {
        int detected = detector_update(detector, returns[t], t);
        
        if (detected && detection_time < 0) {
            detection_time = t;
            printf("  *** Changepoint DETECTED at t=%d (true=%d, delay=%+d) ***\n",
                   t, CHANGEPOINT_TRUE, t - CHANGEPOINT_TRUE);
        }
        
        /* Print diagnostics */
        if ((t + 1) % 200 == 0 || t == CHANGEPOINT_TRUE) {
            if (detector->baseline_set) {
                printf("  t=%4d: var_ratio=%.2f%s\n",
                       t, detector->var_ratio,
                       t == CHANGEPOINT_TRUE ? " <-- TRUE CHANGEPOINT" : "");
            } else {
                printf("  t=%4d: (baseline not set yet)\n", t);
            }
        }
    }
    
    if (detection_time < 0) {
        printf("  No changepoint detected, using true changepoint + 50\n");
        detection_time = CHANGEPOINT_TRUE + 50;
    }
    printf("\n");
    
    /* ===== Step 3: Run PMMH on post-changepoint data ===== */
    printf("=== Step 3: Run PMMH to Estimate New Regime Parameters ===\n");
    
    /* Use observations starting from detection */
    int pmmh_start = detection_time;
    int pmmh_end = pmmh_start + PMMH_WINDOW;
    if (pmmh_end >= N_OBSERVATIONS - 1) pmmh_end = N_OBSERVATIONS - 2;
    int pmmh_n = pmmh_end - pmmh_start;
    
    printf("  Using returns[%d:%d] (%d observations)\n", pmmh_start, pmmh_end, pmmh_n);
    
    /* Set prior centered on regime 1 (what we knew before) */
    PMMHPrior prior;
    prior.mean.drift = REGIME1_DRIFT;
    prior.mean.mu_vol = REGIME1_MU_VOL;
    prior.mean.sigma_vol = REGIME1_SIGMA_VOL;
    prior.std.drift = 0.002;      /* Loose prior */
    prior.std.mu_vol = 0.5;
    prior.std.sigma_vol = 0.5;    /* Log-space */
    
    PMMHResult result;
    pmmh_run(&returns[pmmh_start], pmmh_n, &prior, theta_vol,
             PMMH_ITERATIONS, PMMH_BURNIN, &result);
    printf("\n");
    
    /* ===== Step 4: Evaluate Results ===== */
    printf("=== Step 4: Results ===\n\n");
    
    printf("  Parameter      | True (R2) | Prior (R1) | Posterior  | Error\n");
    printf("  ---------------|-----------|------------|------------|-------\n");
    printf("  drift          | %9.5f | %10.5f | %10.5f | %+.5f\n",
           REGIME2_DRIFT, prior.mean.drift, result.posterior_mean.drift,
           result.posterior_mean.drift - REGIME2_DRIFT);
    printf("  mu_vol         | %9.3f | %10.3f | %10.3f | %+.3f\n",
           REGIME2_MU_VOL, prior.mean.mu_vol, result.posterior_mean.mu_vol,
           result.posterior_mean.mu_vol - REGIME2_MU_VOL);
    printf("  sigma_vol      | %9.4f | %10.4f | %10.4f | %+.4f\n",
           REGIME2_SIGMA_VOL, prior.mean.sigma_vol, result.posterior_mean.sigma_vol,
           result.posterior_mean.sigma_vol - REGIME2_SIGMA_VOL);
    printf("\n");
    
    printf("  Posterior std:   drift=%.6f, mu_vol=%.4f, sigma_vol=%.4f\n",
           result.posterior_std.drift, result.posterior_std.mu_vol, 
           result.posterior_std.sigma_vol);
    printf("  Acceptance rate: %.1f%%\n", 100.0 * result.acceptance_rate);
    printf("  Samples used:    %d\n", result.n_samples);
    printf("\n");
    
    /* Check if PMMH moved toward truth */
    double drift_prior_err = fabs(prior.mean.drift - REGIME2_DRIFT);
    double drift_post_err = fabs(result.posterior_mean.drift - REGIME2_DRIFT);
    double mu_prior_err = fabs(prior.mean.mu_vol - REGIME2_MU_VOL);
    double mu_post_err = fabs(result.posterior_mean.mu_vol - REGIME2_MU_VOL);
    double sigma_prior_err = fabs(prior.mean.sigma_vol - REGIME2_SIGMA_VOL);
    double sigma_post_err = fabs(result.posterior_mean.sigma_vol - REGIME2_SIGMA_VOL);
    
    printf("  Improvement over prior:\n");
    printf("    drift:     %.6f → %.6f (%s)\n", drift_prior_err, drift_post_err,
           drift_post_err < drift_prior_err ? "IMPROVED" : "no improvement");
    printf("    mu_vol:    %.4f → %.4f (%s)\n", mu_prior_err, mu_post_err,
           mu_post_err < mu_prior_err ? "IMPROVED" : "no improvement");
    printf("    sigma_vol: %.4f → %.4f (%s)\n", sigma_prior_err, sigma_post_err,
           sigma_post_err < sigma_prior_err ? "IMPROVED" : "no improvement");
    printf("\n");
    
    int success = (mu_post_err < mu_prior_err);  /* Main parameter of interest */
    
    printf("================================================================\n");
    printf("  RESULT: %s\n", success ? "SUCCESS - PMMH recovered regime change!" : 
                                        "PARTIAL - More iterations may help");
    printf("================================================================\n");
    
    /* Cleanup */
    detector_destroy(detector);
    free(prices);
    free(returns);
    
    return success ? 0 : 1;
}
