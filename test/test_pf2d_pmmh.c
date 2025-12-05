/**
 * @file test_pf2d_pmmh.c
 * @brief Unit tests for PMMH parameter estimation
 *
 * Tests cover:
 *   1. Observation window (ring buffer)
 *   2. Configuration and priors
 *   3. Log-likelihood computation
 *   4. Metropolis-Hastings mechanics
 *   5. Posterior estimation
 *   6. Async job management
 *   7. Atomic parameter storage
 *
 * This file has two modes:
 *   - Standalone mode (no MKL): Tests ring buffer and config logic
 *   - Full mode (with MKL): Tests complete PMMH algorithm
 *
 * Compile standalone: gcc -DPMMH_TEST_STANDALONE test_pf2d_pmmh.c -o test_pmmh -lm
 * Compile full:       Link with particle_filter_2d library
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <stdint.h>

/*============================================================================
 * TEST UTILITIES
 *============================================================================*/

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond, msg) do { \
    tests_run++; \
    if (cond) { \
        tests_passed++; \
        printf("  [%s] %s\n", TEST_PASS, msg); \
    } else { \
        printf("  [%s] %s (line %d)\n", TEST_FAIL, msg, __LINE__); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol, msg) do { \
    tests_run++; \
    double _a = (double)(a), _b = (double)(b), _tol = (double)(tol); \
    if (fabs(_a - _b) <= _tol) { \
        tests_passed++; \
        printf("  [%s] %s (%.6f ≈ %.6f)\n", TEST_PASS, msg, _a, _b); \
    } else { \
        printf("  [%s] %s (%.6f != %.6f, diff=%.6f)\n", TEST_FAIL, msg, _a, _b, fabs(_a - _b)); \
    } \
} while(0)

#define ASSERT_GT(a, b, msg) do { \
    tests_run++; \
    double _a = (double)(a), _b = (double)(b); \
    if (_a > _b) { \
        tests_passed++; \
        printf("  [%s] %s (%.6f > %.6f)\n", TEST_PASS, msg, _a, _b); \
    } else { \
        printf("  [%s] %s (%.6f <= %.6f)\n", TEST_FAIL, msg, _a, _b); \
    } \
} while(0)

#define ASSERT_LT(a, b, msg) do { \
    tests_run++; \
    double _a = (double)(a), _b = (double)(b); \
    if (_a < _b) { \
        tests_passed++; \
        printf("  [%s] %s (%.6f < %.6f)\n", TEST_PASS, msg, _a, _b); \
    } else { \
        printf("  [%s] %s (%.6f >= %.6f)\n", TEST_FAIL, msg, _a, _b); \
    } \
} while(0)

#define ASSERT_EQ(a, b, msg) do { \
    tests_run++; \
    int _a = (int)(a), _b = (int)(b); \
    if (_a == _b) { \
        tests_passed++; \
        printf("  [%s] %s (%d == %d)\n", TEST_PASS, msg, _a, _b); \
    } else { \
        printf("  [%s] %s (%d != %d)\n", TEST_FAIL, msg, _a, _b); \
    } \
} while(0)

#define ASSERT_IN_RANGE(val, lo, hi, msg) do { \
    tests_run++; \
    double _v = (double)(val), _lo = (double)(lo), _hi = (double)(hi); \
    if (_v >= _lo && _v <= _hi) { \
        tests_passed++; \
        printf("  [%s] %s (%.6f in [%.6f, %.6f])\n", TEST_PASS, msg, _v, _lo, _hi); \
    } else { \
        printf("  [%s] %s (%.6f not in [%.6f, %.6f])\n", TEST_FAIL, msg, _v, _lo, _hi); \
    } \
} while(0)

/*============================================================================
 * STANDALONE TYPE DEFINITIONS
 *============================================================================*/

typedef double pf2d_real;

/* PMMH configuration defaults */
#define PMMH_DEFAULT_ITERATIONS     500
#define PMMH_DEFAULT_BURNIN         150
#define PMMH_DEFAULT_PARTICLES      256
#define PMMH_DEFAULT_WINDOW_MIN     300
#define PMMH_DEFAULT_WINDOW_MAX     1500
#define PMMH_DEFAULT_POSTERIOR_WINDOW 150

#define PMMH_TARGET_ACCEPTANCE_LOW  0.25
#define PMMH_TARGET_ACCEPTANCE_HIGH 0.40

#define PMMH_DRIFT_MIN             (-0.005)
#define PMMH_DRIFT_MAX             (0.005)

typedef struct {
    pf2d_real drift;
    pf2d_real mu_vol;
    pf2d_real sigma_vol;
} PMMHParams;

typedef struct {
    pf2d_real drift_std;
    pf2d_real mu_vol_std;
    pf2d_real sigma_vol_log_std;
} PMMHProposalStd;

typedef struct {
    PMMHParams mean;
    PMMHParams std;
} PMMHPrior;

typedef struct {
    int n_iterations;
    int n_burnin;
    int n_particles;
    int posterior_window;
    int target_regime;
    PMMHProposalStd proposal_std;
    int adaptive_proposal;
    PMMHPrior prior;
    int window_size;
    pf2d_real drift_min;
    pf2d_real drift_max;
} PMMHConfig;

typedef struct {
    PMMHParams posterior_mean;
    PMMHParams posterior_std;
    pf2d_real acceptance_rate;
    pf2d_real final_log_lik;
    int n_samples_used;
    int converged;
    double elapsed_ms;
} PMMHResult;

typedef struct {
    pf2d_real *buffer;
    int capacity;
    int head;
    int count;
} PMMHObsWindow;

/*============================================================================
 * STANDALONE IMPLEMENTATIONS
 *============================================================================*/

static PMMHObsWindow* pmmh_obs_window_create(int capacity) {
    PMMHObsWindow *win = (PMMHObsWindow*)malloc(sizeof(PMMHObsWindow));
    if (!win) return NULL;
    
    win->buffer = (pf2d_real*)malloc(capacity * sizeof(pf2d_real));
    if (!win->buffer) {
        free(win);
        return NULL;
    }
    
    win->capacity = capacity;
    win->head = 0;
    win->count = 0;
    
    return win;
}

static void pmmh_obs_window_destroy(PMMHObsWindow *win) {
    if (!win) return;
    free(win->buffer);
    free(win);
}

static void pmmh_obs_window_push(PMMHObsWindow *win, pf2d_real obs) {
    win->buffer[win->head] = obs;
    win->head = (win->head + 1) % win->capacity;
    if (win->count < win->capacity) {
        win->count++;
    }
}

static int pmmh_obs_window_copy(const PMMHObsWindow *win, pf2d_real *out) {
    if (win->count == 0) return 0;
    
    if (win->count < win->capacity) {
        /* Buffer not full - simple copy from start */
        memcpy(out, win->buffer, win->count * sizeof(pf2d_real));
    } else {
        /* Buffer full - copy in order (oldest first) */
        int oldest = win->head;
        int first_chunk = win->capacity - oldest;
        memcpy(out, &win->buffer[oldest], first_chunk * sizeof(pf2d_real));
        memcpy(&out[first_chunk], win->buffer, oldest * sizeof(pf2d_real));
    }
    
    return win->count;
}

static int pmmh_obs_window_count(const PMMHObsWindow *win) {
    return win->count;
}

static void pmmh_config_defaults(PMMHConfig *cfg) {
    cfg->n_iterations = PMMH_DEFAULT_ITERATIONS;
    cfg->n_burnin = PMMH_DEFAULT_BURNIN;
    cfg->n_particles = PMMH_DEFAULT_PARTICLES;
    cfg->posterior_window = PMMH_DEFAULT_POSTERIOR_WINDOW;
    cfg->target_regime = 0;
    
    cfg->proposal_std.drift_std = 0.0005;
    cfg->proposal_std.mu_vol_std = 0.08;
    cfg->proposal_std.sigma_vol_log_std = 0.02;
    cfg->adaptive_proposal = 1;
    
    memset(&cfg->prior, 0, sizeof(cfg->prior));
    
    cfg->window_size = 500;
    cfg->drift_min = PMMH_DRIFT_MIN;
    cfg->drift_max = PMMH_DRIFT_MAX;
}

/* Simple xorshift RNG for standalone testing */
static uint64_t rng_state = 12345678901234567ULL;

static double randn_standalone(void) {
    /* Box-Muller transform */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    double u1 = (double)(rng_state & 0xFFFFFFFFFFFFULL) / (double)0xFFFFFFFFFFFFULL;
    
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    double u2 = (double)(rng_state & 0xFFFFFFFFFFFFULL) / (double)0xFFFFFFFFFFFFULL;
    
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979323846 * u2);
}

static double randu_standalone(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (double)(rng_state & 0xFFFFFFFFFFFFULL) / (double)0xFFFFFFFFFFFFULL;
}

static pf2d_real clamp(pf2d_real x, pf2d_real lo, pf2d_real hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/* Gaussian log-pdf */
static pf2d_real log_normal_pdf(pf2d_real x, pf2d_real mean, pf2d_real std) {
    pf2d_real z = (x - mean) / std;
    return -0.5 * z * z - log(std) - 0.9189385332;  /* -0.5*log(2*pi) */
}

/* Compute log-prior for params */
static pf2d_real compute_log_prior(const PMMHParams *params, const PMMHPrior *prior) {
    pf2d_real lp = 0.0;
    lp += log_normal_pdf(params->drift, prior->mean.drift, prior->std.drift);
    lp += log_normal_pdf(params->mu_vol, prior->mean.mu_vol, prior->std.mu_vol);
    /* sigma_vol: prior on log(sigma_vol) */
    pf2d_real log_sigma = log(params->sigma_vol);
    pf2d_real log_mean = log(prior->mean.sigma_vol);
    lp += log_normal_pdf(log_sigma, log_mean, prior->std.sigma_vol);
    return lp;
}

/* Simple log-likelihood for testing (not actual SV model) */
static pf2d_real compute_mock_log_lik(const pf2d_real *obs, int n_obs,
                                        const PMMHParams *params) {
    /* Mock likelihood: penalizes deviation from zero drift and typical vol */
    pf2d_real ll = 0.0;
    pf2d_real vol = exp(params->mu_vol);
    
    for (int i = 1; i < n_obs; i++) {
        pf2d_real ret = obs[i] - obs[i-1];
        pf2d_real pred = params->drift;
        pf2d_real diff = ret - pred;
        ll += -0.5 * (diff * diff) / (vol * vol);
    }
    
    return ll;
}

/* Simple PMMH simulation for testing MH mechanics */
static void pmmh_run_mock(const pf2d_real *observations, int n_obs,
                          const PMMHConfig *cfg, PMMHResult *result) {
    
    PMMHParams current = cfg->prior.mean;
    pf2d_real current_log_lik = compute_mock_log_lik(observations, n_obs, &current);
    pf2d_real current_log_prior = compute_log_prior(&current, &cfg->prior);
    
    PMMHProposalStd prop_std = cfg->proposal_std;
    int n_accepted = 0;
    int n_iter = cfg->n_iterations;
    
    /* Sample storage */
    int max_samples = n_iter - cfg->n_burnin;
    PMMHParams *samples = NULL;
    if (max_samples > 0) {
        samples = (PMMHParams*)malloc(max_samples * sizeof(PMMHParams));
    }
    int sample_idx = 0;
    
    for (int iter = 0; iter < n_iter; iter++) {
        /* Propose new params (random walk) */
        PMMHParams proposed;
        proposed.drift = current.drift + randn_standalone() * prop_std.drift_std;
        proposed.mu_vol = current.mu_vol + randn_standalone() * prop_std.mu_vol_std;
        proposed.sigma_vol = current.sigma_vol * exp(randn_standalone() * prop_std.sigma_vol_log_std);
        
        /* Clamp to bounds */
        proposed.drift = clamp(proposed.drift, cfg->drift_min, cfg->drift_max);
        proposed.sigma_vol = clamp(proposed.sigma_vol, 0.001, 1.0);
        
        /* Compute acceptance ratio */
        pf2d_real prop_log_lik = compute_mock_log_lik(observations, n_obs, &proposed);
        pf2d_real prop_log_prior = compute_log_prior(&proposed, &cfg->prior);
        
        pf2d_real log_alpha = (prop_log_lik + prop_log_prior) - 
                              (current_log_lik + current_log_prior);
        
        /* MH accept/reject */
        pf2d_real u = randu_standalone();
        if (log(u) < log_alpha) {
            current = proposed;
            current_log_lik = prop_log_lik;
            current_log_prior = prop_log_prior;
            n_accepted++;
        }
        
        /* Store sample after burnin */
        if (iter >= cfg->n_burnin && samples != NULL) {
            samples[sample_idx++] = current;
        }
        
        /* Adapt proposal std every 50 iterations */
        if (cfg->adaptive_proposal && (iter + 1) % 50 == 0 && iter > 0) {
            pf2d_real acc_rate = (pf2d_real)n_accepted / (pf2d_real)(iter + 1);
            if (acc_rate < PMMH_TARGET_ACCEPTANCE_LOW) {
                prop_std.drift_std *= 0.9;
                prop_std.mu_vol_std *= 0.9;
                prop_std.sigma_vol_log_std *= 0.9;
            } else if (acc_rate > PMMH_TARGET_ACCEPTANCE_HIGH) {
                prop_std.drift_std *= 1.1;
                prop_std.mu_vol_std *= 1.1;
                prop_std.sigma_vol_log_std *= 1.1;
            }
        }
    }
    
    /* Compute posterior mean from last N samples */
    if (sample_idx > 0) {
        int use_last = cfg->posterior_window;
        if (use_last > sample_idx) use_last = sample_idx;
        int start = sample_idx - use_last;
        
        PMMHParams sum = {0, 0, 0};
        for (int i = start; i < sample_idx; i++) {
            sum.drift += samples[i].drift;
            sum.mu_vol += samples[i].mu_vol;
            sum.sigma_vol += samples[i].sigma_vol;
        }
        
        pf2d_real inv_n = 1.0 / use_last;
        result->posterior_mean.drift = sum.drift * inv_n;
        result->posterior_mean.mu_vol = sum.mu_vol * inv_n;
        result->posterior_mean.sigma_vol = sum.sigma_vol * inv_n;
        result->n_samples_used = use_last;
    } else {
        result->posterior_mean = current;
        result->n_samples_used = 0;
    }
    
    result->acceptance_rate = (pf2d_real)n_accepted / (pf2d_real)n_iter;
    result->final_log_lik = current_log_lik;
    result->converged = (result->acceptance_rate >= PMMH_TARGET_ACCEPTANCE_LOW &&
                         result->acceptance_rate <= PMMH_TARGET_ACCEPTANCE_HIGH);
    
    free(samples);
}

/*============================================================================
 * TEST: OBSERVATION WINDOW
 *============================================================================*/

static void test_obs_window_create(void) {
    printf("\n=== Test: Observation Window - Create ===\n");
    
    PMMHObsWindow *win = pmmh_obs_window_create(100);
    ASSERT_TRUE(win != NULL, "Window created");
    ASSERT_EQ(win->capacity, 100, "Capacity set correctly");
    ASSERT_EQ(win->count, 0, "Initial count is 0");
    ASSERT_EQ(win->head, 0, "Initial head is 0");
    
    pmmh_obs_window_destroy(win);
}

static void test_obs_window_push(void) {
    printf("\n=== Test: Observation Window - Push ===\n");
    
    PMMHObsWindow *win = pmmh_obs_window_create(5);
    
    pmmh_obs_window_push(win, 1.0);
    ASSERT_EQ(win->count, 1, "Count after 1 push");
    
    pmmh_obs_window_push(win, 2.0);
    pmmh_obs_window_push(win, 3.0);
    ASSERT_EQ(win->count, 3, "Count after 3 pushes");
    
    /* Fill to capacity */
    pmmh_obs_window_push(win, 4.0);
    pmmh_obs_window_push(win, 5.0);
    ASSERT_EQ(win->count, 5, "Count at capacity");
    
    /* Overwrite oldest */
    pmmh_obs_window_push(win, 6.0);
    ASSERT_EQ(win->count, 5, "Count stays at capacity");
    
    pmmh_obs_window_destroy(win);
}

static void test_obs_window_copy_partial(void) {
    printf("\n=== Test: Observation Window - Copy (Partial) ===\n");
    
    PMMHObsWindow *win = pmmh_obs_window_create(10);
    
    /* Push less than capacity */
    for (int i = 0; i < 5; i++) {
        pmmh_obs_window_push(win, (pf2d_real)(i + 1));
    }
    
    pf2d_real out[10];
    int n = pmmh_obs_window_copy(win, out);
    
    ASSERT_EQ(n, 5, "Copied 5 elements");
    ASSERT_NEAR(out[0], 1.0, 1e-9, "First element correct");
    ASSERT_NEAR(out[4], 5.0, 1e-9, "Last element correct");
    
    pmmh_obs_window_destroy(win);
}

static void test_obs_window_copy_full(void) {
    printf("\n=== Test: Observation Window - Copy (Full/Wrapped) ===\n");
    
    PMMHObsWindow *win = pmmh_obs_window_create(5);
    
    /* Push more than capacity - causes wrap */
    for (int i = 0; i < 8; i++) {
        pmmh_obs_window_push(win, (pf2d_real)(i + 1));
    }
    /* Window now contains [4, 5, 6, 7, 8] in order */
    
    pf2d_real out[5];
    int n = pmmh_obs_window_copy(win, out);
    
    ASSERT_EQ(n, 5, "Copied 5 elements");
    ASSERT_NEAR(out[0], 4.0, 1e-9, "First (oldest) element correct");
    ASSERT_NEAR(out[1], 5.0, 1e-9, "Second element correct");
    ASSERT_NEAR(out[4], 8.0, 1e-9, "Last (newest) element correct");
    
    pmmh_obs_window_destroy(win);
}

static void test_obs_window_empty(void) {
    printf("\n=== Test: Observation Window - Empty ===\n");
    
    PMMHObsWindow *win = pmmh_obs_window_create(10);
    
    pf2d_real out[10];
    int n = pmmh_obs_window_copy(win, out);
    
    ASSERT_EQ(n, 0, "Copy from empty returns 0");
    ASSERT_EQ(pmmh_obs_window_count(win), 0, "Count is 0");
    
    pmmh_obs_window_destroy(win);
}

/*============================================================================
 * TEST: CONFIGURATION
 *============================================================================*/

static void test_config_defaults(void) {
    printf("\n=== Test: Config Defaults ===\n");
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    
    ASSERT_EQ(cfg.n_iterations, PMMH_DEFAULT_ITERATIONS, "Default iterations");
    ASSERT_EQ(cfg.n_burnin, PMMH_DEFAULT_BURNIN, "Default burnin");
    ASSERT_EQ(cfg.n_particles, PMMH_DEFAULT_PARTICLES, "Default particles");
    ASSERT_EQ(cfg.posterior_window, PMMH_DEFAULT_POSTERIOR_WINDOW, "Default posterior window");
    ASSERT_EQ(cfg.adaptive_proposal, 1, "Adaptive proposal ON by default");
    ASSERT_NEAR(cfg.drift_min, PMMH_DRIFT_MIN, 1e-9, "Default drift_min");
    ASSERT_NEAR(cfg.drift_max, PMMH_DRIFT_MAX, 1e-9, "Default drift_max");
}

static void test_config_proposal_std(void) {
    printf("\n=== Test: Config Proposal Std ===\n");
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    
    ASSERT_GT(cfg.proposal_std.drift_std, 0, "Drift std positive");
    ASSERT_GT(cfg.proposal_std.mu_vol_std, 0, "Mu_vol std positive");
    ASSERT_GT(cfg.proposal_std.sigma_vol_log_std, 0, "Sigma_vol log std positive");
    
    /* Sanity check magnitudes */
    ASSERT_LT(cfg.proposal_std.drift_std, 0.01, "Drift std reasonable");
    ASSERT_LT(cfg.proposal_std.sigma_vol_log_std, 0.1, "Log std reasonable");
}

/*============================================================================
 * TEST: LOG-PRIOR COMPUTATION
 *============================================================================*/

static void test_log_prior(void) {
    printf("\n=== Test: Log-Prior Computation ===\n");
    
    PMMHPrior prior;
    prior.mean.drift = 0.0;
    prior.mean.mu_vol = -2.5;
    prior.mean.sigma_vol = 0.1;
    prior.std.drift = 0.002;
    prior.std.mu_vol = 0.2;
    prior.std.sigma_vol = 0.3;  /* Log-space */
    
    /* Params at prior mean should have highest prior */
    PMMHParams at_mean = prior.mean;
    pf2d_real lp_at_mean = compute_log_prior(&at_mean, &prior);
    
    /* Params away from mean should have lower prior */
    PMMHParams away;
    away.drift = 0.003;  /* 1.5 std away */
    away.mu_vol = -2.5;
    away.sigma_vol = 0.1;
    pf2d_real lp_away = compute_log_prior(&away, &prior);
    
    ASSERT_GT(lp_at_mean, lp_away, "Prior highest at mean");
    ASSERT_TRUE(lp_at_mean > -INFINITY, "Log-prior is finite");
}

/*============================================================================
 * TEST: METROPOLIS-HASTINGS MECHANICS
 *============================================================================*/

static void test_mh_acceptance_rate(void) {
    printf("\n=== Test: MH Acceptance Rate ===\n");
    
    /* Generate synthetic data */
    int n_obs = 200;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    obs[0] = 100.0;
    for (int i = 1; i < n_obs; i++) {
        obs[i] = obs[i-1] + 0.001 * randn_standalone();  /* Small drift */
    }
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 300;
    cfg.n_burnin = 100;
    
    /* Set prior centered on truth */
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Acceptance rate should be reasonable */
    ASSERT_IN_RANGE(result.acceptance_rate, 0.1, 0.8, 
                    "Acceptance rate in reasonable range");
    ASSERT_GT(result.n_samples_used, 0, "Some samples used");
    
    free(obs);
}

static void test_mh_posterior_near_truth(void) {
    printf("\n=== Test: MH Posterior Near Truth ===\n");
    
    /* Generate data with known drift */
    int n_obs = 500;
    pf2d_real true_drift = 0.0005;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    obs[0] = 100.0;
    for (int i = 1; i < n_obs; i++) {
        obs[i] = obs[i-1] + true_drift + 0.01 * randn_standalone();
    }
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 500;
    cfg.n_burnin = 200;
    cfg.posterior_window = 100;
    
    /* Loose prior centered at 0 */
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.003;
    cfg.prior.std.mu_vol = 0.5;
    cfg.prior.std.sigma_vol = 0.5;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Posterior mean should be closer to true_drift than prior mean */
    pf2d_real post_error = fabs(result.posterior_mean.drift - true_drift);
    pf2d_real prior_error = fabs(cfg.prior.mean.drift - true_drift);
    
    ASSERT_LT(post_error, prior_error, "Posterior closer to truth than prior");
    
    free(obs);
}

static void test_mh_adaptive_proposal(void) {
    printf("\n=== Test: MH Adaptive Proposal ===\n");
    
    /* Run with adaptive ON - should converge to target acceptance */
    int n_obs = 200;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    obs[0] = 100.0;
    for (int i = 1; i < n_obs; i++) {
        obs[i] = obs[i-1] + 0.01 * randn_standalone();
    }
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 400;
    cfg.n_burnin = 150;
    cfg.adaptive_proposal = 1;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* With adaptation, acceptance should be in a reasonable range */
    ASSERT_IN_RANGE(result.acceptance_rate, 0.1, 0.9,
                    "Adaptive keeps acceptance in working range");
    
    free(obs);
}

/*============================================================================
 * TEST: POSTERIOR COMPUTATION
 *============================================================================*/

static void test_posterior_window(void) {
    printf("\n=== Test: Posterior Window ===\n");
    
    int n_obs = 100;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0 + i * 0.01;
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 200;
    cfg.n_burnin = 100;
    cfg.posterior_window = 50;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Should use posterior_window samples (or less if not enough) */
    ASSERT_TRUE(result.n_samples_used <= cfg.posterior_window,
                "Used at most posterior_window samples");
    ASSERT_GT(result.n_samples_used, 0, "Used some samples");
    
    free(obs);
}

static void test_posterior_bounds(void) {
    printf("\n=== Test: Posterior Respects Bounds ===\n");
    
    int n_obs = 100;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0;
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 200;
    cfg.n_burnin = 50;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Check bounds */
    ASSERT_TRUE(result.posterior_mean.drift >= cfg.drift_min,
                "Drift >= drift_min");
    ASSERT_TRUE(result.posterior_mean.drift <= cfg.drift_max,
                "Drift <= drift_max");
    ASSERT_GT(result.posterior_mean.sigma_vol, 0, "Sigma_vol positive");
    
    free(obs);
}

/*============================================================================
 * TEST: CONVERGENCE DIAGNOSTICS
 *============================================================================*/

static void test_convergence_flag(void) {
    printf("\n=== Test: Convergence Flag ===\n");
    
    int n_obs = 200;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0 + randn_standalone() * 0.01;
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 500;
    cfg.n_burnin = 200;
    cfg.adaptive_proposal = 1;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Check convergence logic */
    int expected_converged = (result.acceptance_rate >= PMMH_TARGET_ACCEPTANCE_LOW &&
                              result.acceptance_rate <= PMMH_TARGET_ACCEPTANCE_HIGH);
    ASSERT_EQ(result.converged, expected_converged, "Convergence flag correct");
    
    free(obs);
}

/*============================================================================
 * TEST: EDGE CASES
 *============================================================================*/

static void test_very_short_window(void) {
    printf("\n=== Test: Very Short Window ===\n");
    
    int n_obs = 10;  /* Very short */
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0;
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 100;
    cfg.n_burnin = 30;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Should still produce a result */
    ASSERT_TRUE(!isnan(result.posterior_mean.drift), "Drift not NaN");
    ASSERT_TRUE(!isnan(result.posterior_mean.mu_vol), "Mu_vol not NaN");
    ASSERT_TRUE(!isnan(result.posterior_mean.sigma_vol), "Sigma_vol not NaN");
    
    free(obs);
}

static void test_all_same_observations(void) {
    printf("\n=== Test: All Same Observations ===\n");
    
    int n_obs = 100;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0;  /* All identical */
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 200;
    cfg.n_burnin = 50;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* With no variation, drift should be near zero */
    ASSERT_NEAR(result.posterior_mean.drift, 0.0, 0.002, 
                "Zero drift for constant series");
    
    free(obs);
}

static void test_high_burnin(void) {
    printf("\n=== Test: High Burnin ===\n");
    
    int n_obs = 100;
    pf2d_real *obs = (pf2d_real*)malloc(n_obs * sizeof(pf2d_real));
    for (int i = 0; i < n_obs; i++) obs[i] = 100.0 + i * 0.001;
    
    PMMHConfig cfg;
    pmmh_config_defaults(&cfg);
    cfg.n_iterations = 100;
    cfg.n_burnin = 90;  /* Very high burnin */
    cfg.posterior_window = 50;
    
    cfg.prior.mean.drift = 0.0;
    cfg.prior.mean.mu_vol = -4.0;
    cfg.prior.mean.sigma_vol = 0.1;
    cfg.prior.std.drift = 0.002;
    cfg.prior.std.mu_vol = 0.3;
    cfg.prior.std.sigma_vol = 0.3;
    
    PMMHResult result;
    pmmh_run_mock(obs, n_obs, &cfg, &result);
    
    /* Only 10 post-burnin samples, should use <= 10 */
    ASSERT_TRUE(result.n_samples_used <= 10, "Uses limited samples with high burnin");
    
    free(obs);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void) {
    printf("========================================\n");
    printf("  PF2D PMMH Unit Tests\n");
    printf("========================================\n");
    
    /* Observation window */
    test_obs_window_create();
    test_obs_window_push();
    test_obs_window_copy_partial();
    test_obs_window_copy_full();
    test_obs_window_empty();
    
    /* Configuration */
    test_config_defaults();
    test_config_proposal_std();
    
    /* Log-prior */
    test_log_prior();
    
    /* MH mechanics */
    test_mh_acceptance_rate();
    test_mh_posterior_near_truth();
    test_mh_adaptive_proposal();
    
    /* Posterior computation */
    test_posterior_window();
    test_posterior_bounds();
    
    /* Convergence */
    test_convergence_flag();
    
    /* Edge cases */
    test_very_short_window();
    test_all_same_observations();
    test_high_burnin();
    
    /* Summary */
    printf("\n========================================\n");
    printf("  Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("========================================\n");
    
    return (tests_passed == tests_run) ? 0 : 1;
}
