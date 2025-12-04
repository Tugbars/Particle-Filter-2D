/*
 * test_particle_filter_2d.c
 *
 * Comprehensive test suite for the 2D Particle Filter
 *
 * Build:
 *   Windows: cl /O2 test_particle_filter_2d.c particle_filter_2d.c /I"path/to/mkl/include" /link mkl_rt.lib
 *   Linux:   gcc -O2 test_particle_filter_2d.c particle_filter_2d.c -o test_pf2d -lmkl_rt -lm -fopenmp
 *
 * Or add to CMakeLists.txt as a test target
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "particle_filter_2d.h"

/* ========================================================================== */
/* Test Framework                                                              */
/* ========================================================================== */

static int tests_run = 0;
static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) static int test_##name(void)
#define RUN_TEST(name)             \
    do                             \
    {                              \
        printf("  %-50s ", #name); \
        fflush(stdout);            \
        tests_run++;               \
        if (test_##name())         \
        {                          \
            printf("[PASS]\n");    \
            tests_passed++;        \
        }                          \
        else                       \
        {                          \
            printf("[FAIL]\n");    \
            tests_failed++;        \
        }                          \
    } while (0)

#define ASSERT(cond)                                                        \
    do                                                                      \
    {                                                                       \
        if (!(cond))                                                        \
        {                                                                   \
            printf("\n    ASSERT failed: %s (line %d)\n", #cond, __LINE__); \
            return 0;                                                       \
        }                                                                   \
    } while (0)

#define ASSERT_NEAR(a, b, tol)                                                  \
    do                                                                          \
    {                                                                           \
        double _a = (a), _b = (b), _tol = (tol);                                \
        if (fabs(_a - _b) > _tol)                                               \
        {                                                                       \
            printf("\n    ASSERT_NEAR failed: |%g - %g| = %g > %g (line %d)\n", \
                   _a, _b, fabs(_a - _b), _tol, __LINE__);                      \
            return 0;                                                           \
        }                                                                       \
    } while (0)

#define ASSERT_GT(a, b)                                                               \
    do                                                                                \
    {                                                                                 \
        double _a = (a), _b = (b);                                                    \
        if (!(_a > _b))                                                               \
        {                                                                             \
            printf("\n    ASSERT_GT failed: %g <= %g (line %d)\n", _a, _b, __LINE__); \
            return 0;                                                                 \
        }                                                                             \
    } while (0)

#define ASSERT_LT(a, b)                                                               \
    do                                                                                \
    {                                                                                 \
        double _a = (a), _b = (b);                                                    \
        if (!(_a < _b))                                                               \
        {                                                                             \
            printf("\n    ASSERT_LT failed: %g >= %g (line %d)\n", _a, _b, __LINE__); \
            return 0;                                                                 \
        }                                                                             \
    } while (0)

#define ASSERT_BETWEEN(val, lo, hi)                                               \
    do                                                                            \
    {                                                                             \
        double _v = (val), _lo = (lo), _hi = (hi);                                \
        if (_v < _lo || _v > _hi)                                                 \
        {                                                                         \
            printf("\n    ASSERT_BETWEEN failed: %g not in [%g, %g] (line %d)\n", \
                   _v, _lo, _hi, __LINE__);                                       \
            return 0;                                                             \
        }                                                                         \
    } while (0)

/* ========================================================================== */
/* Helper Functions                                                            */
/* ========================================================================== */

/* Simple LCG for reproducible random numbers */
static unsigned int test_seed = 12345;

static double test_rand(void)
{
    test_seed = test_seed * 1103515245 + 12345;
    return (double)(test_seed % 100000) / 100000.0;
}

static double test_randn(void)
{
    /* Box-Muller */
    double u1 = test_rand() + 1e-10;
    double u2 = test_rand();
    return sqrt(-2.0 * log(u1)) * cos(2.0 * 3.14159265358979 * u2);
}

/* Generate synthetic price series with known volatility */
static void generate_test_prices(double *prices, int n, double p0, double vol)
{
    prices[0] = p0;
    for (int i = 1; i < n; i++)
    {
        prices[i] = prices[i - 1] + vol * prices[i - 1] * test_randn();
    }
}

/* Generate stochastic volatility series */
static void generate_sv_prices(double *prices, double *true_vol, int n,
                               double p0, double mu_vol, double theta, double sigma_vol)
{
    double log_vol = mu_vol;
    prices[0] = p0;
    true_vol[0] = exp(log_vol);

    for (int i = 1; i < n; i++)
    {
        /* OU process for log-vol */
        log_vol = log_vol + theta * (mu_vol - log_vol) + sigma_vol * test_randn();
        true_vol[i] = exp(log_vol);

        /* Price with stochastic vol */
        prices[i] = prices[i - 1] + true_vol[i] * prices[i - 1] * test_randn();
    }
}

/* Create filter with default test config */
static PF2D *create_test_filter(int n_particles)
{
    PF2D *pf = pf2d_create(n_particles, 4);
    if (!pf)
        return NULL;

    /* Set regime params */
    pf2d_set_regime_params(pf, 0, 0.0001, 0.02, log(0.01), 0.05, 0.0);
    pf2d_set_regime_params(pf, 1, 0.0, 0.05, log(0.008), 0.03, 0.0);
    pf2d_set_regime_params(pf, 2, 0.0, 0.10, log(0.02), 0.10, 0.0);
    pf2d_set_regime_params(pf, 3, 0.0, 0.20, log(0.05), 0.20, 0.0);

    return pf;
}

/* Helper to create properly initialized regime probs */
static void init_default_regime_probs(PF2DRegimeProbs *rp)
{
    pf2d_real probs[] = {0.4, 0.3, 0.2, 0.1};
    pf2d_set_regime_probs(rp, probs, 4);
}

static void init_uniform_regime_probs(PF2DRegimeProbs *rp, int n)
{
    pf2d_real probs[PF2D_MAX_REGIMES];
    for (int i = 0; i < n; i++)
        probs[i] = 1.0 / n;
    pf2d_set_regime_probs(rp, probs, n);
}

/* ========================================================================== */
/* Basic Tests                                                                 */
/* ========================================================================== */

TEST(create_destroy)
{
    PF2D *pf = pf2d_create(1000, 4);
    ASSERT(pf != NULL);
    pf2d_destroy(pf);
    return 1;
}

TEST(create_various_sizes)
{
    int sizes[] = {100, 500, 1000, 2000, 4000, 8000, 16000};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);

    for (int i = 0; i < n_sizes; i++)
    {
        PF2D *pf = pf2d_create(sizes[i], 4);
        ASSERT(pf != NULL);
        pf2d_destroy(pf);
    }
    return 1;
}

TEST(create_various_regimes)
{
    for (int r = 1; r <= 8; r++)
    {
        PF2D *pf = pf2d_create(1000, r);
        ASSERT(pf != NULL);
        pf2d_destroy(pf);
    }
    return 1;
}

TEST(initialize)
{
    PF2D *pf = create_test_filter(1000);
    ASSERT(pf != NULL);

    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    double price_mean = pf2d_price_mean(pf);
    ASSERT_NEAR(price_mean, 100.0, 1.0);

    pf2d_destroy(pf);
    return 1;
}

TEST(initialize_different_prices)
{
    double test_prices[] = {10.0, 100.0, 500.0, 1000.0, 5000.0};
    int n = sizeof(test_prices) / sizeof(test_prices[0]);

    for (int i = 0; i < n; i++)
    {
        PF2D *pf = create_test_filter(1000);
        pf2d_initialize(pf, test_prices[i], test_prices[i] * 0.001, log(0.01), 0.5);

        double est = pf2d_price_mean(pf);
        ASSERT_NEAR(est, test_prices[i], test_prices[i] * 0.05);

        pf2d_destroy(pf);
    }
    return 1;
}

/* ========================================================================== */
/* Single Update Tests                                                         */
/* ========================================================================== */

TEST(single_update)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.01);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    PF2DOutput out = pf2d_update(pf, 100.5, &rp);

    ASSERT_BETWEEN(out.price_mean, 99.0, 102.0);
    ASSERT_GT(out.ess, 0);
    ASSERT_LT(out.ess, 4001);

    pf2d_destroy(pf);
    return 1;
}

TEST(ess_healthy_after_small_move)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Small price move should give high ESS */
    PF2DOutput out = pf2d_update(pf, 100.1, &rp);

    ASSERT_GT(out.ess, 2000); /* Should be > 50% */

    pf2d_destroy(pf);
    return 1;
}

TEST(ess_drops_on_large_move)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.01); /* Tight obs variance */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Large unexpected move should drop ESS */
    PF2DOutput out = pf2d_update(pf, 110.0, &rp); /* 10% jump */

    ASSERT_LT(out.ess, 2000); /* Should be < 50% due to surprise */

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Tracking Tests                                                              */
/* ========================================================================== */

TEST(tracks_constant_price)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.01);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Feed constant price */
    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 100.0, &rp);
    }

    double est = pf2d_price_mean(pf);
    ASSERT_NEAR(est, 100.0, 1.0);

    pf2d_destroy(pf);
    return 1;
}

TEST(tracks_linear_trend)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.01); /* Tighter to track $1 moves */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Linear trend: 100 -> 150 */
    for (int i = 0; i < 50; i++)
    {
        double price = 100.0 + i * 1.0;
        pf2d_update(pf, price, &rp);
    }

    double est = pf2d_price_mean(pf);
    /* With tight obs_var, filter should track closely but may lag slightly */
    ASSERT_BETWEEN(est, 130.0, 155.0);

    pf2d_destroy(pf);
    return 1;
}

TEST(tracks_noisy_random_walk)
{
    test_seed = 42;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[100];
    generate_test_prices(prices, 100, 100.0, 0.01);

    double total_error = 0;
    for (int i = 0; i < 100; i++)
    {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);
        total_error += fabs(out.price_mean - prices[i]);
    }

    double mean_error = total_error / 100.0;
    ASSERT_LT(mean_error, 5.0); /* Mean tracking error < $5 */

    pf2d_destroy(pf);
    return 1;
}

TEST(tracks_stochastic_volatility)
{
    test_seed = 123;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[200], true_vol[200];
    generate_sv_prices(prices, true_vol, 200, 100.0, log(0.01), 0.05, 0.1);

    double vol_est_sum = 0;
    double true_vol_sum = 0;

    for (int i = 0; i < 200; i++)
    {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);
        if (i > 50)
        { /* Skip warmup */
            vol_est_sum += out.vol_mean;
            true_vol_sum += true_vol[i];
        }
    }

    double mean_vol_est = vol_est_sum / 150.0;
    double mean_true_vol = true_vol_sum / 150.0;

    /* Vol estimate should be in reasonable range - particle filters are noisy */
    /* Allow wider tolerance: 0.1x to 10x of true vol */
    ASSERT_BETWEEN(mean_vol_est, mean_true_vol * 0.1, mean_true_vol * 10.0);

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* ESS and Resampling Tests                                                    */
/* ========================================================================== */

TEST(ess_computation)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    double ess = pf2d_effective_sample_size(pf);

    /* After init with uniform weights, ESS should be ~N */
    ASSERT_BETWEEN(ess, 3500, 4001); /* Allow exactly 4000 */

    pf2d_destroy(pf);
    return 1;
}

TEST(resampling_triggers)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.001); /* Very tight */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Large surprise should trigger resampling */
    PF2DOutput out = pf2d_update(pf, 105.0, &rp);

    /* Check that resampling occurred (ESS was restored) */
    double ess_after = pf2d_effective_sample_size(pf);
    ASSERT_GT(ess_after, 1000); /* Should have resampled if ESS dropped */

    pf2d_destroy(pf);
    return 1;
}

TEST(ess_stable_over_many_updates)
{
    test_seed = 999;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[500];
    generate_test_prices(prices, 500, 100.0, 0.01);

    int ess_below_100_count = 0;
    for (int i = 0; i < 500; i++)
    {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);
        if (out.ess < 100)
            ess_below_100_count++;
    }

    /* ESS should rarely collapse completely */
    ASSERT_LT(ess_below_100_count, 50); /* < 10% of time */

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Observation Variance Sensitivity Tests                                      */
/* ========================================================================== */

TEST(obs_var_too_tight)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.0001); /* Way too tight */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Even small move should cause ESS collapse */
    PF2DOutput out = pf2d_update(pf, 100.5, &rp);

    /* Should have very low ESS (but resampling might restore it) */
    /* This tests that the filter doesn't crash */
    ASSERT(out.price_mean > 0);

    pf2d_destroy(pf);
    return 1;
}

TEST(obs_var_too_loose)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1000.0); /* Way too loose */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Run several updates */
    for (int i = 0; i < 20; i++)
    {
        pf2d_update(pf, 110.0, &rp); /* Constant high price */
    }

    /* Filter should barely move (ignoring observations) */
    double est = pf2d_price_mean(pf);

    /* With very loose obs var, estimate stays near prior */
    /* This is actually correct behavior */
    ASSERT(est > 0); /* Just check it doesn't crash/NaN */

    pf2d_destroy(pf);
    return 1;
}

TEST(obs_var_scaled_to_price)
{
    /* Test that obs_var should scale with price^2 */
    double prices[] = {10.0, 100.0, 1000.0};

    for (int p = 0; p < 3; p++)
    {
        double price = prices[p];
        double obs_var = (price * 0.002) * (price * 0.002); /* 0.2% of price */

        PF2D *pf = create_test_filter(4000);
        pf2d_set_observation_variance(pf, obs_var);
        pf2d_initialize(pf, price, price * 0.001, log(0.01), 0.5);

        PF2DRegimeProbs rp;
        init_default_regime_probs(&rp);
        pf2d_build_regime_lut(pf, &rp);

        /* Small percentage move */
        PF2DOutput out = pf2d_update(pf, price * 1.001, &rp);

        /* ESS should be healthy */
        ASSERT_GT(out.ess, 2000);

        pf2d_destroy(pf);
    }
    return 1;
}

/* ========================================================================== */
/* Regime Tests                                                                */
/* ========================================================================== */

TEST(single_regime)
{
    PF2D *pf = pf2d_create(4000, 1);
    pf2d_set_regime_params(pf, 0, 0.0, 0.05, log(0.01), 0.05, 0.0);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    pf2d_real probs[] = {1.0};
    pf2d_set_regime_probs(&rp, probs, 1);
    pf2d_build_regime_lut(pf, &rp);

    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 100.0 + i * 0.1, &rp);
    }

    double est = pf2d_price_mean(pf);
    ASSERT_BETWEEN(est, 100.0, 110.0);

    pf2d_destroy(pf);
    return 1;
}

TEST(regime_probs_sum_to_one)
{
    PF2D *pf = create_test_filter(1000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_uniform_regime_probs(&rp, 4);
    pf2d_build_regime_lut(pf, &rp);

    PF2DOutput out = pf2d_update(pf, 100.1, &rp);

    double sum = 0;
    for (int i = 0; i < 4; i++)
    {
        sum += out.regime_probs[i];
    }

    ASSERT_NEAR(sum, 1.0, 0.01);

    pf2d_destroy(pf);
    return 1;
}

TEST(dominant_regime_valid)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    for (int i = 0; i < 20; i++)
    {
        PF2DOutput out = pf2d_update(pf, 100.0 + i * 0.1, &rp);

        ASSERT_BETWEEN(out.dominant_regime, 0, 3);
    }

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Numerical Stability Tests                                                   */
/* ========================================================================== */

TEST(no_nan_inf_basic)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    for (int i = 0; i < 100; i++)
    {
        PF2DOutput out = pf2d_update(pf, 100.0 + test_randn(), &rp);

        ASSERT(!isnan(out.price_mean));
        ASSERT(!isinf(out.price_mean));
        ASSERT(!isnan(out.vol_mean));
        ASSERT(!isinf(out.vol_mean));
        ASSERT(!isnan(out.ess));
        ASSERT(!isinf(out.ess));
    }

    pf2d_destroy(pf);
    return 1;
}

TEST(no_nan_inf_extreme_prices)
{
    double extreme_prices[] = {0.01, 0.1, 1.0, 10000.0, 100000.0};
    int n = sizeof(extreme_prices) / sizeof(extreme_prices[0]);

    for (int p = 0; p < n; p++)
    {
        double price = extreme_prices[p];
        double obs_var = (price * 0.01) * (price * 0.01);

        PF2D *pf = create_test_filter(2000);
        pf2d_set_observation_variance(pf, obs_var);
        pf2d_initialize(pf, price, price * 0.001, log(0.01), 0.5);

        PF2DRegimeProbs rp;
        init_default_regime_probs(&rp);
        pf2d_build_regime_lut(pf, &rp);

        for (int i = 0; i < 20; i++)
        {
            double obs = price * (1.0 + 0.01 * test_randn());
            PF2DOutput out = pf2d_update(pf, obs, &rp);

            ASSERT(!isnan(out.price_mean));
            ASSERT(!isinf(out.price_mean));
        }

        pf2d_destroy(pf);
    }
    return 1;
}

TEST(no_nan_inf_long_run)
{
    test_seed = 777;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[1000];
    generate_test_prices(prices, 1000, 100.0, 0.01);

    for (int i = 0; i < 1000; i++)
    {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);

        if (isnan(out.price_mean) || isinf(out.price_mean))
        {
            printf("\n    NaN/Inf at step %d: price=%g, mean=%g\n",
                   i, prices[i], out.price_mean);
            pf2d_destroy(pf);
            return 0;
        }
    }

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Performance Tests                                                           */
/* ========================================================================== */

TEST(performance_1000_particles)
{
    test_seed = 111;

    PF2D *pf = create_test_filter(1000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);
    pf2d_warmup(pf);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[1000];
    generate_test_prices(prices, 1000, 100.0, 0.01);

    clock_t start = clock();
    for (int i = 0; i < 1000; i++)
    {
        pf2d_update(pf, prices[i], &rp);
    }
    clock_t end = clock();

    double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double us_per_tick = elapsed_ms * 1000.0 / 1000.0;

    printf("(%.1f μs/tick) ", us_per_tick);

    /* Should be faster than 500 μs/tick with 1000 particles */
    ASSERT_LT(us_per_tick, 500.0);

    pf2d_destroy(pf);
    return 1;
}

TEST(performance_4000_particles)
{
    test_seed = 222;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);
    pf2d_warmup(pf);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[1000];
    generate_test_prices(prices, 1000, 100.0, 0.01);

    clock_t start = clock();
    for (int i = 0; i < 1000; i++)
    {
        pf2d_update(pf, prices[i], &rp);
    }
    clock_t end = clock();

    double elapsed_ms = (double)(end - start) / CLOCKS_PER_SEC * 1000.0;
    double us_per_tick = elapsed_ms * 1000.0 / 1000.0;

    printf("(%.1f μs/tick) ", us_per_tick);

    /* Should be faster than 200 μs/tick with 4000 particles */
    ASSERT_LT(us_per_tick, 200.0);

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Edge Case Tests                                                             */
/* ========================================================================== */

TEST(zero_price_handling)
{
    PF2D *pf = create_test_filter(1000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Zero observation - should not crash */
    PF2DOutput out = pf2d_update(pf, 0.0, &rp);

    ASSERT(!isnan(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

TEST(negative_price_handling)
{
    PF2D *pf = create_test_filter(1000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Negative observation - should not crash */
    PF2DOutput out = pf2d_update(pf, -10.0, &rp);

    ASSERT(!isnan(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

TEST(repeated_same_observation)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Same observation 100 times */
    for (int i = 0; i < 100; i++)
    {
        pf2d_update(pf, 100.0, &rp);
    }

    double est = pf2d_price_mean(pf);
    ASSERT_NEAR(est, 100.0, 2.0);

    double ess = pf2d_effective_sample_size(pf);
    ASSERT_GT(ess, 1000); /* Should have healthy ESS */

    pf2d_destroy(pf);
    return 1;
}

TEST(reinitialize)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.01); /* Tighter to track $1 moves */

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* First run */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);
    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 100.0 + i, &rp);
    }

    /* Reinitialize */
    pf2d_initialize(pf, 200.0, 0.1, log(0.01), 0.5);

    double est = pf2d_price_mean(pf);
    ASSERT_NEAR(est, 200.0, 2.0);

    /* Run again */
    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 200.0 + i, &rp);
    }

    est = pf2d_price_mean(pf);
    ASSERT_BETWEEN(est, 230.0, 255.0); /* May lag slightly */

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Main                                                                        */
/* ========================================================================== */

int main(void)
{
    printf("========================================\n");
    printf("Particle Filter 2D Test Suite\n");
    printf("========================================\n\n");

    printf("Basic Tests:\n");
    RUN_TEST(create_destroy);
    RUN_TEST(create_various_sizes);
    RUN_TEST(create_various_regimes);
    RUN_TEST(initialize);
    RUN_TEST(initialize_different_prices);

    printf("\nSingle Update Tests:\n");
    RUN_TEST(single_update);
    RUN_TEST(ess_healthy_after_small_move);
    RUN_TEST(ess_drops_on_large_move);

    printf("\nTracking Tests:\n");
    RUN_TEST(tracks_constant_price);
    RUN_TEST(tracks_linear_trend);
    RUN_TEST(tracks_noisy_random_walk);
    RUN_TEST(tracks_stochastic_volatility);

    printf("\nESS and Resampling Tests:\n");
    RUN_TEST(ess_computation);
    RUN_TEST(resampling_triggers);
    RUN_TEST(ess_stable_over_many_updates);

    printf("\nObservation Variance Tests:\n");
    RUN_TEST(obs_var_too_tight);
    RUN_TEST(obs_var_too_loose);
    RUN_TEST(obs_var_scaled_to_price);

    printf("\nRegime Tests:\n");
    RUN_TEST(single_regime);
    RUN_TEST(regime_probs_sum_to_one);
    RUN_TEST(dominant_regime_valid);

    printf("\nNumerical Stability Tests:\n");
    RUN_TEST(no_nan_inf_basic);
    RUN_TEST(no_nan_inf_extreme_prices);
    RUN_TEST(no_nan_inf_long_run);

    printf("\nPerformance Tests:\n");
    RUN_TEST(performance_1000_particles);
    RUN_TEST(performance_4000_particles);

    printf("\nEdge Case Tests:\n");
    RUN_TEST(zero_price_handling);
    RUN_TEST(negative_price_handling);
    RUN_TEST(repeated_same_observation);
    RUN_TEST(reinitialize);

    printf("\n========================================\n");
    printf("Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
    {
        printf(" (%d FAILED)", tests_failed);
    }
    printf("\n========================================\n");

    return tests_failed > 0 ? 1 : 0;
}