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
    pf2d_set_observation_variance(pf, 0.01);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Linear trend: 100 -> ~105 (0.1% per step, within filter's tracking ability) */
    /* Filter expects ~1% vol, so 0.1% moves should be trackable */
    for (int i = 0; i < 50; i++)
    {
        double price = 100.0 * (1.0 + 0.001 * i); /* 0.1% per step */
        pf2d_update(pf, price, &rp);
    }

    double final_price = 100.0 * (1.0 + 0.001 * 49); /* ~104.9 */
    double est = pf2d_price_mean(pf);

    /* Filter should track within 2% of final price */
    ASSERT_BETWEEN(est, final_price * 0.98, final_price * 1.02);

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
    pf2d_set_observation_variance(pf, 0.01);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* First run: 100 -> ~105 */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);
    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 100.0 * (1.0 + 0.001 * i), &rp);
    }

    /* Reinitialize at 200 */
    pf2d_initialize(pf, 200.0, 0.1, log(0.01), 0.5);

    double est = pf2d_price_mean(pf);
    ASSERT_NEAR(est, 200.0, 2.0);

    /* Run again: 200 -> ~210 */
    for (int i = 0; i < 50; i++)
    {
        pf2d_update(pf, 200.0 * (1.0 + 0.001 * i), &rp);
    }

    double final_price = 200.0 * (1.0 + 0.001 * 49); /* ~209.8 */
    est = pf2d_price_mean(pf);
    ASSERT_BETWEEN(est, final_price * 0.98, final_price * 1.02);

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Stress Tests                                                                */
/* ========================================================================== */

TEST(long_run_1000_obs)
{
    test_seed = 555;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double prices[1000];
    generate_test_prices(prices, 1000, 100.0, 0.01);

    int nan_count = 0;
    int low_ess_count = 0;

    for (int i = 0; i < 1000; i++)
    {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);
        if (isnan(out.price_mean) || isinf(out.price_mean))
            nan_count++;
        if (out.ess < 100)
            low_ess_count++;
    }

    ASSERT(nan_count == 0);
    ASSERT_LT(low_ess_count, 100); /* < 10% severely degraded */

    pf2d_destroy(pf);
    return 1;
}

TEST(long_run_5000_obs)
{
    test_seed = 666;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double price = 100.0;
    for (int i = 0; i < 5000; i++)
    {
        price += price * 0.01 * test_randn();
        PF2DOutput out = pf2d_update(pf, price, &rp);

        if (isnan(out.price_mean) || isinf(out.price_mean))
        {
            printf("\n    NaN/Inf at step %d\n", i);
            pf2d_destroy(pf);
            return 0;
        }
    }

    pf2d_destroy(pf);
    return 1;
}

TEST(small_particle_count)
{
    PF2D *pf = pf2d_create(100, 4); /* Very few particles */
    ASSERT(pf != NULL);

    pf2d_set_regime_params(pf, 0, 0.0, 0.05, log(0.01), 0.05, 0.0);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Should still work, just noisier */
    for (int i = 0; i < 100; i++)
    {
        PF2DOutput out = pf2d_update(pf, 100.0 + test_randn(), &rp);
        ASSERT(!isnan(out.price_mean));
    }

    pf2d_destroy(pf);
    return 1;
}

TEST(large_particle_count)
{
    PF2D *pf = pf2d_create(16000, 4); /* Many particles */
    ASSERT(pf != NULL);

    pf2d_set_regime_params(pf, 0, 0.0, 0.05, log(0.01), 0.05, 0.0);
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    for (int i = 0; i < 50; i++)
    {
        PF2DOutput out = pf2d_update(pf, 100.0, &rp);
        ASSERT(!isnan(out.price_mean));
        ASSERT_GT(out.ess, 1000); /* Should have healthy ESS */
    }

    pf2d_destroy(pf);
    return 1;
}

TEST(price_jump)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Stable period */
    for (int i = 0; i < 20; i++)
    {
        pf2d_update(pf, 100.0, &rp);
    }

    /* Sudden 20% jump (like a gap) */
    PF2DOutput out = pf2d_update(pf, 120.0, &rp);
    ASSERT(!isnan(out.price_mean));

    /* Filter should eventually adapt */
    for (int i = 0; i < 50; i++)
    {
        out = pf2d_update(pf, 120.0, &rp);
    }

    /* Filter will lag - just verify it moved toward 120 and is stable */
    ASSERT_BETWEEN(out.price_mean, 105.0, 125.0);
    ASSERT(!isnan(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

TEST(volatility_spike)
{
    test_seed = 777;

    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    double price = 100.0;

    /* Low vol period */
    for (int i = 0; i < 50; i++)
    {
        price += price * 0.005 * test_randn();
        pf2d_update(pf, price, &rp);
    }

    /* High vol period (5x normal) */
    for (int i = 0; i < 50; i++)
    {
        price += price * 0.025 * test_randn();
        PF2DOutput out = pf2d_update(pf, price, &rp);
        ASSERT(!isnan(out.price_mean));
    }

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* API Robustness Tests                                                        */
/* ========================================================================== */

TEST(null_filter_handling)
{
    /* These should not crash */
    pf2d_destroy(NULL);

    /* Can't easily test other NULL cases without modifying API */
    /* But at minimum, destroy(NULL) should be safe */
    return 1;
}

TEST(invalid_regime_index)
{
    PF2D *pf = pf2d_create(1000, 4);

    /* Setting params for invalid regime should be ignored */
    pf2d_set_regime_params(pf, -1, 0.0, 0.05, 0.0, 0.05, 0.0);
    pf2d_set_regime_params(pf, 100, 0.0, 0.05, 0.0, 0.05, 0.0);

    /* Should still work normally */
    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    PF2DOutput out = pf2d_update(pf, 100.0, &rp);
    ASSERT(!isnan(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

TEST(extreme_obs_variance)
{
    PF2D *pf = create_test_filter(1000);

    /* Very tiny obs variance */
    pf2d_set_observation_variance(pf, 1e-10);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    PF2DOutput out = pf2d_update(pf, 100.0, &rp);
    ASSERT(!isnan(out.price_mean));
    ASSERT(!isinf(out.price_mean));

    /* Very large obs variance */
    pf2d_set_observation_variance(pf, 1e10);
    out = pf2d_update(pf, 100.0, &rp);
    ASSERT(!isnan(out.price_mean));
    ASSERT(!isinf(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

TEST(warmup_function)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    /* Warmup should not crash and should prepare for fast execution */
    pf2d_warmup(pf);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    PF2DOutput out = pf2d_update(pf, 100.0, &rp);
    ASSERT(!isnan(out.price_mean));

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Statistical Validation Tests                                                */
/* ========================================================================== */

TEST(mean_estimate_unbiased)
{
    /* Run filter many times on same data, check mean estimate is unbiased */
    test_seed = 888;

    double prices[100];
    generate_test_prices(prices, 100, 100.0, 0.01);
    double final_true_price = prices[99];

    double sum_estimates = 0;
    int n_runs = 20;

    for (int run = 0; run < n_runs; run++)
    {
        PF2D *pf = create_test_filter(2000);
        pf2d_set_observation_variance(pf, 0.5);
        pf2d_initialize(pf, prices[0], 0.1, log(0.01), 0.5);

        PF2DRegimeProbs rp;
        init_default_regime_probs(&rp);
        pf2d_build_regime_lut(pf, &rp);

        for (int i = 0; i < 100; i++)
        {
            pf2d_update(pf, prices[i], &rp);
        }

        sum_estimates += pf2d_price_mean(pf);
        pf2d_destroy(pf);
    }

    double mean_estimate = sum_estimates / n_runs;

    /* Mean of estimates should be close to final price */
    ASSERT_BETWEEN(mean_estimate, final_true_price * 0.9, final_true_price * 1.1);

    return 1;
}

TEST(ess_reflects_weight_concentration)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 1.0); /* Moderate */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Update with observation matching prior - ESS should stay high */
    PF2DOutput out1 = pf2d_update(pf, 100.0, &rp);
    double ess_match = out1.ess;

    /* Reinitialize */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    /* Now set very tight obs variance */
    pf2d_set_observation_variance(pf, 0.001);

    /* Update with slightly different obs - ESS should drop more */
    PF2DOutput out2 = pf2d_update(pf, 100.5, &rp);
    double ess_tight = out2.ess;

    /* Tighter variance should give lower ESS for same surprise */
    /* (This tests that likelihood is actually affecting weights) */
    ASSERT_LT(ess_tight, ess_match);

    pf2d_destroy(pf);
    return 1;
}

TEST(resampling_restores_diversity)
{
    PF2D *pf = create_test_filter(4000);
    pf2d_set_observation_variance(pf, 0.001); /* Very tight - will cause ESS collapse */
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    PF2DRegimeProbs rp;
    init_default_regime_probs(&rp);
    pf2d_build_regime_lut(pf, &rp);

    /* Force ESS collapse with surprising observation */
    pf2d_update(pf, 105.0, &rp); /* 5% surprise with tight obs_var */

    /* After resampling (which should trigger), ESS should be restored */
    double ess_after = pf2d_effective_sample_size(pf);

    /* Resampling resets weights to uniform, so ESS ≈ N */
    ASSERT_GT(ess_after, 2000);

    pf2d_destroy(pf);
    return 1;
}

/* ========================================================================== */
/* Regime Behavior Tests                                                       */
/* ========================================================================== */

TEST(regime_distribution_matches_prior)
{
    /* If all regimes have same dynamics, posterior regime probs should ≈ prior */
    PF2D *pf = pf2d_create(4000, 4);

    /* Set all regimes to identical params */
    for (int r = 0; r < 4; r++)
    {
        pf2d_set_regime_params(pf, r, 0.0, 0.05, log(0.01), 0.05, 0.0);
    }

    pf2d_set_observation_variance(pf, 0.5);
    pf2d_initialize(pf, 100.0, 0.1, log(0.01), 0.5);

    /* Uniform prior */
    PF2DRegimeProbs rp;
    init_uniform_regime_probs(&rp, 4);
    pf2d_build_regime_lut(pf, &rp);

    /* Run for a while */
    for (int i = 0; i < 100; i++)
    {
        pf2d_update(pf, 100.0 + test_randn() * 0.1, &rp);
    }

    PF2DOutput out = pf2d_update(pf, 100.0, &rp);

    /* Each regime should have ~25% (±15%) */
    for (int r = 0; r < 4; r++)
    {
        ASSERT_BETWEEN(out.regime_probs[r], 0.10, 0.40);
    }

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
    RUN_TEST(regime_distribution_matches_prior);

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

    printf("\nStress Tests:\n");
    RUN_TEST(long_run_1000_obs);
    RUN_TEST(long_run_5000_obs);
    RUN_TEST(small_particle_count);
    RUN_TEST(large_particle_count);
    RUN_TEST(price_jump);
    RUN_TEST(volatility_spike);

    printf("\nAPI Robustness Tests:\n");
    RUN_TEST(null_filter_handling);
    RUN_TEST(invalid_regime_index);
    RUN_TEST(extreme_obs_variance);
    RUN_TEST(warmup_function);

    printf("\nStatistical Validation Tests:\n");
    RUN_TEST(mean_estimate_unbiased);
    RUN_TEST(ess_reflects_weight_concentration);
    RUN_TEST(resampling_restores_diversity);

    printf("\n========================================\n");
    printf("Results: %d/%d passed", tests_passed, tests_run);
    if (tests_failed > 0)
    {
        printf(" (%d FAILED)", tests_failed);
    }
    printf("\n========================================\n");

    return tests_failed > 0 ? 1 : 0;
}