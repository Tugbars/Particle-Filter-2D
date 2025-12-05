/**
 * @file test_pf2d_adaptive.c
 * @brief Unit tests for adaptive self-calibration layer
 *
 * This test file uses minimal stubs to test the adaptive logic
 * without requiring the full MKL infrastructure.
 *
 * Tests cover:
 *   1. ESS-driven σ_vol scaling
 *   2. Volatility regime detection
 *   3. Regime posterior feedback
 *   4. PMMH/BOCPD integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>

/*============================================================================
 * MINIMAL TYPE DEFINITIONS (matching particle_filter_2d.h)
 *============================================================================*/

typedef double pf2d_real;

#define PF2D_MAX_REGIMES 8
#define PF2D_ADAPTIVE_SIGMA_SCALE_MIN ((pf2d_real)0.5)
#define PF2D_ADAPTIVE_SIGMA_SCALE_MAX ((pf2d_real)2.0)

typedef struct
{
    /* ESS-driven σ_vol scaling */
    pf2d_real ess_ema;
    pf2d_real ess_ratio_ema;
    int low_ess_streak;
    int high_ess_streak;
    pf2d_real sigma_vol_scale;

    /* Thresholds */
    pf2d_real ess_low_thresh;
    pf2d_real ess_high_thresh;
    int low_streak_thresh;
    int high_streak_thresh;

    /* Regime posterior feedback */
    pf2d_real regime_ema[PF2D_MAX_REGIMES];
    int lut_update_countdown;
    int bocpd_cooldown;
    int lut_update_interval;
    int bocpd_cooldown_duration;

    /* Volatility regime detection */
    pf2d_real vol_short_ema;
    pf2d_real vol_long_ema;
    int high_vol_mode;
    pf2d_real vol_enter_ratio;
    pf2d_real vol_exit_ratio;

    /* Saved defaults */
    pf2d_real base_resample_threshold;
    pf2d_real base_bandwidth_price;
    pf2d_real base_bandwidth_vol;

    /* Feature flags */
    int enable_sigma_scaling;
    int enable_regime_feedback;
    int enable_vol_detection;
} PF2DAdaptive;

typedef struct
{
    pf2d_real price_mean;
    pf2d_real vol_mean;
    pf2d_real ess;
    pf2d_real regime_probs[PF2D_MAX_REGIMES];

    /* Adaptive diagnostics */
    pf2d_real sigma_vol_scale;
    pf2d_real ess_ema;
    int high_vol_mode;
    int regime_feedback_active;
    int bocpd_cooldown_remaining;
} PF2DOutput;

/* Minimal PF2D stub */
typedef struct
{
    int n_particles;
    int n_regimes;
    pf2d_real resample_threshold;
    pf2d_real reg_bandwidth_price;
    pf2d_real reg_bandwidth_vol;
    PF2DAdaptive adaptive;
} PF2D;

/*============================================================================
 * ADAPTIVE IMPLEMENTATION (embedded for standalone testing)
 *============================================================================*/

static void pf2d_adaptive_init(PF2D *pf)
{
    PF2DAdaptive *a = &pf->adaptive;

    a->ess_ema = (pf2d_real)pf->n_particles;
    a->ess_ratio_ema = 1.0;
    a->low_ess_streak = 0;
    a->high_ess_streak = 0;
    a->sigma_vol_scale = 1.0;

    a->ess_low_thresh = 0.3;
    a->ess_high_thresh = 0.7;
    a->low_streak_thresh = 1000;
    a->high_streak_thresh = 2000;

    for (int r = 0; r < PF2D_MAX_REGIMES; r++)
    {
        a->regime_ema[r] = 1.0 / pf->n_regimes;
    }
    a->lut_update_countdown = 50;
    a->bocpd_cooldown = 0;
    a->lut_update_interval = 50;
    a->bocpd_cooldown_duration = 3000;

    a->vol_short_ema = 0.01;
    a->vol_long_ema = 0.01;
    a->high_vol_mode = 0;
    a->vol_enter_ratio = 1.8;
    a->vol_exit_ratio = 1.15;

    a->base_resample_threshold = pf->resample_threshold;
    a->base_bandwidth_price = pf->reg_bandwidth_price;
    a->base_bandwidth_vol = pf->reg_bandwidth_vol;

    a->enable_sigma_scaling = 1;
    a->enable_regime_feedback = 0;
    a->enable_vol_detection = 1;
}

static void pf2d_adaptive_tick(PF2D *pf, PF2DOutput *out)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* Update ESS EMA */
    const pf2d_real alpha_ess = 0.01;
    pf2d_real ess_ratio = out->ess / pf->n_particles;
    a->ess_ratio_ema = alpha_ess * ess_ratio + (1.0 - alpha_ess) * a->ess_ratio_ema;
    a->ess_ema = a->ess_ratio_ema * pf->n_particles;

    /* ESS-driven σ_vol scaling */
    if (a->enable_sigma_scaling)
    {
        if (a->ess_ratio_ema < a->ess_low_thresh)
        {
            a->low_ess_streak++;
            a->high_ess_streak = 0;

            if (a->low_ess_streak > a->low_streak_thresh)
            {
                a->sigma_vol_scale *= 1.02;
                if (a->sigma_vol_scale > PF2D_ADAPTIVE_SIGMA_SCALE_MAX)
                {
                    a->sigma_vol_scale = PF2D_ADAPTIVE_SIGMA_SCALE_MAX;
                }
            }
        }
        else if (a->ess_ratio_ema > a->ess_high_thresh)
        {
            a->high_ess_streak++;
            a->low_ess_streak = 0;

            if (a->high_ess_streak > a->high_streak_thresh)
            {
                a->sigma_vol_scale *= 0.99;
                if (a->sigma_vol_scale < PF2D_ADAPTIVE_SIGMA_SCALE_MIN)
                {
                    a->sigma_vol_scale = PF2D_ADAPTIVE_SIGMA_SCALE_MIN;
                }
            }
        }
        else
        {
            /* Healthy range - decay toward 1.0 */
            a->low_ess_streak = 0;
            a->high_ess_streak = 0;
            a->sigma_vol_scale = 0.999 * a->sigma_vol_scale + 0.001 * 1.0;
        }
    }

    /* Volatility regime detection */
    if (a->enable_vol_detection)
    {
        const pf2d_real alpha_short = 0.1;
        const pf2d_real alpha_long = 0.01;

        a->vol_short_ema = alpha_short * out->vol_mean + (1.0 - alpha_short) * a->vol_short_ema;
        a->vol_long_ema = alpha_long * out->vol_mean + (1.0 - alpha_long) * a->vol_long_ema;

        pf2d_real ratio = a->vol_short_ema / a->vol_long_ema;

        if (!a->high_vol_mode && ratio > a->vol_enter_ratio)
        {
            a->high_vol_mode = 1;
            pf->resample_threshold = 0.1; /* More aggressive */
            pf->reg_bandwidth_price = a->base_bandwidth_price * 1.5;
            pf->reg_bandwidth_vol = a->base_bandwidth_vol * 1.5;
        }
        else if (a->high_vol_mode && ratio < a->vol_exit_ratio)
        {
            a->high_vol_mode = 0;
            pf->resample_threshold = a->base_resample_threshold;
            pf->reg_bandwidth_price = a->base_bandwidth_price;
            pf->reg_bandwidth_vol = a->base_bandwidth_vol;
        }
    }

    /* Regime posterior feedback */
    if (a->bocpd_cooldown > 0)
    {
        a->bocpd_cooldown--;
    }

    if (a->enable_regime_feedback && a->bocpd_cooldown == 0)
    {
        const pf2d_real alpha_regime = 0.02;
        for (int r = 0; r < pf->n_regimes; r++)
        {
            a->regime_ema[r] = alpha_regime * out->regime_probs[r] +
                               (1.0 - alpha_regime) * a->regime_ema[r];
        }
    }

    /* Fill output diagnostics */
    out->sigma_vol_scale = a->sigma_vol_scale;
    out->ess_ema = a->ess_ema;
    out->high_vol_mode = a->high_vol_mode;
    out->regime_feedback_active = (a->enable_regime_feedback && a->bocpd_cooldown == 0);
    out->bocpd_cooldown_remaining = a->bocpd_cooldown;
}

static void pf2d_adaptive_reset_after_pmcmc(PF2D *pf)
{
    PF2DAdaptive *a = &pf->adaptive;
    a->sigma_vol_scale = 1.0;
    a->low_ess_streak = 0;
    a->high_ess_streak = 0;
    /* Note: high_vol_mode NOT reset - vol conditions persist */
}

static void pf2d_adaptive_notify_bocpd(PF2D *pf)
{
    pf->adaptive.bocpd_cooldown = pf->adaptive.bocpd_cooldown_duration;
}

/*============================================================================
 * TEST UTILITIES
 *============================================================================*/

#define TEST_PASS "\033[32mPASS\033[0m"
#define TEST_FAIL "\033[31mFAIL\033[0m"

static int tests_run = 0;
static int tests_passed = 0;

#define ASSERT_TRUE(cond, msg)                                         \
    do                                                                 \
    {                                                                  \
        tests_run++;                                                   \
        if (cond)                                                      \
        {                                                              \
            tests_passed++;                                            \
            printf("  [%s] %s\n", TEST_PASS, msg);                     \
        }                                                              \
        else                                                           \
        {                                                              \
            printf("  [%s] %s (line %d)\n", TEST_FAIL, msg, __LINE__); \
        }                                                              \
    } while (0)

#define ASSERT_NEAR(a, b, tol, msg)                                                                 \
    do                                                                                              \
    {                                                                                               \
        tests_run++;                                                                                \
        double _a = (double)(a), _b = (double)(b), _tol = (double)(tol);                            \
        if (fabs(_a - _b) <= _tol)                                                                  \
        {                                                                                           \
            tests_passed++;                                                                         \
            printf("  [%s] %s (%.6f ≈ %.6f)\n", TEST_PASS, msg, _a, _b);                            \
        }                                                                                           \
        else                                                                                        \
        {                                                                                           \
            printf("  [%s] %s (%.6f != %.6f, diff=%.6f)\n", TEST_FAIL, msg, _a, _b, fabs(_a - _b)); \
        }                                                                                           \
    } while (0)

#define ASSERT_GT(a, b, msg)                                              \
    do                                                                    \
    {                                                                     \
        tests_run++;                                                      \
        double _a = (double)(a), _b = (double)(b);                        \
        if (_a > _b)                                                      \
        {                                                                 \
            tests_passed++;                                               \
            printf("  [%s] %s (%.6f > %.6f)\n", TEST_PASS, msg, _a, _b);  \
        }                                                                 \
        else                                                              \
        {                                                                 \
            printf("  [%s] %s (%.6f <= %.6f)\n", TEST_FAIL, msg, _a, _b); \
        }                                                                 \
    } while (0)

#define ASSERT_LT(a, b, msg)                                              \
    do                                                                    \
    {                                                                     \
        tests_run++;                                                      \
        double _a = (double)(a), _b = (double)(b);                        \
        if (_a < _b)                                                      \
        {                                                                 \
            tests_passed++;                                               \
            printf("  [%s] %s (%.6f < %.6f)\n", TEST_PASS, msg, _a, _b);  \
        }                                                                 \
        else                                                              \
        {                                                                 \
            printf("  [%s] %s (%.6f >= %.6f)\n", TEST_FAIL, msg, _a, _b); \
        }                                                                 \
    } while (0)

#define ASSERT_EQ(a, b, msg)                                          \
    do                                                                \
    {                                                                 \
        tests_run++;                                                  \
        int _a = (int)(a), _b = (int)(b);                             \
        if (_a == _b)                                                 \
        {                                                             \
            tests_passed++;                                           \
            printf("  [%s] %s (%d == %d)\n", TEST_PASS, msg, _a, _b); \
        }                                                             \
        else                                                          \
        {                                                             \
            printf("  [%s] %s (%d != %d)\n", TEST_FAIL, msg, _a, _b); \
        }                                                             \
    } while (0)

/*============================================================================
 * HELPER: Create minimal PF2D for testing
 *============================================================================*/

static PF2D *create_test_pf(int n_particles, int n_regimes)
{
    PF2D *pf = (PF2D *)calloc(1, sizeof(PF2D));
    pf->n_particles = n_particles;
    pf->n_regimes = n_regimes;
    pf->resample_threshold = 0.5;
    pf->reg_bandwidth_price = 0.01;
    pf->reg_bandwidth_vol = 0.1;
    pf2d_adaptive_init(pf);
    return pf;
}

static void destroy_test_pf(PF2D *pf)
{
    free(pf);
}

/*============================================================================
 * TEST: INITIALIZATION
 *============================================================================*/

static void test_adaptive_init(void)
{
    printf("\n=== Test: Adaptive Initialization ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    ASSERT_NEAR(a->sigma_vol_scale, 1.0, 1e-9, "sigma_vol_scale init to 1.0");
    ASSERT_EQ(a->low_ess_streak, 0, "low_ess_streak init to 0");
    ASSERT_EQ(a->high_ess_streak, 0, "high_ess_streak init to 0");
    ASSERT_EQ(a->high_vol_mode, 0, "high_vol_mode init to 0");
    ASSERT_EQ(a->bocpd_cooldown, 0, "bocpd_cooldown init to 0");

    ASSERT_EQ(a->enable_sigma_scaling, 1, "sigma_scaling ON by default");
    ASSERT_EQ(a->enable_vol_detection, 1, "vol_detection ON by default");
    ASSERT_EQ(a->enable_regime_feedback, 0, "regime_feedback OFF by default");

    ASSERT_NEAR(a->ess_low_thresh, 0.3, 1e-6, "ess_low_thresh default 0.3");
    ASSERT_NEAR(a->ess_high_thresh, 0.7, 1e-6, "ess_high_thresh default 0.7");
    ASSERT_EQ(a->low_streak_thresh, 1000, "low_streak_thresh default 1000");
    ASSERT_EQ(a->high_streak_thresh, 2000, "high_streak_thresh default 2000");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: ESS-DRIVEN σ_VOL SCALING
 *============================================================================*/

static void test_ess_scaling_low_ess(void)
{
    printf("\n=== Test: ESS Scaling - Low ESS Triggers Widening ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    /* Set short streak threshold for testing */
    a->low_streak_thresh = 10;
    a->high_streak_thresh = 20;

    /* Pre-seed the EMA to low value (simulating previous low ESS) */
    a->ess_ratio_ema = 0.2;

    pf2d_real initial_scale = a->sigma_vol_scale;
    ASSERT_NEAR(initial_scale, 1.0, 1e-9, "Initial scale is 1.0");

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    /* Simulate low ESS for streak_thresh + buffer ticks */
    for (int i = 0; i < 15; i++)
    {
        out.ess = 0.2 * pf->n_particles; /* ESS = 0.2N < 0.3N */
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_GT(a->sigma_vol_scale, initial_scale, "Scale increased after low ESS streak");
    ASSERT_GT(a->low_ess_streak, 10, "Low ESS streak counted");
    ASSERT_EQ(a->high_ess_streak, 0, "High ESS streak reset");

    destroy_test_pf(pf);
}

static void test_ess_scaling_high_ess(void)
{
    printf("\n=== Test: ESS Scaling - High ESS Triggers Tightening ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->low_streak_thresh = 10;
    a->high_streak_thresh = 20;

    /* Start with elevated scale */
    a->sigma_vol_scale = 1.5;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    /* Simulate high ESS */
    for (int i = 0; i < 25; i++)
    {
        out.ess = 0.8 * pf->n_particles; /* ESS = 0.8N > 0.7N */
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_LT(a->sigma_vol_scale, 1.5, "Scale decreased after high ESS streak");
    ASSERT_GT(a->high_ess_streak, 20, "High ESS streak counted");
    ASSERT_EQ(a->low_ess_streak, 0, "Low ESS streak reset");

    destroy_test_pf(pf);
}

static void test_ess_scaling_bounds(void)
{
    printf("\n=== Test: ESS Scaling - Bounds Enforced ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->low_streak_thresh = 5;
    a->high_streak_thresh = 5;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    /* Push scale to max */
    for (int i = 0; i < 500; i++)
    {
        out.ess = 0.1 * pf->n_particles;
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_NEAR(a->sigma_vol_scale, PF2D_ADAPTIVE_SIGMA_SCALE_MAX, 1e-6,
                "Scale capped at max (2.0)");

    /* Push to min */
    a->sigma_vol_scale = 1.0;
    for (int i = 0; i < 500; i++)
    {
        out.ess = 0.9 * pf->n_particles;
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_NEAR(a->sigma_vol_scale, PF2D_ADAPTIVE_SIGMA_SCALE_MIN, 1e-6,
                "Scale capped at min (0.5)");

    destroy_test_pf(pf);
}

static void test_ess_scaling_decay(void)
{
    printf("\n=== Test: ESS Scaling - Decay Toward 1.0 ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->sigma_vol_scale = 1.5;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    /* Healthy ESS triggers decay */
    for (int i = 0; i < 1000; i++)
    {
        out.ess = 0.5 * pf->n_particles; /* In [0.3, 0.7] */
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_LT(a->sigma_vol_scale, 1.5, "Scale decayed from 1.5");
    ASSERT_GT(a->sigma_vol_scale, 1.0, "Scale hasn't fully decayed");

    /* Continue decay */
    for (int i = 0; i < 5000; i++)
    {
        out.ess = 0.5 * pf->n_particles;
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_NEAR(a->sigma_vol_scale, 1.0, 0.05, "Scale decayed close to 1.0");

    destroy_test_pf(pf);
}

static void test_ess_scaling_disabled(void)
{
    printf("\n=== Test: ESS Scaling - Disabled ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_sigma_scaling = 0;
    a->low_streak_thresh = 5;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    for (int i = 0; i < 100; i++)
    {
        out.ess = 0.1 * pf->n_particles;
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_NEAR(a->sigma_vol_scale, 1.0, 1e-9, "Scale unchanged when disabled");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: VOLATILITY REGIME DETECTION
 *============================================================================*/

static void test_vol_detection_enter(void)
{
    printf("\n=== Test: Vol Detection - Enter High-Vol Mode ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    /* Pre-seed EMAs to baseline - this is the "calm before storm" */
    a->vol_short_ema = 0.01;
    a->vol_long_ema = 0.01;

    pf2d_real base_threshold = pf->resample_threshold;
    pf2d_real base_bw = pf->reg_bandwidth_price;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;

    /* Simulate sudden vol spike - short EMA catches up fast, long EMA lags */
    /* With α_short=0.1, short EMA converges ~10x faster than long EMA */
    /* After ~20 ticks, short ≈ 0.03, long ≈ 0.012, ratio ≈ 2.5 */
    for (int i = 0; i < 30; i++)
    {
        out.vol_mean = 0.04; /* 4x baseline - sudden spike */
        pf2d_adaptive_tick(pf, &out);
    }

    pf2d_real ratio = a->vol_short_ema / a->vol_long_ema;
    ASSERT_GT(ratio, 1.8, "Vol ratio elevated after spike");
    ASSERT_EQ(a->high_vol_mode, 1, "Entered high-vol mode");
    ASSERT_LT(pf->resample_threshold, base_threshold, "Resample threshold lowered");
    ASSERT_GT(pf->reg_bandwidth_price, base_bw, "Bandwidth increased");

    destroy_test_pf(pf);
}

static void test_vol_detection_exit(void)
{
    printf("\n=== Test: Vol Detection - Exit High-Vol Mode ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    /* Start in high-vol mode */
    a->high_vol_mode = 1;
    a->vol_short_ema = 0.02;
    a->vol_long_ema = 0.01;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;

    /* Let vol normalize */
    for (int i = 0; i < 300; i++)
    {
        out.vol_mean = 0.01;
        pf2d_adaptive_tick(pf, &out);
    }

    pf2d_real ratio = a->vol_short_ema / a->vol_long_ema;
    ASSERT_LT(ratio, 1.2, "Vol ratio normalized");
    ASSERT_EQ(a->high_vol_mode, 0, "Exited high-vol mode");

    destroy_test_pf(pf);
}

static void test_vol_detection_hysteresis(void)
{
    printf("\n=== Test: Vol Detection - Hysteresis ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    /* Start in high-vol mode with ratio in dead band */
    a->high_vol_mode = 1;
    a->vol_short_ema = 0.015; /* ratio = 1.5 */
    a->vol_long_ema = 0.01;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;
    out.vol_mean = 0.015; /* Maintain 1.5x */

    for (int i = 0; i < 50; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_EQ(a->high_vol_mode, 1, "Stayed in high-vol mode (hysteresis)");

    destroy_test_pf(pf);
}

static void test_vol_detection_disabled(void)
{
    printf("\n=== Test: Vol Detection - Disabled ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_vol_detection = 0;
    a->vol_short_ema = 0.01;
    a->vol_long_ema = 0.01;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;

    for (int i = 0; i < 100; i++)
    {
        out.vol_mean = 0.05; /* 5x baseline */
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_EQ(a->high_vol_mode, 0, "Did not enter high-vol mode when disabled");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: REGIME POSTERIOR FEEDBACK
 *============================================================================*/

static void test_regime_feedback_ema(void)
{
    printf("\n=== Test: Regime Feedback - EMA Update ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_regime_feedback = 1;

    for (int r = 0; r < 4; r++)
    {
        a->regime_ema[r] = 0.25;
    }

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;
    out.vol_mean = 0.01;
    out.regime_probs[0] = 0.05;
    out.regime_probs[1] = 0.05;
    out.regime_probs[2] = 0.85;
    out.regime_probs[3] = 0.05;

    for (int i = 0; i < 200; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_GT(a->regime_ema[2], a->regime_ema[0], "Regime 2 EMA increased");
    ASSERT_GT(a->regime_ema[2], 0.5, "Regime 2 EMA > 0.5 after feedback");

    destroy_test_pf(pf);
}

static void test_regime_feedback_cooldown(void)
{
    printf("\n=== Test: Regime Feedback - BOCPD Cooldown ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_regime_feedback = 1;
    a->bocpd_cooldown_duration = 100;

    pf2d_real initial_ema_2 = a->regime_ema[2];

    pf2d_adaptive_notify_bocpd(pf);
    ASSERT_EQ(a->bocpd_cooldown, 100, "Cooldown set to 100 ticks");

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;
    out.vol_mean = 0.01;
    out.regime_probs[2] = 0.95;

    /* Tick during cooldown */
    for (int i = 0; i < 50; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_EQ(a->bocpd_cooldown, 50, "Cooldown decremented");
    ASSERT_NEAR(a->regime_ema[2], initial_ema_2, 1e-6,
                "Regime EMA unchanged during cooldown");

    /* Finish cooldown */
    for (int i = 0; i < 60; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_EQ(a->bocpd_cooldown, 0, "Cooldown expired");
    ASSERT_GT(a->regime_ema[2], initial_ema_2, "Regime EMA updated after cooldown");

    destroy_test_pf(pf);
}

static void test_regime_feedback_disabled(void)
{
    printf("\n=== Test: Regime Feedback - Disabled by Default ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    pf2d_real initial_ema_2 = a->regime_ema[2];

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;
    out.vol_mean = 0.01;
    out.regime_probs[2] = 0.95;

    for (int i = 0; i < 100; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_NEAR(a->regime_ema[2], initial_ema_2, 1e-6,
                "Regime EMA unchanged when feedback disabled");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: PMMH INTEGRATION
 *============================================================================*/

static void test_reset_after_pmcmc(void)
{
    printf("\n=== Test: Reset After PMCMC ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->sigma_vol_scale = 1.8;
    a->low_ess_streak = 500;
    a->high_ess_streak = 0;
    a->high_vol_mode = 1;

    pf2d_adaptive_reset_after_pmcmc(pf);

    ASSERT_NEAR(a->sigma_vol_scale, 1.0, 1e-9, "sigma_vol_scale reset to 1.0");
    ASSERT_EQ(a->low_ess_streak, 0, "low_ess_streak reset");
    ASSERT_EQ(a->high_ess_streak, 0, "high_ess_streak reset");
    ASSERT_EQ(a->high_vol_mode, 1, "high_vol_mode NOT reset (intentional)");

    destroy_test_pf(pf);
}

static void test_bocpd_notification(void)
{
    printf("\n=== Test: BOCPD Notification ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_regime_feedback = 1;
    a->bocpd_cooldown_duration = 3000;
    a->sigma_vol_scale = 1.5;
    a->high_vol_mode = 1;

    pf2d_adaptive_notify_bocpd(pf);

    ASSERT_EQ(a->bocpd_cooldown, 3000, "Cooldown set");
    ASSERT_NEAR(a->sigma_vol_scale, 1.5, 1e-9, "sigma_vol_scale NOT reset by notify");
    ASSERT_EQ(a->high_vol_mode, 1, "high_vol_mode NOT reset by notify");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: OUTPUT DIAGNOSTICS
 *============================================================================*/

static void test_output_diagnostics(void)
{
    printf("\n=== Test: Output Diagnostics ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->enable_regime_feedback = 1;
    a->bocpd_cooldown_duration = 100;
    a->sigma_vol_scale = 1.3;
    a->bocpd_cooldown = 50;

    /* Set up vol EMAs to maintain high_vol_mode */
    a->high_vol_mode = 1;
    a->vol_short_ema = 0.02; /* ratio = 2.0, above exit threshold */
    a->vol_long_ema = 0.01;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 128.0;
    out.vol_mean = 0.02; /* Maintain high vol */

    pf2d_adaptive_tick(pf, &out);

    ASSERT_NEAR(out.sigma_vol_scale, a->sigma_vol_scale, 0.01,
                "Output sigma_vol_scale matches state");
    ASSERT_GT(out.ess_ema, 0, "Output ess_ema populated");
    ASSERT_EQ(out.high_vol_mode, 1, "Output high_vol_mode correct");
    ASSERT_EQ(out.regime_feedback_active, 0, "Feedback inactive during cooldown");
    ASSERT_EQ(out.bocpd_cooldown_remaining, 49, "Cooldown remaining decremented");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: STREAK RESET ON MODE CHANGE
 *============================================================================*/

static void test_streak_reset_on_mode_change(void)
{
    printf("\n=== Test: Streak Reset on Mode Change ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->low_streak_thresh = 10;

    /* Pre-seed ESS EMA to low value */
    a->ess_ratio_ema = 0.2;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;

    /* Build up low ESS streak */
    for (int i = 0; i < 8; i++)
    {
        out.ess = 0.2 * pf->n_particles;
        pf2d_adaptive_tick(pf, &out);
    }
    ASSERT_EQ(a->low_ess_streak, 8, "Low streak built up");

    /* Pre-seed ESS EMA to high value before switching */
    a->ess_ratio_ema = 0.8;

    /* Switch to high ESS - should reset low streak */
    out.ess = 0.8 * pf->n_particles;
    pf2d_adaptive_tick(pf, &out);

    ASSERT_EQ(a->low_ess_streak, 0, "Low streak reset on high ESS");
    ASSERT_EQ(a->high_ess_streak, 1, "High streak started");

    destroy_test_pf(pf);
}

/*============================================================================
 * TEST: EDGE CASES
 *============================================================================*/

static void test_zero_ess(void)
{
    printf("\n=== Test: Zero ESS Handling ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->low_streak_thresh = 5;

    /* Pre-seed ESS EMA to zero (simulating previous zero ESS) */
    a->ess_ratio_ema = 0.0;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.vol_mean = 0.01;
    out.ess = 0; /* Edge case */

    /* Should not crash, should count as low ESS */
    for (int i = 0; i < 10; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_GT(a->low_ess_streak, 5, "Zero ESS counted as low");
    ASSERT_GT(a->sigma_vol_scale, 1.0, "Scale increased");

    destroy_test_pf(pf);
}

static void test_very_high_vol(void)
{
    printf("\n=== Test: Very High Vol ===\n");

    PF2D *pf = create_test_pf(256, 4);
    PF2DAdaptive *a = &pf->adaptive;

    a->vol_short_ema = 0.01;
    a->vol_long_ema = 0.01;

    PF2DOutput out;
    memset(&out, 0, sizeof(out));
    out.ess = 0.5 * pf->n_particles;
    out.vol_mean = 1.0; /* 100x baseline */

    for (int i = 0; i < 50; i++)
    {
        pf2d_adaptive_tick(pf, &out);
    }

    ASSERT_EQ(a->high_vol_mode, 1, "Entered high-vol mode on extreme vol");

    destroy_test_pf(pf);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(void)
{
    printf("========================================\n");
    printf("  PF2D Adaptive Self-Calibration Tests\n");
    printf("========================================\n");

    /* Initialization */
    test_adaptive_init();

    /* ESS Scaling */
    test_ess_scaling_low_ess();
    test_ess_scaling_high_ess();
    test_ess_scaling_bounds();
    test_ess_scaling_decay();
    test_ess_scaling_disabled();

    /* Vol Detection */
    test_vol_detection_enter();
    test_vol_detection_exit();
    test_vol_detection_hysteresis();
    test_vol_detection_disabled();

    /* Regime Feedback */
    test_regime_feedback_ema();
    test_regime_feedback_cooldown();
    test_regime_feedback_disabled();

    /* PMMH Integration */
    test_reset_after_pmcmc();
    test_bocpd_notification();

    /* Output Diagnostics */
    test_output_diagnostics();

    /* Streak Behavior */
    test_streak_reset_on_mode_change();

    /* Edge Cases */
    test_zero_ess();
    test_very_high_vol();

    /* Summary */
    printf("\n========================================\n");
    printf("  Results: %d/%d tests passed\n", tests_passed, tests_run);
    printf("========================================\n");

    return (tests_passed == tests_run) ? 0 : 1;
}