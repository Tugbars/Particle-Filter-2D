/**
 * @file pf2d_adaptive.c
 * @brief Adaptive Self-Calibration for 2D Particle Filter (OPTIMIZED)
 *
 * Three features:
 *   1. ESS-driven σ_vol scaling - widen/tighten dynamics based on filter health
 *   2. Regime posterior feedback - update LUT from particle weights
 *   3. Volatility regime detection - auto high-vol mode with hysteresis
 *
 * Optimizations applied:
 *   - Precomputed (1-α) constants
 *   - Reciprocal multiply instead of division
 *   - FMA-friendly EMA updates: x += α*(new - x)
 *   - Branchless streak counters
 *   - Cache-aligned hot data layout
 *   - Minimized branching in hot path
 *
 * Designed to work with BOCPD + PMMH:
 *   - BOCPD fires → notify → disables regime feedback for cooldown
 *   - PMMH completes → reset → clears σ_vol scaling to prevent double-adapt
 */

#include "particle_filter_2d.h"
#include <string.h>

/*============================================================================
 * PRECOMPUTED CONSTANTS (avoid runtime subtraction)
 *============================================================================*/

#define ESS_EMA_ONE_MINUS_ALPHA ((pf2d_real)(1.0 - PF2D_ADAPTIVE_ESS_EMA_ALPHA))
#define VOL_SHORT_ONE_MINUS_ALPHA ((pf2d_real)(1.0 - PF2D_ADAPTIVE_VOL_SHORT_ALPHA))
#define VOL_LONG_ONE_MINUS_ALPHA ((pf2d_real)(1.0 - PF2D_ADAPTIVE_VOL_LONG_ALPHA))
#define REGIME_EMA_ONE_MINUS_ALPHA ((pf2d_real)(1.0 - PF2D_ADAPTIVE_REGIME_EMA_ALPHA))
#define SIGMA_DECAY_ONE_MINUS_ALPHA ((pf2d_real)(1.0 - PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA))

/* Scale factors as constants */
#define SIGMA_SCALE_UP ((pf2d_real)1.02)
#define SIGMA_SCALE_DOWN ((pf2d_real)0.99)

/*============================================================================
 * INITIALIZATION
 *============================================================================*/

void pf2d_adaptive_init(PF2D *pf)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* Precompute reciprocal for hot path (avoid division) */
    a->inv_n_particles = (pf2d_real)1.0 / pf->n_particles;

    /* ESS tracking */
    a->ess_ema = (pf2d_real)pf->n_particles;
    a->ess_ratio_ema = (pf2d_real)1.0;
    a->low_ess_streak = 0;
    a->high_ess_streak = 0;
    a->sigma_vol_scale = (pf2d_real)1.0;

    /* ESS thresholds */
    a->ess_low_thresh = PF2D_ADAPTIVE_ESS_LOW_THRESH;
    a->ess_high_thresh = PF2D_ADAPTIVE_ESS_HIGH_THRESH;
    a->low_streak_thresh = PF2D_ADAPTIVE_LOW_ESS_STREAK;
    a->high_streak_thresh = PF2D_ADAPTIVE_HIGH_ESS_STREAK;

    /* Regime feedback */
    pf2d_real init_prob = (pf2d_real)1.0 / pf->n_regimes;
    for (int r = 0; r < PF2D_MAX_REGIMES; r++)
    {
        a->regime_ema[r] = init_prob;
    }
    a->lut_update_countdown = PF2D_ADAPTIVE_LUT_UPDATE_INTERVAL;
    a->bocpd_cooldown = 0;
    a->lut_update_interval = PF2D_ADAPTIVE_LUT_UPDATE_INTERVAL;
    a->bocpd_cooldown_duration = PF2D_ADAPTIVE_BOCPD_COOLDOWN;

    /* Volatility detection */
    a->vol_short_ema = pf->vol_baseline;
    a->vol_long_ema = pf->vol_baseline;
    a->high_vol_mode = 0;
    a->vol_enter_ratio = PF2D_ADAPTIVE_VOL_ENTER_RATIO;
    a->vol_exit_ratio = PF2D_ADAPTIVE_VOL_EXIT_RATIO;

    /* Save base values for restoration */
    a->base_resample_threshold = pf->resample_threshold;
    a->base_bandwidth_price = pf->reg_bandwidth_price;
    a->base_bandwidth_vol = pf->reg_bandwidth_vol;

    /* Feature flags */
    a->enable_sigma_scaling = 1;
    a->enable_vol_detection = 1;
    a->enable_regime_feedback = 0;
}

/*============================================================================
 * PMCMC / BOCPD INTEGRATION
 *============================================================================*/

void pf2d_adaptive_reset_after_pmcmc(PF2D *pf)
{
    PF2DAdaptive *a = &pf->adaptive;

    a->sigma_vol_scale = (pf2d_real)1.0;
    a->low_ess_streak = 0;
    a->high_ess_streak = 0;
}

void pf2d_adaptive_notify_bocpd(PF2D *pf)
{
    pf->adaptive.bocpd_cooldown = pf->adaptive.bocpd_cooldown_duration;
}

/*============================================================================
 * OPTIMIZED UPDATE FUNCTIONS
 *============================================================================*/

/**
 * @brief Update ESS-driven σ_vol scaling (OPTIMIZED)
 *
 * Optimizations:
 *   - FMA-friendly EMA: x += α*(new - x)
 *   - Reciprocal multiply for ratio
 *   - Branchless streak counters
 *   - Single branch for scale adjustment
 */
static inline void pf2d_adaptive_update_sigma_scaling(PF2D *pf, pf2d_real current_ess)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* Multiply by precomputed reciprocal (avoid division) */
    pf2d_real ratio = current_ess * a->inv_n_particles;

    /* FMA-friendly EMA updates: x += α*(new - x)
     * Compiles to single FMA instruction on modern CPUs */
    a->ess_ema += PF2D_ADAPTIVE_ESS_EMA_ALPHA * (current_ess - a->ess_ema);
    a->ess_ratio_ema += PF2D_ADAPTIVE_ESS_EMA_ALPHA * (ratio - a->ess_ratio_ema);

    /* Branchless classification */
    int is_low = (a->ess_ratio_ema < a->ess_low_thresh);
    int is_high = (a->ess_ratio_ema > a->ess_high_thresh);

    /* Branchless streak update:
     * - If is_low: low_streak++, high_streak = 0
     * - If is_high: high_streak++, low_streak = 0
     * - If healthy: both = 0 */
    a->low_ess_streak = is_low * (a->low_ess_streak + 1);
    a->high_ess_streak = is_high * (a->high_ess_streak + 1);

    /* Scale adjustment - at most one branch taken per tick */
    if (a->low_ess_streak > a->low_streak_thresh)
    {
        /* ESS persistently low → widen dynamics */
        a->sigma_vol_scale *= SIGMA_SCALE_UP;
        /* Branchless clamp */
        a->sigma_vol_scale = (a->sigma_vol_scale > PF2D_ADAPTIVE_SIGMA_SCALE_MAX)
                                 ? PF2D_ADAPTIVE_SIGMA_SCALE_MAX
                                 : a->sigma_vol_scale;
    }
    else if (a->high_ess_streak > a->high_streak_thresh)
    {
        /* ESS persistently high → tighten */
        a->sigma_vol_scale *= SIGMA_SCALE_DOWN;
        a->sigma_vol_scale = (a->sigma_vol_scale < PF2D_ADAPTIVE_SIGMA_SCALE_MIN)
                                 ? PF2D_ADAPTIVE_SIGMA_SCALE_MIN
                                 : a->sigma_vol_scale;
    }
    else if (!is_low && !is_high)
    {
        /* Healthy range - decay toward 1.0 using FMA form */
        a->sigma_vol_scale += PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA * ((pf2d_real)1.0 - a->sigma_vol_scale);
    }
}

/**
 * @brief Update volatility regime detection (OPTIMIZED)
 *
 * Optimizations:
 *   - FMA-friendly EMA
 *   - Single division computed once
 *   - Minimal state machine logic
 */
static inline void pf2d_adaptive_update_vol_mode(PF2D *pf, pf2d_real current_vol)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* FMA-friendly EMA updates */
    a->vol_short_ema += PF2D_ADAPTIVE_VOL_SHORT_ALPHA * (current_vol - a->vol_short_ema);
    a->vol_long_ema += PF2D_ADAPTIVE_VOL_LONG_ALPHA * (current_vol - a->vol_long_ema);

    /* Single division for ratio */
    pf2d_real vol_ratio = a->vol_short_ema / (a->vol_long_ema + (pf2d_real)1e-10);

    /* State machine with hysteresis - exactly one branch */
    if (!a->high_vol_mode)
    {
        if (vol_ratio > a->vol_enter_ratio)
        {
            /* Enter high-vol mode */
            a->high_vol_mode = 1;
            pf->resample_threshold = PF2D_RESAMPLE_THRESH_MIN;
            pf->reg_bandwidth_price = a->base_bandwidth_price * PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE;
            pf->reg_bandwidth_vol = a->base_bandwidth_vol * PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE;
        }
    }
    else
    {
        if (vol_ratio < a->vol_exit_ratio)
        {
            /* Exit high-vol mode */
            a->high_vol_mode = 0;
            pf->resample_threshold = a->base_resample_threshold;
            pf->reg_bandwidth_price = a->base_bandwidth_price;
            pf->reg_bandwidth_vol = a->base_bandwidth_vol;
        }
    }
}

/**
 * @brief Update regime posterior feedback (COLD PATH)
 *
 * Only called when feedback enabled AND cooldown expired.
 * Not optimized heavily as it runs infrequently.
 */
static void pf2d_adaptive_update_regime_feedback(PF2D *pf, const PF2DOutput *out)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* Update regime EMAs */
    for (int r = 0; r < pf->n_regimes; r++)
    {
        a->regime_ema[r] += PF2D_ADAPTIVE_REGIME_EMA_ALPHA * (out->regime_probs[r] - a->regime_ema[r]);
    }

    /* Rebuild LUT periodically */
    if (--a->lut_update_countdown <= 0)
    {
        PF2DRegimeProbs rp;
        rp.n_regimes = pf->n_regimes;

        /* Normalize */
        pf2d_real sum = (pf2d_real)0.0;
        for (int r = 0; r < pf->n_regimes; r++)
        {
            sum += a->regime_ema[r];
        }
        pf2d_real inv_sum = (pf2d_real)1.0 / (sum + (pf2d_real)1e-10);

        pf2d_real cumsum = (pf2d_real)0.0;
        for (int r = 0; r < pf->n_regimes; r++)
        {
            rp.probs[r] = a->regime_ema[r] * inv_sum;
            cumsum += rp.probs[r];
            rp.cumprobs[r] = cumsum;
        }
        rp.cumprobs[pf->n_regimes - 1] = (pf2d_real)1.0;

        pf2d_build_regime_lut(pf, &rp);
        a->lut_update_countdown = a->lut_update_interval;
    }
}

/*============================================================================
 * MAIN TICK FUNCTION - OPTIMIZED HOT PATH
 *============================================================================*/

void pf2d_adaptive_tick(PF2D *pf, PF2DOutput *out)
{
    PF2DAdaptive *a = &pf->adaptive;

    /* === HOT PATH (every tick) === */

    /* ESS-driven σ_vol scaling - inline for performance */
    if (a->enable_sigma_scaling)
    {
        pf2d_adaptive_update_sigma_scaling(pf, out->ess);
    }

    /* Volatility regime detection */
    if (a->enable_vol_detection)
    {
        pf2d_adaptive_update_vol_mode(pf, out->vol_mean);
    }

    /* === COLD PATH (rarely runs) === */

    /* Regime feedback - only when enabled AND cooldown expired */
    if (a->bocpd_cooldown > 0)
    {
        a->bocpd_cooldown--;
    }
    else if (a->enable_regime_feedback)
    {
        pf2d_adaptive_update_regime_feedback(pf, out);
    }

    /* === OUTPUT (direct assignment, no branches) === */
    out->sigma_vol_scale = a->sigma_vol_scale;
    out->ess_ema = a->ess_ema;
    out->high_vol_mode = a->high_vol_mode;
    /* Branchless AND using bitwise & */
    out->regime_feedback_active = a->enable_regime_feedback & (a->bocpd_cooldown == 0);
    out->bocpd_cooldown_remaining = a->bocpd_cooldown;
}

/*============================================================================
 * ENABLE/DISABLE API (Cold - called rarely)
 *============================================================================*/

void pf2d_adaptive_enable_sigma_scaling(PF2D *pf, int enable)
{
    pf->adaptive.enable_sigma_scaling = enable;
    if (!enable)
    {
        pf->adaptive.sigma_vol_scale = (pf2d_real)1.0;
        pf->adaptive.low_ess_streak = 0;
        pf->adaptive.high_ess_streak = 0;
    }
}

void pf2d_adaptive_enable_regime_feedback(PF2D *pf, int enable)
{
    pf->adaptive.enable_regime_feedback = enable;
    if (enable)
    {
        pf->adaptive.lut_update_countdown = pf->adaptive.lut_update_interval;
    }
}

void pf2d_adaptive_enable_vol_detection(PF2D *pf, int enable)
{
    pf->adaptive.enable_vol_detection = enable;
    if (!enable && pf->adaptive.high_vol_mode)
    {
        pf->adaptive.high_vol_mode = 0;
        pf->resample_threshold = pf->adaptive.base_resample_threshold;
        pf->reg_bandwidth_price = pf->adaptive.base_bandwidth_price;
        pf->reg_bandwidth_vol = pf->adaptive.base_bandwidth_vol;
    }
}

void pf2d_adaptive_set_mode(PF2D *pf, int enable_all)
{
    pf2d_adaptive_enable_sigma_scaling(pf, enable_all);
    pf2d_adaptive_enable_vol_detection(pf, enable_all);
    if (!enable_all)
    {
        pf2d_adaptive_enable_regime_feedback(pf, 0);
    }
}

/*============================================================================
 * TUNING API (Cold - called at setup)
 *============================================================================*/

void pf2d_adaptive_set_ess_thresholds(PF2D *pf,
                                      pf2d_real low_thresh,
                                      pf2d_real high_thresh,
                                      int low_streak,
                                      int high_streak)
{
    pf->adaptive.ess_low_thresh = low_thresh;
    pf->adaptive.ess_high_thresh = high_thresh;
    pf->adaptive.low_streak_thresh = low_streak;
    pf->adaptive.high_streak_thresh = high_streak;
}

void pf2d_adaptive_set_vol_thresholds(PF2D *pf,
                                      pf2d_real enter_ratio,
                                      pf2d_real exit_ratio)
{
    pf->adaptive.vol_enter_ratio = enter_ratio;
    pf->adaptive.vol_exit_ratio = exit_ratio;
}

void pf2d_adaptive_set_regime_feedback_params(PF2D *pf,
                                              int lut_interval,
                                              int bocpd_cooldown)
{
    pf->adaptive.lut_update_interval = lut_interval;
    pf->adaptive.bocpd_cooldown_duration = bocpd_cooldown;
}

PF2DAdaptive pf2d_adaptive_get_state(const PF2D *pf)
{
    return pf->adaptive;
}