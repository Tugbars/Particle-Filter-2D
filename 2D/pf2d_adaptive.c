/**
 * @file pf2d_adaptive.c
 * @brief Adaptive Self-Calibration for 2D Particle Filter
 *
 * Three features:
 *   1. ESS-driven σ_vol scaling - widen/tighten dynamics based on filter health
 *   2. Regime posterior feedback - update LUT from particle weights  
 *   3. Volatility regime detection - auto high-vol mode with hysteresis
 *
 * Designed to work with BOCPD + PMMH:
 *   - BOCPD fires → notify → disables regime feedback for cooldown
 *   - PMMH completes → reset → clears σ_vol scaling to prevent double-adapt
 */

#include "particle_filter_2d.h"
#include <string.h>

/*============================================================================
 * INITIALIZATION
 *============================================================================*/

void pf2d_adaptive_init(PF2D *pf)
{
    PF2DAdaptive *a = &pf->adaptive;
    
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
    for (int r = 0; r < PF2D_MAX_REGIMES; r++) {
        a->regime_ema[r] = (pf2d_real)1.0 / pf->n_regimes;
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
    
    /* Feature flags - defaults:
     *   sigma_scaling: ON (safe, effective)
     *   vol_detection: ON (safe with hysteresis)
     *   regime_feedback: OFF (dangerous without BOCPD gating) */
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
    
    /* Reset σ_vol scaling - PMCMC already calibrated σ_vol */
    a->sigma_vol_scale = (pf2d_real)1.0;
    a->low_ess_streak = 0;
    a->high_ess_streak = 0;
    
    /* Do NOT reset:
     *   - high_vol_mode (volatility persists across parameter changes)
     *   - regime_ema (regime distribution is still relevant)
     *   - vol_short_ema, vol_long_ema (continuous measurement)
     */
}

void pf2d_adaptive_notify_bocpd(PF2D *pf)
{
    /* Disable regime feedback for cooldown period */
    pf->adaptive.bocpd_cooldown = pf->adaptive.bocpd_cooldown_duration;
    
    /* Do NOT reset:
     *   - sigma_vol_scale (reset happens when PMCMC completes)
     *   - high_vol_mode (vol shocks survive regime changes)
     */
}

/*============================================================================
 * INTERNAL UPDATE FUNCTIONS
 *============================================================================*/

/**
 * @brief Update ESS-driven σ_vol scaling
 * 
 * Logic:
 *   - Track smoothed ESS ratio (ESS/N)
 *   - If persistently low: widen σ_vol (particles collapsing)
 *   - If persistently high: tighten σ_vol (can improve precision)
 *   - When healthy: slowly decay back to 1.0
 */
static void pf2d_adaptive_update_sigma_scaling(PF2D *pf, pf2d_real current_ess)
{
    PF2DAdaptive *a = &pf->adaptive;
    pf2d_real ratio = current_ess / pf->n_particles;
    
    /* Update EMAs (α = 0.01 for smooth tracking) */
    a->ess_ema = PF2D_ADAPTIVE_ESS_EMA_ALPHA * current_ess + 
                 ((pf2d_real)1.0 - PF2D_ADAPTIVE_ESS_EMA_ALPHA) * a->ess_ema;
    a->ess_ratio_ema = PF2D_ADAPTIVE_ESS_EMA_ALPHA * ratio + 
                       ((pf2d_real)1.0 - PF2D_ADAPTIVE_ESS_EMA_ALPHA) * a->ess_ratio_ema;
    
    /* Track streaks with hysteresis */
    if (a->ess_ratio_ema < a->ess_low_thresh) {
        a->low_ess_streak++;
        a->high_ess_streak = 0;
    } else if (a->ess_ratio_ema > a->ess_high_thresh) {
        a->high_ess_streak++;
        a->low_ess_streak = 0;
    } else {
        /* In healthy range - decay both streaks and scale */
        a->low_ess_streak = 0;
        a->high_ess_streak = 0;
        
        /* Slow decay toward 1.0 (α = 0.001 → ~700 tick half-life) */
        a->sigma_vol_scale = PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA * (pf2d_real)1.0 +
                             ((pf2d_real)1.0 - PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA) * a->sigma_vol_scale;
    }
    
    /* Adjust scale based on streaks */
    if (a->low_ess_streak > a->low_streak_thresh) {
        /* ESS persistently low → widen dynamics */
        a->sigma_vol_scale *= (pf2d_real)1.02;
        if (a->sigma_vol_scale > PF2D_ADAPTIVE_SIGMA_SCALE_MAX) {
            a->sigma_vol_scale = PF2D_ADAPTIVE_SIGMA_SCALE_MAX;
        }
    } else if (a->high_ess_streak > a->high_streak_thresh) {
        /* ESS persistently high → can tighten */
        a->sigma_vol_scale *= (pf2d_real)0.99;
        if (a->sigma_vol_scale < PF2D_ADAPTIVE_SIGMA_SCALE_MIN) {
            a->sigma_vol_scale = PF2D_ADAPTIVE_SIGMA_SCALE_MIN;
        }
    }
}

/**
 * @brief Update volatility regime detection
 * 
 * Uses fast/slow EMA ratio with hysteresis to detect elevated volatility.
 * In high-vol mode: lower resample threshold, increase bandwidth.
 */
static void pf2d_adaptive_update_vol_mode(PF2D *pf, pf2d_real current_vol)
{
    PF2DAdaptive *a = &pf->adaptive;
    
    /* Update volatility EMAs */
    a->vol_short_ema = PF2D_ADAPTIVE_VOL_SHORT_ALPHA * current_vol +
                       ((pf2d_real)1.0 - PF2D_ADAPTIVE_VOL_SHORT_ALPHA) * a->vol_short_ema;
    a->vol_long_ema = PF2D_ADAPTIVE_VOL_LONG_ALPHA * current_vol +
                      ((pf2d_real)1.0 - PF2D_ADAPTIVE_VOL_LONG_ALPHA) * a->vol_long_ema;
    
    /* Compute ratio (protect against division by zero) */
    pf2d_real vol_ratio = a->vol_short_ema / (a->vol_long_ema + (pf2d_real)1e-10);
    
    /* State machine with hysteresis */
    if (!a->high_vol_mode && vol_ratio > a->vol_enter_ratio) {
        /* Enter high-vol mode */
        a->high_vol_mode = 1;
        
        /* Adjust filter parameters for high volatility */
        pf->resample_threshold = PF2D_RESAMPLE_THRESH_MIN;
        pf->reg_bandwidth_price = a->base_bandwidth_price * PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE;
        pf->reg_bandwidth_vol = a->base_bandwidth_vol * PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE;
        
    } else if (a->high_vol_mode && vol_ratio < a->vol_exit_ratio) {
        /* Exit high-vol mode */
        a->high_vol_mode = 0;
        
        /* Restore base parameters */
        pf->resample_threshold = a->base_resample_threshold;
        pf->reg_bandwidth_price = a->base_bandwidth_price;
        pf->reg_bandwidth_vol = a->base_bandwidth_vol;
    }
}

/**
 * @brief Update regime posterior feedback
 * 
 * Smoothly updates regime LUT based on particle posterior distribution.
 * Only runs when BOCPD cooldown has expired.
 */
static void pf2d_adaptive_update_regime_feedback(PF2D *pf, const PF2DOutput *out)
{
    PF2DAdaptive *a = &pf->adaptive;
    
    /* Update regime EMAs (slow: α = 0.02) */
    for (int r = 0; r < pf->n_regimes; r++) {
        a->regime_ema[r] = PF2D_ADAPTIVE_REGIME_EMA_ALPHA * out->regime_probs[r] +
                           ((pf2d_real)1.0 - PF2D_ADAPTIVE_REGIME_EMA_ALPHA) * a->regime_ema[r];
    }
    
    /* Rebuild LUT periodically */
    a->lut_update_countdown--;
    if (a->lut_update_countdown <= 0) {
        /* Build new LUT from smoothed posterior */
        PF2DRegimeProbs rp;
        rp.n_regimes = pf->n_regimes;
        
        /* Normalize (in case of numerical drift) */
        pf2d_real sum = (pf2d_real)0.0;
        for (int r = 0; r < pf->n_regimes; r++) {
            sum += a->regime_ema[r];
        }
        pf2d_real inv_sum = (pf2d_real)1.0 / (sum + (pf2d_real)1e-10);
        
        pf2d_real cumsum = (pf2d_real)0.0;
        for (int r = 0; r < pf->n_regimes; r++) {
            rp.probs[r] = a->regime_ema[r] * inv_sum;
            cumsum += rp.probs[r];
            rp.cumprobs[r] = cumsum;
        }
        rp.cumprobs[pf->n_regimes - 1] = (pf2d_real)1.0;  /* Ensure exact */
        
        /* Rebuild LUT */
        pf2d_build_regime_lut(pf, &rp);
        
        /* Reset countdown */
        a->lut_update_countdown = a->lut_update_interval;
    }
}

/*============================================================================
 * MAIN TICK FUNCTION - Called from pf2d_update()
 *============================================================================*/

void pf2d_adaptive_tick(PF2D *pf, PF2DOutput *out)
{
    PF2DAdaptive *a = &pf->adaptive;
    
    /* ESS-driven σ_vol scaling */
    if (a->enable_sigma_scaling) {
        pf2d_adaptive_update_sigma_scaling(pf, out->ess);
    }
    
    /* Volatility regime detection */
    if (a->enable_vol_detection) {
        pf2d_adaptive_update_vol_mode(pf, out->vol_mean);
    }
    
    /* Regime posterior feedback (with BOCPD cooldown) */
    if (a->enable_regime_feedback) {
        if (a->bocpd_cooldown > 0) {
            a->bocpd_cooldown--;
        } else {
            pf2d_adaptive_update_regime_feedback(pf, out);
        }
    }
    
    /* Populate adaptive diagnostics in output */
    out->sigma_vol_scale = a->sigma_vol_scale;
    out->ess_ema = a->ess_ema;
    out->high_vol_mode = a->high_vol_mode;
    out->regime_feedback_active = a->enable_regime_feedback && (a->bocpd_cooldown == 0);
    out->bocpd_cooldown_remaining = a->bocpd_cooldown;
}

/*============================================================================
 * ENABLE/DISABLE API
 *============================================================================*/

void pf2d_adaptive_enable_sigma_scaling(PF2D *pf, int enable)
{
    pf->adaptive.enable_sigma_scaling = enable;
    if (!enable) {
        /* Reset scale when disabling */
        pf->adaptive.sigma_vol_scale = (pf2d_real)1.0;
        pf->adaptive.low_ess_streak = 0;
        pf->adaptive.high_ess_streak = 0;
    }
}

void pf2d_adaptive_enable_regime_feedback(PF2D *pf, int enable)
{
    pf->adaptive.enable_regime_feedback = enable;
    if (enable) {
        /* Reset countdown when enabling */
        pf->adaptive.lut_update_countdown = pf->adaptive.lut_update_interval;
    }
}

void pf2d_adaptive_enable_vol_detection(PF2D *pf, int enable)
{
    pf->adaptive.enable_vol_detection = enable;
    if (!enable && pf->adaptive.high_vol_mode) {
        /* Exit high-vol mode and restore parameters */
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
    /* Regime feedback stays OFF by default even in "enable all" mode
     * due to BOCPD interaction risks. Enable explicitly if needed. */
    if (!enable_all) {
        pf2d_adaptive_enable_regime_feedback(pf, 0);
    }
}

/*============================================================================
 * TUNING API
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
