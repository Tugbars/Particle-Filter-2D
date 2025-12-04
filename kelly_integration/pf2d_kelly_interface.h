/**
 * @file pf2d_kelly_interface.h
 * @brief Bridge between 2D Particle Filter and Kelly Criterion
 *
 * Much simpler than 1D interface because PF2D directly estimates volatility.
 * No EMA smoothing needed - we have real stochastic vol estimates.
 *
 * Part of trading stack: SSA → BOCPD → PF2D → Kelly
 */

#ifndef PF2D_KELLY_INTERFACE_H
#define PF2D_KELLY_INTERFACE_H

#include "particle_filter_2d.h"
#include "kelly.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * BRIDGE STRUCTURE
 *============================================================================*/

typedef struct {
    /* Direct from PF2D - no smoothing needed */
    pf2d_real velocity;           /* Price change from prev tick */
    pf2d_real volatility;         /* E[exp(log_vol)] from particles */
    pf2d_real log_vol_mean;       /* E[log_vol] for Kelly log-normal calcs */
    pf2d_real log_vol_std;        /* Std(log_vol) - volatility uncertainty */
    
    /* Price uncertainty */
    pf2d_real price_std;          /* Std(price) across particles */
    
    /* Health metrics */
    pf2d_real ess_ratio;
    pf2d_real regime_concentration;
    
    /* Regime info */
    int dominant_regime;
    pf2d_real regime_probs[PF2D_MAX_REGIMES];
    
} PF2DKellyBridge;

/*============================================================================
 * VELOCITY TRACKER (minimal - just tracks previous price)
 *============================================================================*/

typedef struct {
    pf2d_real prev_price_mean;
    int initialized;
} PF2DKellyTracker;

static inline void pf2d_kelly_tracker_init(PF2DKellyTracker* t) {
    t->prev_price_mean = 0.0;
    t->initialized = 0;
}

/*============================================================================
 * BRIDGE UPDATE
 *============================================================================*/

/**
 * @brief Update bridge from 2D PF output
 * 
 * Direct mapping - no smoothing needed because PF2D tracks volatility.
 */
static inline void pf2d_kelly_bridge_update(
    PF2DKellyBridge* bridge,
    PF2DKellyTracker* tracker,
    const PF2DOutput* pf_out,
    int n_particles)
{
    /* First tick: initialize */
    if (!tracker->initialized) {
        tracker->prev_price_mean = pf_out->price_mean;
        tracker->initialized = 1;
        
        bridge->velocity = 0.0;
        bridge->volatility = pf_out->vol_mean;
        bridge->log_vol_mean = pf_out->log_vol_mean;
        bridge->log_vol_std = (pf2d_real)sqrt((double)pf_out->log_vol_variance);
        bridge->price_std = (pf2d_real)sqrt((double)pf_out->price_variance);
        bridge->ess_ratio = pf_out->ess / (pf2d_real)n_particles;
        bridge->dominant_regime = pf_out->dominant_regime;
        bridge->regime_concentration = pf_out->regime_probs[pf_out->dominant_regime];
        
        for (int r = 0; r < PF2D_MAX_REGIMES; r++) {
            bridge->regime_probs[r] = pf_out->regime_probs[r];
        }
        return;
    }
    
    /* Velocity = price change (direct, no EMA needed) */
    bridge->velocity = pf_out->price_mean - tracker->prev_price_mean;
    
    /* Direct volatility from PF2D */
    bridge->volatility = pf_out->vol_mean;
    bridge->log_vol_mean = pf_out->log_vol_mean;
    bridge->log_vol_std = (pf2d_real)sqrt((double)pf_out->log_vol_variance);
    
    /* Price uncertainty */
    bridge->price_std = (pf2d_real)sqrt((double)pf_out->price_variance);
    
    /* Health */
    bridge->ess_ratio = pf_out->ess / (pf2d_real)n_particles;
    
    /* Regime info */
    bridge->dominant_regime = pf_out->dominant_regime;
    pf2d_real max_prob = 0;
    for (int r = 0; r < PF2D_MAX_REGIMES; r++) {
        bridge->regime_probs[r] = pf_out->regime_probs[r];
        if (pf_out->regime_probs[r] > max_prob) {
            max_prob = pf_out->regime_probs[r];
        }
    }
    bridge->regime_concentration = max_prob;
    
    /* Update tracker */
    tracker->prev_price_mean = pf_out->price_mean;
}

/*============================================================================
 * KELLY COMPUTATION
 *============================================================================*/

/**
 * @brief Compute Kelly position from 2D PF bridge
 * 
 * Uses proper Bayesian Kelly with log-vol uncertainty:
 *   E[σ²] = exp(2μ_lv + 2σ_lv²)
 * 
 * This naturally shrinks position when vol uncertainty is high.
 */
static inline double pf2d_kelly_compute(
    const PF2DKellyBridge* bridge,
    double nu,              /* Student-t df, INFINITY for Gaussian */
    double fraction)        /* Kelly fraction, e.g., 0.5 */
{
    double mu = (double)bridge->velocity;
    double mu_lv = (double)bridge->log_vol_mean;
    double sigma_lv = (double)bridge->log_vol_std;
    
    /* Bayesian expected variance: E[σ²] = exp(2μ + 2σ²) */
    double expected_var = kelly_expected_variance(mu_lv, sigma_lv);
    
    /* Tail adjustment */
    expected_var = kelly_tail_variance(expected_var, nu);
    
    /* Floor variance */
    if (expected_var < KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY) {
        expected_var = KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY;
    }
    
    /* Weak signal filter using price uncertainty */
    double price_std = (double)bridge->price_std;
    if (price_std > 1e-10) {
        double signal_strength = fabs(mu) / price_std;
        if (signal_strength < KELLY_MIN_SIGNAL_RATIO) {
            return 0.0;
        }
    }
    
    /* Bayesian Kelly: f* = E[μ] / E[σ²] */
    double f = fraction * mu / expected_var;
    
    /* Short penalty */
    if (f < 0) {
        f *= KELLY_SHORT_PENALTY;
    }
    
    /* ESS health scaling */
    double ess_scale = (double)bridge->ess_ratio;
    if (ess_scale < 0.5) {
        ess_scale = ess_scale / 0.5;
    } else {
        ess_scale = 1.0;
    }
    if (ess_scale < 0.1) {
        ess_scale = 0.0;
    }
    
    /* Regime scaling */
    double regime_scale = 1.0;
    regime_scale -= 0.3 * (double)bridge->regime_probs[2];  /* High-vol */
    regime_scale -= 0.5 * (double)bridge->regime_probs[3];  /* Jump */
    if (regime_scale < 0.1) regime_scale = 0.1;
    
    f *= ess_scale * regime_scale;
    
    /* Position limits */
    if (f > KELLY_MAX_LEVERAGE) f = KELLY_MAX_LEVERAGE;
    if (f < KELLY_MIN_LEVERAGE) f = KELLY_MIN_LEVERAGE;
    
    return f;
}

/**
 * @brief Full Kelly result from 2D PF bridge
 */
static inline void pf2d_kelly_compute_full(
    const PF2DKellyBridge* bridge,
    double nu,
    double fraction,
    KellyResult* result)
{
    double mu = (double)bridge->velocity;
    double mu_lv = (double)bridge->log_vol_mean;
    double sigma_lv = (double)bridge->log_vol_std;
    double sigma_point = (double)bridge->volatility;
    
    /* Populate result basics */
    result->expected_return = mu;
    result->volatility = sigma_point;
    result->sharpe = (sigma_point > 1e-10) ? mu / sigma_point : 0.0;
    result->mu_uncertainty = (double)bridge->price_std;
    result->vol_uncertainty = sigma_lv;
    result->capped_long = false;
    result->capped_short = false;
    
    /* Bayesian expected variance */
    double expected_var = kelly_expected_variance(mu_lv, sigma_lv);
    expected_var = kelly_tail_variance(expected_var, nu);
    
    if (expected_var < KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY) {
        expected_var = KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY;
    }
    
    /* Weak signal filter */
    double price_std = (double)bridge->price_std;
    if (price_std > 1e-10) {
        double signal_strength = fabs(mu) / price_std;
        if (signal_strength < KELLY_MIN_SIGNAL_RATIO) {
            result->f_full = 0.0;
            result->f_half = 0.0;
            result->f_adjusted = 0.0;
            result->f_final = 0.0;
            return;
        }
    }
    
    /* Kelly fractions */
    result->f_full = mu / expected_var;
    result->f_half = result->f_full * 0.5;
    result->f_adjusted = result->f_full * fraction;
    
    /* Short penalty */
    if (result->f_adjusted < 0) {
        result->f_adjusted *= KELLY_SHORT_PENALTY;
    }
    
    /* Health scaling */
    double ess_scale = (double)bridge->ess_ratio;
    if (ess_scale < 0.5) ess_scale = ess_scale / 0.5;
    else ess_scale = 1.0;
    if (ess_scale < 0.1) ess_scale = 0.0;
    
    double regime_scale = 1.0;
    regime_scale -= 0.3 * (double)bridge->regime_probs[2];
    regime_scale -= 0.5 * (double)bridge->regime_probs[3];
    if (regime_scale < 0.1) regime_scale = 0.1;
    
    result->f_final = result->f_adjusted * ess_scale * regime_scale;
    
    /* Position limits */
    if (result->f_final > KELLY_MAX_LEVERAGE) {
        result->f_final = KELLY_MAX_LEVERAGE;
        result->capped_long = true;
    } else if (result->f_final < KELLY_MIN_LEVERAGE) {
        result->f_final = KELLY_MIN_LEVERAGE;
        result->capped_short = true;
    }
}

/*============================================================================
 * CONVENIENCE FUNCTION
 *============================================================================*/

/**
 * @brief One-shot Kelly from PF2D output
 */
static inline double pf2d_to_kelly(
    PF2DKellyTracker* tracker,
    const PF2DOutput* pf_out,
    int n_particles,
    double nu,
    double fraction)
{
    PF2DKellyBridge bridge;
    pf2d_kelly_bridge_update(&bridge, tracker, pf_out, n_particles);
    return pf2d_kelly_compute(&bridge, nu, fraction);
}

/*============================================================================
 * DEBUG
 *============================================================================*/

static inline void pf2d_kelly_bridge_print(const PF2DKellyBridge* bridge) {
    printf("PF2D-Kelly Bridge:\n");
    printf("  Velocity:       %.6f\n", (double)bridge->velocity);
    printf("  Volatility:     %.6f (direct from PF)\n", (double)bridge->volatility);
    printf("  Log-vol:        %.4f ± %.4f\n", 
           (double)bridge->log_vol_mean, (double)bridge->log_vol_std);
    printf("  Price std:      %.6f\n", (double)bridge->price_std);
    printf("  ESS ratio:      %.2f%%\n", (double)bridge->ess_ratio * 100.0);
    printf("  Dominant:       regime %d (%.1f%%)\n",
           bridge->dominant_regime, (double)bridge->regime_concentration * 100.0);
    printf("  Regimes:        [%.1f%%, %.1f%%, %.1f%%, %.1f%%]\n",
           (double)bridge->regime_probs[0] * 100.0,
           (double)bridge->regime_probs[1] * 100.0,
           (double)bridge->regime_probs[2] * 100.0,
           (double)bridge->regime_probs[3] * 100.0);
    
    /* Show Bayesian variance calculation */
    double mu_lv = (double)bridge->log_vol_mean;
    double sigma_lv = (double)bridge->log_vol_std;
    double expected_var = exp(2.0 * mu_lv + 2.0 * sigma_lv * sigma_lv);
    double point_var = (double)(bridge->volatility * bridge->volatility);
    printf("  E[σ²] Bayesian: %.8f (vs point: %.8f, ratio: %.2f)\n",
           expected_var, point_var, expected_var / (point_var + 1e-10));
}

#ifdef __cplusplus
}
#endif

#endif /* PF2D_KELLY_INTERFACE_H */
