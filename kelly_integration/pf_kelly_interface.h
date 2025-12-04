/**
 * @file pf_kelly_interface.h
 * @brief Bridge between Particle Filter and Kelly Criterion
 *
 * Converts particle filter output (empirical distribution) to Kelly-compatible
 * estimates (velocity, volatility, uncertainty).
 *
 * Part of trading stack: SSA → BOCPD → PF → Kelly
 */

#ifndef PF_KELLY_INTERFACE_H
#define PF_KELLY_INTERFACE_H

#include "particle_filter.h"
#include "kelly.h"

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * BRIDGE STRUCTURE
 *============================================================================*/

typedef struct {
    /* Derived estimates */
    pf_real velocity;             /* Estimated return (price change rate) */
    pf_real volatility;           /* From particle std dev */
    pf_real log_vol;              /* log(volatility) for Kelly compatibility */
    
    /* Uncertainties (from particle spread) */
    pf_real velocity_std;         /* Uncertainty in velocity estimate */
    pf_real volatility_std;       /* Uncertainty in volatility estimate */
    
    /* Health metrics */
    pf_real ess_ratio;            /* ESS / n_particles (0-1) */
    pf_real regime_concentration; /* Max regime prob (high = confident) */
    
    /* Regime info */
    int dominant_regime;
    pf_real regime_probs[PF_MAX_REGIMES];
    
} PFKellyBridge;

/*============================================================================
 * STATE TRACKER
 *============================================================================
 * 
 * PF gives us price level estimates. Kelly needs velocity (return).
 * Tracker maintains state to compute velocity from price changes.
 *============================================================================*/

typedef struct {
    pf_real prev_mean;            /* Previous tick's mean estimate */
    pf_real prev_std;             /* Previous tick's std dev */
    pf_real velocity_ema;         /* Smoothed velocity (EMA) */
    pf_real volatility_ema;       /* Smoothed volatility (EMA) */
    pf_real velocity_var_ema;     /* Smoothed velocity variance for uncertainty */
    pf_real alpha;                /* EMA smoothing factor (e.g., 0.1) */
    int tick_count;               /* Number of updates */
    int initialized;
} PFKellyTracker;

/*============================================================================
 * TRACKER FUNCTIONS
 *============================================================================*/

/**
 * @brief Initialize the tracker
 * 
 * @param t       Tracker to initialize
 * @param alpha   EMA smoothing factor (0.05-0.2 typical)
 *                Lower = smoother but slower to react
 *                Higher = noisier but faster to react
 */
static inline void pf_kelly_tracker_init(PFKellyTracker* t, pf_real alpha) {
    t->prev_mean = 0.0;
    t->prev_std = 0.0;
    t->velocity_ema = 0.0;
    t->volatility_ema = 0.01;     /* Initial guess */
    t->velocity_var_ema = 0.0001; /* Initial guess */
    t->alpha = alpha;
    t->tick_count = 0;
    t->initialized = 0;
}

/**
 * @brief Reset tracker (e.g., after gap or regime change)
 */
static inline void pf_kelly_tracker_reset(PFKellyTracker* t) {
    t->velocity_ema = 0.0;
    t->velocity_var_ema = 0.0001;
    t->tick_count = 0;
    t->initialized = 0;
}

/*============================================================================
 * BRIDGE UPDATE
 *============================================================================*/

/**
 * @brief Update bridge from particle filter output
 * 
 * Call this every tick after pf_update().
 * 
 * @param bridge       Output bridge structure
 * @param tracker      State tracker (maintains history)
 * @param pf_out       Particle filter output from pf_update()
 * @param n_particles  Number of particles (for ESS ratio)
 */
static inline void pf_kelly_bridge_update(
    PFKellyBridge* bridge,
    PFKellyTracker* tracker,
    const PFOutput* pf_out,
    int n_particles)
{
    pf_real mean = pf_out->mean;
    pf_real variance = pf_out->variance;
    pf_real std_dev = (pf_real)sqrt((double)variance);
    
    /* First tick: initialize and return zeros */
    if (!tracker->initialized) {
        tracker->prev_mean = mean;
        tracker->prev_std = std_dev;
        tracker->volatility_ema = std_dev;
        tracker->initialized = 1;
        tracker->tick_count = 1;
        
        bridge->velocity = 0.0;
        bridge->volatility = std_dev;
        bridge->log_vol = (pf_real)log((double)std_dev + 1e-10);
        bridge->velocity_std = std_dev;
        bridge->volatility_std = std_dev * (pf_real)0.5;
        bridge->ess_ratio = pf_out->ess / (pf_real)n_particles;
        bridge->dominant_regime = 0;
        bridge->regime_concentration = 0.0;
        
        for (int r = 0; r < PF_MAX_REGIMES; r++) {
            bridge->regime_probs[r] = pf_out->regime_probs[r];
        }
        return;
    }
    
    tracker->tick_count++;
    
    /* Raw velocity = price change */
    pf_real raw_velocity = mean - tracker->prev_mean;
    
    /* EMA smoothing of velocity */
    pf_real alpha = tracker->alpha;
    tracker->velocity_ema = alpha * raw_velocity + 
                            ((pf_real)1.0 - alpha) * tracker->velocity_ema;
    
    /* EMA smoothing of volatility */
    tracker->volatility_ema = alpha * std_dev + 
                              ((pf_real)1.0 - alpha) * tracker->volatility_ema;
    
    /* Velocity variance for uncertainty estimation (EMA of squared deviations) */
    pf_real vel_dev = raw_velocity - tracker->velocity_ema;
    tracker->velocity_var_ema = alpha * vel_dev * vel_dev + 
                                ((pf_real)1.0 - alpha) * tracker->velocity_var_ema;
    
    /* Fill bridge structure */
    bridge->velocity = tracker->velocity_ema;
    bridge->volatility = tracker->volatility_ema;
    bridge->log_vol = (pf_real)log((double)tracker->volatility_ema + 1e-10);
    
    /* Uncertainty estimates */
    bridge->velocity_std = (pf_real)sqrt((double)tracker->velocity_var_ema);
    bridge->volatility_std = (pf_real)fabs((double)(std_dev - tracker->prev_std));
    
    /* Health metrics */
    bridge->ess_ratio = pf_out->ess / (pf_real)n_particles;
    
    /* Find dominant regime */
    pf_real max_prob = 0.0;
    int max_regime = 0;
    for (int r = 0; r < PF_MAX_REGIMES; r++) {
        bridge->regime_probs[r] = pf_out->regime_probs[r];
        if (pf_out->regime_probs[r] > max_prob) {
            max_prob = pf_out->regime_probs[r];
            max_regime = r;
        }
    }
    bridge->dominant_regime = max_regime;
    bridge->regime_concentration = max_prob;
    
    /* Update tracker state for next tick */
    tracker->prev_mean = mean;
    tracker->prev_std = std_dev;
}

/*============================================================================
 * KELLY COMPUTATION
 *============================================================================*/

/**
 * @brief Compute Kelly position from bridge
 * 
 * Applies:
 *   - Weak signal filter
 *   - Base Kelly calculation
 *   - ESS health scaling
 *   - Regime-based scaling
 * 
 * @param bridge    Populated bridge structure
 * @param nu        Student-t degrees of freedom (INFINITY for Gaussian)
 * @param fraction  Kelly fraction (0.5 = half Kelly)
 * @return Final position size (-1 to +1 typical, can exceed with leverage)
 */
static inline double pf_kelly_compute(
    const PFKellyBridge* bridge,
    double nu,
    double fraction)
{
    double mu = (double)bridge->velocity;
    double sigma = (double)bridge->volatility;
    double mu_std = (double)bridge->velocity_std;
    
    /* Weak signal filter: require |μ| > threshold × σ_μ */
    if (mu_std > 1e-10) {
        double signal_strength = fabs(mu) / mu_std;
        if (signal_strength < KELLY_MIN_SIGNAL_RATIO) {
            return 0.0;
        }
    }
    
    /* Base Kelly calculation */
    double f = kelly_simple(mu, sigma, nu, fraction);
    
    /* ESS health scaling
     * Low ESS = particle degeneracy = model uncertainty
     * Scale down position when ESS is low */
    double ess_scale = (double)bridge->ess_ratio;
    if (ess_scale < 0.5) {
        /* Below 50% ESS, start scaling down */
        ess_scale = ess_scale / 0.5;  /* Linear scale 0-1 */
    } else {
        ess_scale = 1.0;
    }
    if (ess_scale < 0.1) {
        ess_scale = 0.0;  /* Kill position if ESS < 10% */
    }
    
    /* Regime-based scaling
     * Reduce position in high-volatility or jump regimes */
    double regime_scale = 1.0;
    regime_scale -= 0.3 * (double)bridge->regime_probs[2];  /* High-vol: regime 2 */
    regime_scale -= 0.5 * (double)bridge->regime_probs[3];  /* Jump: regime 3 */
    if (regime_scale < 0.1) {
        regime_scale = 0.1;
    }
    
    return f * ess_scale * regime_scale;
}

/**
 * @brief Full Kelly result from bridge (more detailed output)
 * 
 * @param bridge    Populated bridge structure  
 * @param nu        Student-t degrees of freedom
 * @param fraction  Kelly fraction
 * @param result    Output Kelly result structure
 */
static inline void pf_kelly_compute_full(
    const PFKellyBridge* bridge,
    double nu,
    double fraction,
    KellyResult* result)
{
    double mu = (double)bridge->velocity;
    double sigma = (double)bridge->volatility;
    double mu_std = (double)bridge->velocity_std;
    double sigma_std = (double)bridge->volatility_std;
    
    /* Populate result basics */
    result->expected_return = mu;
    result->volatility = sigma;
    result->sharpe = (sigma > 1e-10) ? mu / sigma : 0.0;
    result->mu_uncertainty = mu_std;
    result->vol_uncertainty = sigma_std;
    result->capped_long = false;
    result->capped_short = false;
    
    /* Weak signal filter */
    if (mu_std > 1e-10) {
        double signal_strength = fabs(mu) / mu_std;
        if (signal_strength < KELLY_MIN_SIGNAL_RATIO) {
            result->f_full = 0.0;
            result->f_half = 0.0;
            result->f_adjusted = 0.0;
            result->f_final = 0.0;
            return;
        }
    }
    
    /* Variance with tail adjustment */
    double var = sigma * sigma;
    var = kelly_tail_variance(var, nu);
    if (var < KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY) {
        var = KELLY_MIN_VOLATILITY * KELLY_MIN_VOLATILITY;
    }
    
    /* Kelly fractions */
    result->f_full = mu / var;
    result->f_half = result->f_full * 0.5;
    result->f_adjusted = result->f_full * fraction;
    
    /* Short penalty */
    if (result->f_adjusted < 0) {
        result->f_adjusted *= KELLY_SHORT_PENALTY;
    }
    
    /* ESS and regime scaling */
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
 * CONVENIENCE: DIRECT FROM PF OUTPUT
 *============================================================================*/

/**
 * @brief One-shot Kelly from PF output (updates tracker internally)
 * 
 * Convenience function for simple usage.
 */
static inline double pf_to_kelly(
    PFKellyTracker* tracker,
    const PFOutput* pf_out,
    int n_particles,
    double nu,
    double fraction)
{
    PFKellyBridge bridge;
    pf_kelly_bridge_update(&bridge, tracker, pf_out, n_particles);
    return pf_kelly_compute(&bridge, nu, fraction);
}

/*============================================================================
 * DEBUG / LOGGING
 *============================================================================*/

/**
 * @brief Print bridge state for debugging
 */
static inline void pf_kelly_bridge_print(const PFKellyBridge* bridge) {
    printf("PF-Kelly Bridge:\n");
    printf("  Velocity:     %.6f (std: %.6f)\n", 
           (double)bridge->velocity, (double)bridge->velocity_std);
    printf("  Volatility:   %.6f (std: %.6f)\n", 
           (double)bridge->volatility, (double)bridge->volatility_std);
    printf("  Log-vol:      %.4f\n", (double)bridge->log_vol);
    printf("  ESS ratio:    %.2f%%\n", (double)bridge->ess_ratio * 100.0);
    printf("  Dominant:     regime %d (%.1f%%)\n", 
           bridge->dominant_regime, (double)bridge->regime_concentration * 100.0);
    printf("  Regimes:      [%.1f%%, %.1f%%, %.1f%%, %.1f%%]\n",
           (double)bridge->regime_probs[0] * 100.0,
           (double)bridge->regime_probs[1] * 100.0,
           (double)bridge->regime_probs[2] * 100.0,
           (double)bridge->regime_probs[3] * 100.0);
}

#ifdef __cplusplus
}
#endif

#endif /* PF_KELLY_INTERFACE_H */
