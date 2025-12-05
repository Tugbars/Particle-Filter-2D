/**
 * @file pf2d_adaptive_defs.h
 * @brief Optimized PF2DAdaptive struct layout
 *
 * Add these to particle_filter_2d.h
 * 
 * Key optimization: Cache-aligned layout with hot data first
 */

#ifndef PF2D_ADAPTIVE_DEFS_H
#define PF2D_ADAPTIVE_DEFS_H

/*============================================================================
 * ADAPTIVE CONSTANTS (precomputed where possible)
 *============================================================================*/

/* ESS tracking */
#define PF2D_ADAPTIVE_ESS_EMA_ALPHA       0.01f   /* Slow EMA for ESS */
#define PF2D_ADAPTIVE_ESS_LOW_THRESH      0.3f    /* ESS/N below this = trouble */
#define PF2D_ADAPTIVE_ESS_HIGH_THRESH     0.7f    /* ESS/N above this = can tighten */
#define PF2D_ADAPTIVE_LOW_ESS_STREAK      10      /* Ticks below threshold to trigger */
#define PF2D_ADAPTIVE_HIGH_ESS_STREAK     50      /* Ticks above threshold to tighten */

/* σ_vol scaling */
#define PF2D_ADAPTIVE_SIGMA_SCALE_MIN     0.5f    /* Min scale factor */
#define PF2D_ADAPTIVE_SIGMA_SCALE_MAX     3.0f    /* Max scale factor */
#define PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA   0.001f  /* Decay rate toward 1.0 */

/* Volatility detection */
#define PF2D_ADAPTIVE_VOL_SHORT_ALPHA     0.05f   /* Fast EMA (~20 tick half-life) */
#define PF2D_ADAPTIVE_VOL_LONG_ALPHA      0.005f  /* Slow EMA (~140 tick half-life) */
#define PF2D_ADAPTIVE_VOL_ENTER_RATIO     1.5f    /* Short/Long > this = high vol */
#define PF2D_ADAPTIVE_VOL_EXIT_RATIO      1.1f    /* Short/Long < this = exit high vol */
#define PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE   1.5f    /* Bandwidth multiplier in high vol */

/* Regime feedback */
#define PF2D_ADAPTIVE_REGIME_EMA_ALPHA    0.02f   /* Slow update of regime priors */
#define PF2D_ADAPTIVE_LUT_UPDATE_INTERVAL 100     /* Ticks between LUT rebuilds */
#define PF2D_ADAPTIVE_BOCPD_COOLDOWN      200     /* Ticks to disable feedback after BOCPD */

/* Resampling */
#define PF2D_RESAMPLE_THRESH_MIN          0.3f    /* Aggressive resampling in high vol */

/*============================================================================
 * OPTIMIZED STRUCT LAYOUT
 * 
 * Hot data (accessed every tick) packed into first 64 bytes
 * Cold data (accessed rarely) follows
 *============================================================================*/

typedef struct {
    /* === CACHE LINE 1: Hot data (64 bytes) === */
    
    /* ESS tracking - accessed every tick */
    pf2d_real ess_ema;              /* 4/8 bytes */
    pf2d_real ess_ratio_ema;        /* 4/8 bytes */
    pf2d_real sigma_vol_scale;      /* 4/8 bytes */
    pf2d_real inv_n_particles;      /* 4/8 bytes - PRECOMPUTED RECIPROCAL */
    
    /* Volatility tracking - accessed every tick */
    pf2d_real vol_short_ema;        /* 4/8 bytes */
    pf2d_real vol_long_ema;         /* 4/8 bytes */
    
    /* Streak counters - accessed every tick */
    int low_ess_streak;             /* 4 bytes */
    int high_ess_streak;            /* 4 bytes */
    
    /* State flags - accessed every tick */
    int high_vol_mode;              /* 4 bytes */
    int bocpd_cooldown;             /* 4 bytes */
    
    /* Feature enables - checked every tick */
    int enable_sigma_scaling;       /* 4 bytes */
    int enable_vol_detection;       /* 4 bytes */
    int enable_regime_feedback;     /* 4 bytes */
    int lut_update_countdown;       /* 4 bytes */
    
    /* === CACHE LINE 2+: Cold data === */
    
    /* Thresholds - read-only after init */
    pf2d_real ess_low_thresh;
    pf2d_real ess_high_thresh;
    int low_streak_thresh;
    int high_streak_thresh;
    
    /* Vol detection thresholds */
    pf2d_real vol_enter_ratio;
    pf2d_real vol_exit_ratio;
    
    /* Base values for restoration */
    pf2d_real base_resample_threshold;
    pf2d_real base_bandwidth_price;
    pf2d_real base_bandwidth_vol;
    
    /* Regime feedback params */
    int lut_update_interval;
    int bocpd_cooldown_duration;
    
    /* Regime EMAs - only accessed when feedback enabled */
    pf2d_real regime_ema[PF2D_MAX_REGIMES];
    
} PF2DAdaptive;

/*============================================================================
 * OUTPUT STRUCT ADDITIONS
 * 
 * Add these fields to PF2DOutput
 *============================================================================*/

/*
    pf2d_real sigma_vol_scale;          // Current σ_vol multiplier
    pf2d_real ess_ema;                  // Smoothed ESS
    int high_vol_mode;                  // 1 if in high-vol mode
    int regime_feedback_active;         // 1 if feedback running
    int bocpd_cooldown_remaining;       // Ticks until feedback resumes
*/

#endif /* PF2D_ADAPTIVE_DEFS_H */
