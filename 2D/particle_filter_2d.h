/**
 * @file particle_filter_2d.h
 * @brief 2D Particle Filter with Stochastic Volatility
 *
 * State: [price, log_vol]
 * - Price follows regime-dependent drift + stochastic vol
 * - Log-vol follows mean-reverting process per regime
 *
 * Part of trading stack: SSA → BOCPD → PF2D → Kelly
 */

#ifndef PARTICLE_FILTER_2D_H
#define PARTICLE_FILTER_2D_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

    /*============================================================================
     * PRECISION CONFIGURATION
     *============================================================================*/

#define PF2D_USE_FLOAT

#ifdef PF2D_USE_FLOAT
    typedef float pf2d_real;
#define PF2D_REAL_SIZE 4
#define pf2d_vExp vsExp
#define pf2d_vSqr vsSqr
#define pf2d_cblas_scal cblas_sscal
#define pf2d_cblas_asum cblas_sasum
#define pf2d_cblas_dot cblas_sdot
#define pf2d_cblas_axpy cblas_saxpy
#define pf2d_RngGaussian vsRngGaussian
#define pf2d_RngUniform vsRngUniform
#else
typedef double pf2d_real;
#define PF2D_REAL_SIZE 8
#define pf2d_vExp vdExp
#define pf2d_vSqr vdSqr
#define pf2d_cblas_scal cblas_dscal
#define pf2d_cblas_asum cblas_dasum
#define pf2d_cblas_dot cblas_ddot
#define pf2d_cblas_axpy cblas_daxpy
#define pf2d_RngGaussian vdRngGaussian
#define pf2d_RngUniform vdRngUniform
#endif

    /*============================================================================
     * CONFIGURATION
     *============================================================================*/

#define PF2D_MAX_REGIMES 8
#define PF2D_ALIGN 64
#define PF2D_MAX_THREADS 128

/* Regime lookup table */
#define PF2D_REGIME_LUT_SIZE 1024

/* BLAS threshold */
#define PF2D_BLAS_THRESHOLD 4000

/* Adaptive resampling */
#define PF2D_RESAMPLE_THRESH_MIN 0.3
#define PF2D_RESAMPLE_THRESH_MAX 0.7
#define PF2D_RESAMPLE_THRESH_DEFAULT 0.5

    /*============================================================================
     * RESAMPLING METHOD SELECTION
     *
     * PF2D_RESAMPLE_SYSTEMATIC:
     *   Standard systematic resampling. O(N) with cumsum.
     *   Low variance but can create exact duplicates.
     *
     * PF2D_RESAMPLE_RESIDUAL:
     *   Deterministic copies for floor(N*w[i]), then systematic on remainder.
     *   Lower variance than pure systematic.
     *
     * PF2D_RESAMPLE_REGULARIZED:
     *   Systematic + kernel jitter post-resample.
     *   Prevents sample impoverishment in continuous state spaces.
     *   RECOMMENDED for stochastic volatility models.
     *
     * PF2D_RESAMPLE_RESIDUAL_REGULARIZED:
     *   Residual + kernel jitter. Best of both.
     *============================================================================*/

    typedef enum
    {
        PF2D_RESAMPLE_SYSTEMATIC = 0,          /* Default: standard systematic */
        PF2D_RESAMPLE_RESIDUAL = 1,            /* Lower variance */
        PF2D_RESAMPLE_REGULARIZED = 2,         /* Adds jitter - recommended for SV */
        PF2D_RESAMPLE_RESIDUAL_REGULARIZED = 3 /* Combined */
    } PF2DResampleMethod;

    /*============================================================================
     * HIGH-N MODE (for offline research with 50k+ particles)
     *
     * Enable with: -DPF2D_HIGH_N
     *
     * Adds:
     *   - Parallel prefix sum (cumsum)
     *   - Parallel binary search in resampling
     *   - Threshold-based algorithm selection
     *============================================================================*/

#ifdef PF2D_HIGH_N
#define PF2D_PARALLEL_CUMSUM_THRESH 8000  /* Use parallel cumsum above this */
#define PF2D_PARALLEL_SEARCH_THRESH 16000 /* Use parallel search above this */
#endif

    /*============================================================================
     * PCG32 RNG
     *============================================================================*/

    typedef struct
    {
        uint64_t state;
        uint64_t inc;
    } pf2d_pcg32_t;

    static inline void pf2d_pcg32_seed(pf2d_pcg32_t *rng, uint64_t seed, uint64_t seq)
    {
        rng->state = 0U;
        rng->inc = (seq << 1u) | 1u;
        rng->state += seed;
        rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    }

    static inline uint32_t pf2d_pcg32_random(pf2d_pcg32_t *rng)
    {
        uint64_t oldstate = rng->state;
        rng->state = oldstate * 6364136223846793005ULL + rng->inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
    }

    static inline pf2d_real pf2d_pcg32_uniform(pf2d_pcg32_t *rng)
    {
        return (pf2d_real)pf2d_pcg32_random(rng) / (pf2d_real)4294967296.0;
    }

    static inline pf2d_real pf2d_pcg32_gaussian(pf2d_pcg32_t *rng)
    {
        pf2d_real u1 = pf2d_pcg32_uniform(rng);
        pf2d_real u2 = pf2d_pcg32_uniform(rng);
        if (u1 < (pf2d_real)1e-10)
            u1 = (pf2d_real)1e-10;
        return (pf2d_real)sqrt(-2.0 * log((double)u1)) * (pf2d_real)cos(6.283185307179586 * (double)u2);
    }

    /*============================================================================
     * STRUCTURES
     *============================================================================*/

    /**
     * Per-regime parameters for 2D state dynamics
     */
    typedef struct
    {
        /* Price dynamics */
        pf2d_real drift; /* Regime drift (can be modulated by SSA) */

        /* Log-volatility dynamics: log_vol = (1-θ)*log_vol + θ*μ + σ*noise */
        pf2d_real theta_vol; /* Mean-reversion speed */
        pf2d_real mu_vol;    /* Long-term mean of log-vol */
        pf2d_real sigma_vol; /* Vol-of-vol */

        /* Price-vol correlation (leverage effect) */
        pf2d_real rho; /* Correlation: -1 to 1, 0 = uncorrelated */

        /* Precomputed for speed */
        pf2d_real one_minus_theta;       /* 1 - theta_vol */
        pf2d_real theta_mu;              /* theta_vol * mu_vol */
        pf2d_real sqrt_one_minus_rho_sq; /* sqrt(1 - rho^2) */
    } PF2DRegimeParams;

    /**
     * SSA features for drift modulation
     */
    typedef struct
    {
        pf2d_real trend;
        pf2d_real volatility_scale; /* Multiplier for sigma_vol */
    } PF2DSSAFeatures;

    /**
     * Regime probabilities with precomputed cumulative
     */
    typedef struct
    {
        pf2d_real probs[PF2D_MAX_REGIMES];
        pf2d_real cumprobs[PF2D_MAX_REGIMES];
        int n_regimes;
    } PF2DRegimeProbs;

/*============================================================================
 * ADAPTIVE SELF-CALIBRATION
 *
 * Enables the filter to automatically adjust based on its own performance:
 *   - ESS-driven σ_vol scaling: widen/tighten dynamics based on filter health
 *   - Regime posterior feedback: smoothly update regime LUT from particle weights
 *   - Volatility regime detection: boost exploration in high-vol conditions
 *============================================================================*/

/* Adaptive tuning defaults */
#define PF2D_ADAPTIVE_ESS_EMA_ALPHA ((pf2d_real)0.01)
#define PF2D_ADAPTIVE_ESS_LOW_THRESH ((pf2d_real)0.3)
#define PF2D_ADAPTIVE_ESS_HIGH_THRESH ((pf2d_real)0.7)
#define PF2D_ADAPTIVE_LOW_ESS_STREAK 1000
#define PF2D_ADAPTIVE_HIGH_ESS_STREAK 2000
#define PF2D_ADAPTIVE_SIGMA_SCALE_MIN ((pf2d_real)0.5)
#define PF2D_ADAPTIVE_SIGMA_SCALE_MAX ((pf2d_real)2.0)
#define PF2D_ADAPTIVE_SIGMA_DECAY_ALPHA ((pf2d_real)0.001)

#define PF2D_ADAPTIVE_VOL_ENTER_RATIO ((pf2d_real)1.8)
#define PF2D_ADAPTIVE_VOL_EXIT_RATIO ((pf2d_real)1.15)
#define PF2D_ADAPTIVE_VOL_SHORT_ALPHA ((pf2d_real)0.1)
#define PF2D_ADAPTIVE_VOL_LONG_ALPHA ((pf2d_real)0.01)
#define PF2D_ADAPTIVE_HIGH_VOL_BW_SCALE ((pf2d_real)1.5)

#define PF2D_ADAPTIVE_REGIME_EMA_ALPHA ((pf2d_real)0.02)
#define PF2D_ADAPTIVE_LUT_UPDATE_INTERVAL 50
#define PF2D_ADAPTIVE_BOCPD_COOLDOWN 3000

    /**
     * @brief Adaptive state for self-calibration
     */
    typedef struct
    {
        /*====================================================================
         * ESS-DRIVEN SIGMA_VOL SCALING
         *====================================================================*/
        pf2d_real ess_ema;         /* Exponential moving average of ESS */
        pf2d_real ess_ratio_ema;   /* EMA of ESS/N ratio */
        int low_ess_streak;        /* Consecutive ticks with ESS < low_thresh */
        int high_ess_streak;       /* Consecutive ticks with ESS > high_thresh */
        pf2d_real sigma_vol_scale; /* Multiplier on σ_vol [0.5, 2.0] */

        /* Configurable thresholds */
        pf2d_real ess_low_thresh;  /* ESS/N below this is "low" (default 0.3) */
        pf2d_real ess_high_thresh; /* ESS/N above this is "high" (default 0.7) */
        int low_streak_thresh;     /* Ticks before widening (default 1000) */
        int high_streak_thresh;    /* Ticks before tightening (default 2000) */

        /*====================================================================
         * REGIME POSTERIOR FEEDBACK
         *====================================================================*/
        pf2d_real regime_ema[PF2D_MAX_REGIMES]; /* Smoothed regime probs */
        int lut_update_countdown;               /* Ticks until next LUT rebuild */
        int bocpd_cooldown;                     /* Ticks until regime feedback re-enabled */
        int lut_update_interval;                /* Rebuild LUT every N ticks (default 50) */
        int bocpd_cooldown_duration;            /* Cooldown after BOCPD (default 3000) */

        /*====================================================================
         * VOLATILITY REGIME DETECTION
         *====================================================================*/
        pf2d_real vol_short_ema;   /* Fast EMA of volatility */
        pf2d_real vol_long_ema;    /* Slow EMA of volatility */
        int high_vol_mode;         /* 1 = elevated volatility detected */
        pf2d_real vol_enter_ratio; /* Enter high-vol when short/long > this */
        pf2d_real vol_exit_ratio;  /* Exit high-vol when short/long < this */

        /* Saved defaults for restoration */
        pf2d_real base_resample_threshold;
        pf2d_real base_bandwidth_price;
        pf2d_real base_bandwidth_vol;

        /*====================================================================
         * FEATURE FLAGS
         *====================================================================*/
        int enable_sigma_scaling;   /* ESS-driven σ_vol scaling */
        int enable_regime_feedback; /* Posterior → LUT feedback */
        int enable_vol_detection;   /* Auto high-vol mode */

    } PF2DAdaptive;

    /**
     * Main 2D Particle Filter structure
     */
    typedef struct
    {
        /*========================================================================
         * PARTICLE STATE - Structure of Arrays (SoA) layout
         *========================================================================*/

        pf2d_real *prices;     /* Current price estimate per particle [n_particles] */
        pf2d_real *prices_tmp; /* Double-buffer for resampling swap [n_particles] */

        pf2d_real *log_vols;     /* Log-volatility state per particle [n_particles] */
        pf2d_real *log_vols_tmp; /* Double-buffer for resampling swap [n_particles] */

        pf2d_real *weights;     /* Normalized importance weights [n_particles] */
        pf2d_real *log_weights; /* Unnormalized log-weights [n_particles] */

        pf2d_real *cumsum; /* Cumulative sum of weights for resampling [n_particles] */

        int *regimes;     /* Current regime per particle [n_particles] */
        int *regimes_tmp; /* Double-buffer for resampling swap [n_particles] */

        /*========================================================================
         * SCRATCH BUFFERS - Temporary storage reused each tick
         *========================================================================*/

        pf2d_real *scratch1;       /* Gaussian noise z1 / jitter buffer [n_particles] */
        pf2d_real *scratch2;       /* Gaussian noise z2 / jitter buffer [n_particles] */
        pf2d_real *regime_uniform; /* Dedicated buffer for regime uniform RNG [n_particles] */
        int *indices;              /* Resampling indices [n_particles] */
        int *resample_count;       /* Residual resampling: deterministic copies [n_particles] */

        /*========================================================================
         * RANDOM NUMBER GENERATION
         *========================================================================*/

        VSLStreamStatePtr mkl_rng[PF2D_MAX_THREADS];
        pf2d_pcg32_t pcg[PF2D_MAX_THREADS];
        int use_pcg;
        int n_threads;

        /*========================================================================
         * REGIME SYSTEM
         *========================================================================*/

        uint8_t regime_lut[PF2D_REGIME_LUT_SIZE];
        PF2DRegimeParams regimes_params[PF2D_MAX_REGIMES];
        int n_regimes;

        /*========================================================================
         * OBSERVATION MODEL
         *========================================================================*/

        pf2d_real obs_variance;
        pf2d_real inv_obs_variance;
        pf2d_real neg_half_inv_var;

        /*========================================================================
         * ADAPTIVE RESAMPLING
         *========================================================================*/

        pf2d_real resample_threshold;
        pf2d_real vol_ema;
        pf2d_real vol_baseline;

        /*========================================================================
         * RESAMPLING METHOD CONFIGURATION
         *========================================================================*/

        PF2DResampleMethod resample_method; /* Selected resampling algorithm */

        /* Regularization (kernel jitter) parameters
         * Bandwidth scales with particle spread and ESS:
         *   h_effective = h_base * scale_factor
         *   scale_factor adapts based on ESS/N ratio */
        pf2d_real reg_bandwidth_price; /* Base bandwidth for price jitter */
        pf2d_real reg_bandwidth_vol;   /* Base bandwidth for log-vol jitter */
        pf2d_real reg_scale_min;       /* Min scaling when ESS is healthy (e.g., 0.1) */
        pf2d_real reg_scale_max;       /* Max scaling when ESS is low (e.g., 0.6) */

        pf2d_real last_ess; /* Cached ESS for bandwidth adaptation */

        /*========================================================================
         * DIMENSIONS
         *========================================================================*/

        int n_particles;
        pf2d_real uniform_weight;

        /*========================================================================
         * ADAPTIVE SELF-CALIBRATION STATE
         *========================================================================*/

        PF2DAdaptive adaptive;

    } PF2D;

    /**
     * @brief Output from pf2d_update() - estimates and diagnostics for one tick
     */
    typedef struct
    {
        pf2d_real price_mean;
        pf2d_real price_variance;
        pf2d_real log_vol_mean;
        pf2d_real log_vol_variance;
        pf2d_real vol_mean;
        pf2d_real ess;
        pf2d_real regime_probs[PF2D_MAX_REGIMES];
        int dominant_regime;
        int resampled;

        /* Adaptive diagnostics */
        pf2d_real sigma_vol_scale;    /* Current σ_vol multiplier */
        pf2d_real ess_ema;            /* Smoothed ESS */
        int high_vol_mode;            /* 1 if in high-vol mode */
        int regime_feedback_active;   /* 1 if regime feedback is running */
        int bocpd_cooldown_remaining; /* Ticks until regime feedback re-enabled */
    } PF2DOutput;

    /*============================================================================
     * API
     *============================================================================*/

    /* Create/destroy */
    PF2D *pf2d_create(int n_particles, int n_regimes);
    void pf2d_destroy(PF2D *pf);

    /* Configuration */
    void pf2d_set_observation_variance(PF2D *pf, pf2d_real var);

    void pf2d_set_regime_params(PF2D *pf, int regime,
                                pf2d_real drift,
                                pf2d_real theta_vol,
                                pf2d_real mu_vol,
                                pf2d_real sigma_vol,
                                pf2d_real rho);

    void pf2d_precompute(PF2D *pf, const PF2DSSAFeatures *ssa);

    void pf2d_set_regime_probs(PF2DRegimeProbs *rp, const pf2d_real *probs, int n);
    void pf2d_build_regime_lut(PF2D *pf, const PF2DRegimeProbs *rp);

    void pf2d_enable_pcg(PF2D *pf, int enable);

    void pf2d_set_resample_adaptive(PF2D *pf, pf2d_real baseline_vol);

    /*========================================================================
     * RESAMPLING METHOD API
     *========================================================================*/

    /**
     * @brief Set resampling method
     * @param method One of PF2D_RESAMPLE_* enum values
     *
     * For stochastic volatility models, PF2D_RESAMPLE_REGULARIZED or
     * PF2D_RESAMPLE_RESIDUAL_REGULARIZED are recommended to prevent
     * sample impoverishment.
     */
    void pf2d_set_resample_method(PF2D *pf, PF2DResampleMethod method);

    /**
     * @brief Set regularization (kernel jitter) bandwidths
     * @param h_price  Base bandwidth for price jitter (e.g., 0.0001 for small tick)
     * @param h_vol    Base bandwidth for log-vol jitter (e.g., 0.01)
     *
     * Bandwidth is scaled by ESS ratio:
     *   - When ESS ≈ N (healthy): scale ≈ reg_scale_min (less jitter)
     *   - When ESS << N (degenerate): scale ≈ reg_scale_max (more jitter)
     *
     * Rule of thumb: set h_price ≈ expected tick size / 10
     *                set h_vol ≈ sigma_vol / 5
     */
    void pf2d_set_regularization_bandwidth(PF2D *pf, pf2d_real h_price, pf2d_real h_vol);

    /**
     * @brief Set regularization scaling range
     * @param scale_min  Minimum scale factor when ESS is healthy (default 0.1)
     * @param scale_max  Maximum scale factor when ESS is low (default 0.6)
     */
    void pf2d_set_regularization_scaling(PF2D *pf, pf2d_real scale_min, pf2d_real scale_max);

    /* Initialize particles */
    void pf2d_initialize(PF2D *pf, pf2d_real price0, pf2d_real price_spread,
                         pf2d_real log_vol0, pf2d_real log_vol_spread);

    /* Core operations */
    void pf2d_propagate(PF2D *pf, const PF2DRegimeProbs *rp);
    void pf2d_update_weights(PF2D *pf, pf2d_real observation);
    pf2d_real pf2d_effective_sample_size(const PF2D *pf);
    void pf2d_resample(PF2D *pf);
    int pf2d_resample_if_needed(PF2D *pf);

    /* Estimates */
    pf2d_real pf2d_price_mean(const PF2D *pf);
    pf2d_real pf2d_price_variance(const PF2D *pf);
    pf2d_real pf2d_log_vol_mean(const PF2D *pf);
    pf2d_real pf2d_log_vol_variance(const PF2D *pf);
    pf2d_real pf2d_vol_mean(const PF2D *pf);

    /* Full update */
    PF2DOutput pf2d_update(PF2D *pf, pf2d_real observation, const PF2DRegimeProbs *rp);

    /* Warmup */
    void pf2d_warmup(PF2D *pf);
    void pf2d_disable_denormals_all_threads(void);

    /*============================================================================
     * ADAPTIVE SELF-CALIBRATION API
     *============================================================================*/

    /**
     * @brief Initialize adaptive state with defaults
     *
     * Called automatically by pf2d_create(). Call again to reset.
     * Default state:
     *   - sigma_scaling: ON
     *   - vol_detection: ON
     *   - regime_feedback: OFF (opt-in due to BOCPD interaction)
     */
    void pf2d_adaptive_init(PF2D *pf);

    /**
     * @brief Reset adaptive state after PMCMC parameter update
     *
     * MUST be called after applying PMCMC results to prevent double-adaptation.
     * Resets sigma_vol_scale to 1.0 and clears ESS streaks.
     */
    void pf2d_adaptive_reset_after_pmcmc(PF2D *pf);

    /**
     * @brief Notify that BOCPD detected a changepoint
     *
     * Disables regime feedback for cooldown period.
     * Does NOT reset sigma_vol_scale or high_vol_mode.
     */
    void pf2d_adaptive_notify_bocpd(PF2D *pf);

    /**
     * @brief Run adaptive tick - called from pf2d_update()
     *
     * Runs all enabled adaptive features and populates diagnostic output.
     */
    void pf2d_adaptive_tick(PF2D *pf, PF2DOutput *out);

    /* Feature enable/disable */
    void pf2d_adaptive_enable_sigma_scaling(PF2D *pf, int enable);
    void pf2d_adaptive_enable_regime_feedback(PF2D *pf, int enable);
    void pf2d_adaptive_enable_vol_detection(PF2D *pf, int enable);

    /**
     * @brief Enable/disable all adaptive features at once
     */
    void pf2d_adaptive_set_mode(PF2D *pf, int enable_all);

    /* Tuning */
    void pf2d_adaptive_set_ess_thresholds(PF2D *pf,
                                          pf2d_real low_thresh,
                                          pf2d_real high_thresh,
                                          int low_streak,
                                          int high_streak);

    void pf2d_adaptive_set_vol_thresholds(PF2D *pf,
                                          pf2d_real enter_ratio,
                                          pf2d_real exit_ratio);

    void pf2d_adaptive_set_regime_feedback_params(PF2D *pf,
                                                  int lut_interval,
                                                  int bocpd_cooldown);

    /* Query current state */
    PF2DAdaptive pf2d_adaptive_get_state(const PF2D *pf);

    /* Debug */
    void pf2d_print_config(const PF2D *pf);

#ifdef __cplusplus
}
#endif

#endif /* PARTICLE_FILTER_2D_H */