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

#ifdef PF2D_USE_FLOAT
    typedef float pf2d_real;
#define PF2D_REAL_SIZE 4
#define pf2d_vExp vsExp
#define pf2d_vSqr vsSqr
#define pf2d_cblas_scal cblas_sscal
#define pf2d_cblas_asum cblas_sasum
#define pf2d_cblas_dot cblas_sdot
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
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
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

    /**
     * Main 2D Particle Filter structure
     */
    typedef struct
    {
        /* Particle state - SoA layout */
        pf2d_real *prices;
        pf2d_real *prices_tmp;
        pf2d_real *log_vols;
        pf2d_real *log_vols_tmp;
        pf2d_real *weights;
        pf2d_real *log_weights;
        pf2d_real *cumsum;
        int *regimes;
        int *regimes_tmp;

        /* Scratch buffers */
        pf2d_real *scratch1;
        pf2d_real *scratch2;
        int *indices; /* For HIGH_N resampling - proper int buffer */

        /* RNG streams */
        VSLStreamStatePtr mkl_rng[PF2D_MAX_THREADS];
        pf2d_pcg32_t pcg[PF2D_MAX_THREADS];
        int use_pcg;
        int n_threads;

        /* Regime lookup table */
        uint8_t regime_lut[PF2D_REGIME_LUT_SIZE];

        /* Per-regime parameters */
        PF2DRegimeParams regimes_params[PF2D_MAX_REGIMES];
        int n_regimes;

        /* Observation model */
        pf2d_real obs_variance;
        pf2d_real inv_obs_variance;
        pf2d_real neg_half_inv_var;

        /* Adaptive resampling */
        pf2d_real resample_threshold;
        pf2d_real vol_ema;
        pf2d_real vol_baseline;

        /* Dimensions */
        int n_particles;
        pf2d_real uniform_weight;

    } PF2D;

    /**
     * Output from pf2d_update()
     */
    typedef struct
    {
        /* Price estimates */
        pf2d_real price_mean;
        pf2d_real price_variance;

        /* Volatility estimates */
        pf2d_real log_vol_mean;     /* E[log_vol] across particles */
        pf2d_real log_vol_variance; /* Uncertainty in log-vol */
        pf2d_real vol_mean;         /* E[exp(log_vol)] */

        /* Health metrics */
        pf2d_real ess;
        pf2d_real regime_probs[PF2D_MAX_REGIMES];
        int dominant_regime;
        int resampled;
    } PF2DOutput;

    /*============================================================================
     * API
     *============================================================================*/

    /* Create/destroy */
    PF2D *pf2d_create(int n_particles, int n_regimes);
    void pf2d_destroy(PF2D *pf);

    /* Configuration */
    void pf2d_set_observation_variance(PF2D *pf, pf2d_real var);

    /**
     * Set regime parameters
     * @param drift       Price drift for this regime
     * @param theta_vol   Log-vol mean-reversion speed (0.01 - 0.2)
     * @param mu_vol      Long-term log-vol mean (e.g., log(0.01) ≈ -4.6)
     * @param sigma_vol   Vol-of-vol (keep < 2*theta_vol for stability)
     * @param rho         Price-vol correlation (-1 to 1, 0 = uncorrelated)
     */
    void pf2d_set_regime_params(PF2D *pf, int regime,
                                pf2d_real drift,
                                pf2d_real theta_vol,
                                pf2d_real mu_vol,
                                pf2d_real sigma_vol,
                                pf2d_real rho);

    /* Precompute SSA-derived terms */
    void pf2d_precompute(PF2D *pf, const PF2DSSAFeatures *ssa);

    /* Regime probability setup */
    void pf2d_set_regime_probs(PF2DRegimeProbs *rp, const pf2d_real *probs, int n);
    void pf2d_build_regime_lut(PF2D *pf, const PF2DRegimeProbs *rp);

    /* RNG selection */
    void pf2d_enable_pcg(PF2D *pf, int enable);

    /* Adaptive resampling */
    void pf2d_set_resample_adaptive(PF2D *pf, pf2d_real baseline_vol);

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

    /* Debug */
    void pf2d_print_config(const PF2D *pf);

#ifdef __cplusplus
}
#endif

#endif /* PARTICLE_FILTER_2D_H */
