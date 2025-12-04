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

    /**
     * Main 2D Particle Filter structure
     */
    typedef struct
    {
        /*========================================================================
         * PARTICLE STATE - Structure of Arrays (SoA) layout
         *
         * Why SoA over AoS?
         *   - SIMD (AVX2/512) can process 4-8 particles per instruction
         *   - Better cache utilization when accessing one field across particles
         *   - MKL vExp, cblas_* operate on contiguous arrays
         *========================================================================*/

        pf2d_real *prices;     /* Current price estimate per particle [n_particles] */
        pf2d_real *prices_tmp; /* Double-buffer for resampling swap [n_particles]
                                * Also reused as volatility buffer in propagate */

        pf2d_real *log_vols;     /* Log-volatility state per particle [n_particles]
                                  * Stored in log-space for numerical stability
                                  * Actual vol = exp(log_vol) */
        pf2d_real *log_vols_tmp; /* Double-buffer for resampling swap [n_particles] */

        pf2d_real *weights;     /* Normalized importance weights [n_particles]
                                 * Sum to 1.0, used for weighted estimates
                                 * Updated each tick via likelihood */
        pf2d_real *log_weights; /* Unnormalized log-weights [n_particles]
                                 * Intermediate storage before softmax normalization
                                 * Also reused for uniform RNG in propagate */

        pf2d_real *cumsum; /* Cumulative sum of weights for resampling [n_particles]
                            * cumsum[i] = sum(weights[0:i])
                            * Used for systematic resampling via binary search */

        int *regimes;     /* Current regime per particle [n_particles]
                           * Values in [0, n_regimes-1]
                           * Determines which dynamics parameters to use */
        int *regimes_tmp; /* Double-buffer for resampling swap [n_particles] */

        /*========================================================================
         * SCRATCH BUFFERS - Temporary storage reused each tick
         *========================================================================*/

        pf2d_real *scratch1; /* Gaussian noise z1 for price dynamics [n_particles] */
        pf2d_real *scratch2; /* Gaussian noise z2 for volatility dynamics [n_particles] */
        int *indices;        /* Resampling indices for HIGH_N mode [n_particles]
                              * indices[i] = which particle to copy to position i */

        /*========================================================================
         * RANDOM NUMBER GENERATION
         *
         * Each thread has its own RNG stream to avoid contention.
         * Two RNG options:
         *   - MKL SFMT: High quality, vectorized, better for large N
         *   - PCG32: Faster per-call, better for small N with inline generation
         *========================================================================*/

        VSLStreamStatePtr mkl_rng[PF2D_MAX_THREADS]; /* MKL RNG streams, one per thread
                                                      * Seeded deterministically: 42 + tid*8192 */
        pf2d_pcg32_t pcg[PF2D_MAX_THREADS];          /* PCG32 RNG states, one per thread
                                                      * Lightweight alternative to MKL */
        int use_pcg;                                 /* RNG selection flag:
                                                      *   0 = MKL SFMT (default for N >= BLAS_THRESHOLD)
                                                      *   1 = PCG32 (faster for small N) */
        int n_threads;                               /* Actual thread count (capped at PF2D_MAX_THREADS) */

        /*========================================================================
         * REGIME SYSTEM
         *
         * Multi-regime model allows different market states:
         *   Regime 0: Trending (positive drift, low vol)
         *   Regime 1: Mean-reverting (zero drift, stable vol)
         *   Regime 2: High volatility
         *   Regime 3: Jump/crisis
         *
         * Regime probabilities come from external model (e.g., BOCPD).
         *========================================================================*/

        uint8_t regime_lut[PF2D_REGIME_LUT_SIZE]; /* Lookup table: uniform[0,1] → regime
                                                   * Pre-built from regime probabilities
                                                   * Size 256 gives ~0.4% granularity
                                                   * Avoids branches in hot loop */

        PF2DRegimeParams regimes_params[PF2D_MAX_REGIMES]; /* Dynamics parameters per regime:
                                                            *   drift, theta_vol, mu_vol,
                                                            *   sigma_vol, rho, + precomputed */
        int n_regimes;                                     /* Active regime count (1-4 typically) */

        /*========================================================================
         * OBSERVATION MODEL
         *
         * Gaussian likelihood: p(obs | price) = N(obs; price, obs_variance)
         * Log-likelihood: -0.5 * (obs - price)² / obs_variance
         *
         * Precomputed terms avoid division in hot loop.
         *========================================================================*/

        pf2d_real obs_variance;     /* Observation noise variance σ²_obs
                                     * Represents measurement uncertainty
                                     * Larger = more tolerant of price mismatch */
        pf2d_real inv_obs_variance; /* 1 / obs_variance (precomputed) */
        pf2d_real neg_half_inv_var; /* -0.5 / obs_variance (precomputed)
                                     * Directly multiplied with squared error */

        /*========================================================================
         * ADAPTIVE RESAMPLING
         *
         * Resample when Effective Sample Size (ESS) drops below threshold.
         * Threshold adapts based on volatility regime:
         *   High vol → resample more often (lower threshold)
         *   Low vol  → resample less often (higher threshold)
         *========================================================================*/

        pf2d_real resample_threshold; /* Current ESS threshold as fraction of N
                                       * Resample if ESS < N * threshold
                                       * Range: [THRESH_MIN, THRESH_MAX] */
        pf2d_real vol_ema;            /* Exponential moving average of volatility
                                       * Smoothed estimate for threshold adaptation
                                       * Alpha = 0.05 (slow adaptation) */
        pf2d_real vol_baseline;       /* Baseline volatility for threshold scaling
                                       * Set via pf2d_set_resample_adaptive() */

        /*========================================================================
         * DIMENSIONS
         *========================================================================*/

        int n_particles;          /* Number of particles (N)
                                   * More particles = better accuracy, higher cost
                                   * Typical: 1000-10000 for trading */
        pf2d_real uniform_weight; /* 1.0 / n_particles (precomputed)
                                   * Used for weight reset after resampling */

    } PF2D;

    /**
     * @brief Output from pf2d_update() - estimates and diagnostics for one tick
     *
     * Contains:
     *   - Weighted mean/variance estimates for price and volatility
     *   - Filter health metrics (ESS, regime distribution)
     *   - Resampling flag for monitoring
     *
     * All estimates are importance-weighted:
     *   E[X] = Σ weights[i] * X[i]
     *   Var[X] = Σ weights[i] * (X[i] - E[X])²
     *
     * Usage:
     *   PF2DOutput out = pf2d_update(pf, observation, &regime_probs);
     *   double price_est = out.price_mean;
     *   double vol_est = out.vol_mean;
     *   if (out.ess < 1000) printf("Warning: low ESS\n");
     */
    typedef struct
    {
        /*========================================================================
         * PRICE ESTIMATES
         *========================================================================*/

        pf2d_real price_mean; /* Weighted mean price estimate
                               * E[price] = Σ w[i] * price[i]
                               * Primary output for trading signals */

        pf2d_real price_variance; /* Weighted variance of price estimate
                                   * Var[price] = Σ w[i] * (price[i] - mean)²
                                   * Represents estimation uncertainty
                                   * sqrt(variance) = standard error */

        /*========================================================================
         * VOLATILITY ESTIMATES
         *
         * Volatility is modeled in log-space for positivity and stability.
         * Three related quantities:
         *   log_vol_mean: E[log(vol)] - mean of log-volatility
         *   log_vol_variance: uncertainty in log-space
         *   vol_mean: E[vol] - actual volatility for position sizing
         *========================================================================*/

        pf2d_real log_vol_mean; /* Weighted mean of log-volatility
                                 * E[log_vol] = Σ w[i] * log_vol[i] */

        pf2d_real log_vol_variance; /* Weighted variance of log-volatility
                                     * Var[log_vol] = Σ w[i] * (log_vol[i] - mean)²
                                     * High variance = uncertain about vol regime */

        pf2d_real vol_mean; /* Expected volatility (not log)
                             * E[exp(log_vol)] = exp(μ + σ²/2) for log-normal
                             * Use this for Kelly sizing, not exp(log_vol_mean) */

        /*========================================================================
         * FILTER HEALTH METRICS
         *========================================================================*/

        pf2d_real ess; /* Effective Sample Size
                        * ESS = 1 / Σ w[i]²
                        * Range: [1, n_particles]
                        * ESS ≈ N: weights uniform (good)
                        * ESS << N: weights concentrated (resample needed)
                        * Rule of thumb: resample if ESS < N/2 */

        pf2d_real regime_probs[PF2D_MAX_REGIMES]; /* Posterior regime distribution
                                                   * regime_probs[r] = Σ w[i] for particles in regime r
                                                   * Sums to 1.0
                                                   * Useful for regime detection */

        int dominant_regime; /* Most likely regime: argmax(regime_probs)
                              * Quick indicator of market state */

        int resampled; /* Flag: did resampling occur this tick?
                        * 0 = no resampling (ESS was healthy)
                        * 1 = resampled (weights were reset to uniform)
                        * High resample rate may indicate model mismatch */

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

    /* Warmup - eliminates first-call latency from MKL/OpenMP */
    void pf2d_warmup(PF2D *pf);

    /* Disable denormals (FTZ+DAZ) for speed - called automatically by pf2d_warmup */
    void pf2d_disable_denormals_all_threads(void);

    /* Debug */
    void pf2d_print_config(const PF2D *pf);

#ifdef __cplusplus
}
#endif

#endif /* PARTICLE_FILTER_2D_H */