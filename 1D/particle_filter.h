/* particle_filter.h
 * High-performance particle filter with Intel MKL
 * Part of quantitative trading stack: SSA → BOCPD → PF → Kelly
 */
#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <stdint.h>
#include <math.h>

#ifdef __cplusplus
extern "C"
{
#endif

/* ========================================================================== */
/* PRECISION CONFIGURATION                                                    */
/* ========================================================================== */

/* Define PF_USE_FLOAT before including to use single precision */
#ifdef PF_USE_FLOAT
    typedef float pf_real;
#define PF_REAL_SIZE 4
#define pf_vExp vsExp
#define pf_vSqr vsSqr
#define pf_cblas_dot cblas_sdot
#define pf_cblas_scal cblas_sscal
#define pf_cblas_asum cblas_sasum
#define pf_cblas_iamax cblas_isamax
#define pf_RngGaussian vsRngGaussian
#define pf_RngUniform vsRngUniform
#else
typedef double pf_real;
#define PF_REAL_SIZE 8
#define pf_vExp vdExp
#define pf_vSqr vdSqr
#define pf_cblas_dot cblas_ddot
#define pf_cblas_scal cblas_dscal
#define pf_cblas_asum cblas_dasum
#define pf_cblas_iamax cblas_idamax
#define pf_RngGaussian vdRngGaussian
#define pf_RngUniform vdRngUniform
#endif

    /* ========================================================================== */
    /* CONFIGURATION                                                              */
    /* ========================================================================== */

#define PF_MAX_REGIMES 8
#define PF_ALIGN 64

    /* ========================================================================== */
    /* STRUCTURES                                                                 */
    /* ========================================================================== */

    typedef struct
    {
        pf_real eigentriples[8];
        pf_real trend;
        pf_real volatility;
    } SSAFeatures;

    typedef struct
    {
        pf_real probs[PF_MAX_REGIMES];
        pf_real cumprobs[PF_MAX_REGIMES]; /* Precomputed for branchless sampling */
        int n_regimes;
    } RegimeProbs;

    /* Precomputed SSA-derived terms (updated when SSA refreshes) */
    typedef struct
    {
        pf_real drift_scaled[PF_MAX_REGIMES];
        pf_real sigma_scaled[PF_MAX_REGIMES];
        pf_real one_minus_theta[PF_MAX_REGIMES];
        pf_real theta_mean[PF_MAX_REGIMES];
    } PFPrecomputed;

/* Regime lookup table size (power of 2 for fast indexing) */
#define PF_REGIME_LUT_SIZE 1024
#define PF_REGIME_LUT_MASK (PF_REGIME_LUT_SIZE - 1)

/* Max threads for thread-local RNG */
#define PF_MAX_THREADS 128

/* BLAS threshold: use manual loops below this N (BLAS call overhead) */
#define PF_BLAS_THRESHOLD 4000

/* Adaptive resampling bounds */
#define PF_RESAMPLE_THRESH_MIN 0.3 /* Don't resample too often in high-vol */
#define PF_RESAMPLE_THRESH_MAX 0.7 /* Don't wait too long in low-vol */
#define PF_RESAMPLE_THRESH_DEFAULT 0.5

    /*============================================================================
     * PCG32 RNG (Permuted Congruential Generator)
     *============================================================================
     * Fast, statistically excellent, minimal state (16 bytes per stream).
     * Better than MKL VSL for embarrassingly parallel workloads.
     *============================================================================*/

    typedef struct
    {
        uint64_t state;
        uint64_t inc;
    } pcg32_random_t;

    static inline void pcg32_seed(pcg32_random_t *rng, uint64_t seed, uint64_t seq)
    {
        rng->state = 0U;
        rng->inc = (seq << 1u) | 1u;
        rng->state += seed;
        /* Advance once */
        rng->state = rng->state * 6364136223846793005ULL + rng->inc;
    }

    static inline uint32_t pcg32_random(pcg32_random_t *rng)
    {
        uint64_t oldstate = rng->state;
        rng->state = oldstate * 6364136223846793005ULL + rng->inc;
        uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
        uint32_t rot = (uint32_t)(oldstate >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    }

    /* Uniform [0, 1) */
    static inline pf_real pcg32_uniform(pcg32_random_t *rng)
    {
        return (pf_real)pcg32_random(rng) / (pf_real)4294967296.0;
    }

    /* Box-Muller for Gaussian (generates pairs, we discard one for simplicity) */
    static inline pf_real pcg32_gaussian(pcg32_random_t *rng)
    {
        pf_real u1 = pcg32_uniform(rng);
        pf_real u2 = pcg32_uniform(rng);
        /* Avoid log(0) */
        if (u1 < (pf_real)1e-10)
            u1 = (pf_real)1e-10;
        return (pf_real)sqrt(-2.0 * log((double)u1)) * (pf_real)cos(6.283185307179586 * (double)u2);
    }

    typedef struct
    {
        /* Particle state - SoA layout, MKL aligned */
        pf_real *states;
        pf_real *states_tmp;
        pf_real *weights;
        pf_real *log_weights;
        pf_real *cumsum;
        pf_real *noise;
        pf_real *uniform;
        pf_real *scratch;
        int *regimes;
        int *regimes_tmp;

        /* MKL RNG streams - for bulk generation fallback */
        VSLStreamStatePtr rng[PF_MAX_THREADS];

        /* PCG RNG streams - fast, per-thread, for parallel generation */
        pcg32_random_t pcg[PF_MAX_THREADS];
        int use_pcg; /* 1 = use PCG, 0 = use MKL */
        int n_threads;

        /* Regime lookup table for O(1) sampling */
        uint8_t regime_lut[PF_REGIME_LUT_SIZE];

        /* Base regime parameters */
        pf_real drift[PF_MAX_REGIMES];
        pf_real mean[PF_MAX_REGIMES];
        pf_real theta[PF_MAX_REGIMES];
        pf_real sigma[PF_MAX_REGIMES];

        /* Precomputed terms */
        PFPrecomputed pre;

        /* Observation model */
        pf_real obs_variance;
        pf_real inv_obs_variance;
        pf_real neg_half_inv_var;

        /* Adaptive resampling */
        pf_real resample_threshold;  /* Current threshold (adaptive) */
        pf_real volatility_ema;      /* EMA of particle variance for adaptation */
        pf_real volatility_baseline; /* Baseline for comparison */

        /* Dimensions */
        int n_particles;
        int n_regimes;

        /* Precomputed uniform weight */
        pf_real uniform_weight;

    } ParticleFilter;

    typedef struct
    {
        pf_real mean;
        pf_real variance;
        pf_real ess;
        pf_real regime_probs[PF_MAX_REGIMES];
        int resampled;
    } PFOutput;

    /* ========================================================================== */
    /* API                                                                        */
    /* ========================================================================== */

    /* Create/destroy */
    ParticleFilter *pf_create(int n_particles, int n_regimes);
    void pf_destroy(ParticleFilter *pf);

    /* Configuration */
    void pf_set_observation_variance(ParticleFilter *pf, pf_real var);
    void pf_set_regime_params(ParticleFilter *pf, int regime,
                              pf_real drift, pf_real mean, pf_real theta, pf_real sigma);

    /* Precompute SSA-derived terms (call when SSA refreshes every 50-100 ticks) */
    void pf_precompute(ParticleFilter *pf, const SSAFeatures *ssa);

    /* Set regime probs with precomputed cumulative */
    void pf_set_regime_probs(RegimeProbs *rp, const pf_real *probs, int n);

    /* Build regime lookup table for O(1) sampling - call after pf_set_regime_probs */
    void pf_build_regime_lut(ParticleFilter *pf, const RegimeProbs *rp);

    /* Enable PCG RNG (faster than MKL for parallel generation) */
    void pf_enable_pcg(ParticleFilter *pf, int enable);

    /* Set adaptive resampling parameters */
    void pf_set_resample_adaptive(ParticleFilter *pf, pf_real baseline_volatility);

    /* Initialize particles */
    void pf_initialize(ParticleFilter *pf, pf_real x0, pf_real spread);

    /* Core operations */
    void pf_propagate(ParticleFilter *pf, const RegimeProbs *regime_probs);
    void pf_update_weights(ParticleFilter *pf, pf_real observation);
    pf_real pf_effective_sample_size(const ParticleFilter *pf);
    void pf_resample(ParticleFilter *pf);
    int pf_resample_if_needed(ParticleFilter *pf, pf_real threshold);

    /* Estimates */
    pf_real pf_mean(const ParticleFilter *pf);
    pf_real pf_variance(const ParticleFilter *pf);
    void pf_regime_distribution(const ParticleFilter *pf, pf_real *out);

    /* Full update step */
    PFOutput pf_update(ParticleFilter *pf, pf_real observation,
                       const RegimeProbs *regime_probs);

    /* Debug */
    void pf_print_config(const ParticleFilter *pf);

#ifdef __cplusplus
}
#endif

#endif /* PARTICLE_FILTER_H */
