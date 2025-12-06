/**
 * @file rbpf_ksc.h
 * @brief RBPF with Kim-Shephard-Chib (1998) log-squared observation model
 *
 * Observation model: r_t = exp(ℓ_t) × ε_t,  ε_t ~ N(0,1)
 * Transform: y_t = log(r_t²) = 2ℓ_t + log(ε_t²)
 * 
 * log(ε_t²) is log-chi-squared(1), approximated as 7-component Gaussian mixture.
 * Linear observation: y = H×ℓ + noise, H = 2
 *
 * Latency target: <15μs for 1000 particles
 */

#ifndef RBPF_KSC_H
#define RBPF_KSC_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

#define RBPF_MAX_REGIMES 8
#define RBPF_ALIGN 64
#define RBPF_MAX_THREADS 32

/* KSC mixture components */
#define KSC_N_COMPONENTS 7

/*─────────────────────────────────────────────────────────────────────────────
 * PCG32 RNG (same as PF2D for consistency)
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    uint64_t state;
    uint64_t inc;
} rbpf_pcg32_t;

static inline void rbpf_pcg32_seed(rbpf_pcg32_t *rng, uint64_t seed, uint64_t seq) {
    rng->state = 0U;
    rng->inc = (seq << 1u) | 1u;
    rng->state += seed;
    rng->state = rng->state * 6364136223846793005ULL + rng->inc;
}

static inline uint32_t rbpf_pcg32_random(rbpf_pcg32_t *rng) {
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = (uint32_t)(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = (uint32_t)(oldstate >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

static inline float rbpf_pcg32_uniform(rbpf_pcg32_t *rng) {
    return (float)rbpf_pcg32_random(rng) / 4294967296.0f;
}

static inline float rbpf_pcg32_gaussian(rbpf_pcg32_t *rng) {
    float u1 = rbpf_pcg32_uniform(rng);
    float u2 = rbpf_pcg32_uniform(rng);
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(6.283185307179586f * u2);
}

/*─────────────────────────────────────────────────────────────────────────────
 * STRUCTURES
 *───────────────────────────────────────────────────────────────────────────*/

typedef struct {
    float theta;      /* Mean reversion speed */
    float mu_vol;     /* Long-run mean of log-vol */
    float sigma_vol;  /* Vol-of-vol (stored for regularization) */
    float q;          /* Process variance = sigma_vol² */
} RBPF_RegimeParams;

/**
 * Self-aware detection state (no external model needed)
 */
typedef struct {
    float vol_ema_short;     /* Fast EMA of vol_mean */
    float vol_ema_long;      /* Slow EMA of vol_mean */
    int prev_regime;         /* For structural change detection */
    int cooldown;            /* Ticks since last detection */
} RBPF_Detection;

/**
 * Main RBPF-KSC structure
 */
typedef struct {
    int n_particles;
    int n_regimes;
    
    /*========================================================================
     * PARTICLE STATE (SoA layout)
     *======================================================================*/
    float *mu;              /* [n] log-vol mean (Kalman state) */
    float *var;             /* [n] log-vol variance (Kalman covariance) */
    int *regime;            /* [n] regime index */
    float *log_weight;      /* [n] log-weights for numerical stability */
    
    /* Double buffers for resampling (pointer swap, no memcpy) */
    float *mu_tmp;
    float *var_tmp;
    int *regime_tmp;
    
    /*========================================================================
     * REGIME SYSTEM
     *======================================================================*/
    RBPF_RegimeParams params[RBPF_MAX_REGIMES];
    uint8_t trans_lut[RBPF_MAX_REGIMES][1024];  /* Precomputed transition LUT */
    
    /*========================================================================
     * WORKSPACE (preallocated - NO malloc in hot path)
     *======================================================================*/
    float *mu_pred;         /* [n] predicted mean */
    float *var_pred;        /* [n] predicted variance */
    float *theta_arr;       /* [n] gathered theta per particle */
    float *mu_vol_arr;      /* [n] gathered mu_vol per particle */
    float *q_arr;           /* [n] gathered q per particle */
    float *lik_total;       /* [n] total likelihood per particle */
    float *lik_comp;        /* [n] likelihood for current component */
    float *innov;           /* [n] innovation */
    float *S;               /* [n] innovation variance */
    float *K;               /* [n] Kalman gain */
    float *w_norm;          /* [n] normalized weights */
    float *cumsum;          /* [n] cumulative sum for resampling */
    float *mu_accum;        /* [n] accumulated mu across mixture */
    float *var_accum;       /* [n] accumulated var across mixture */
    float *scratch1;        /* [n] general workspace */
    float *scratch2;        /* [n] general workspace */
    int *indices;           /* [n] resampling indices */
    
    /*========================================================================
     * RNG
     *======================================================================*/
    rbpf_pcg32_t pcg[RBPF_MAX_THREADS];
    VSLStreamStatePtr mkl_rng[RBPF_MAX_THREADS];
    int n_threads;
    
    /*========================================================================
     * DETECTION STATE
     *======================================================================*/
    RBPF_Detection detection;
    
    /*========================================================================
     * REGULARIZATION
     *======================================================================*/
    float reg_bandwidth_mu;   /* Jitter bandwidth for mu after resample */
    float reg_bandwidth_var;  /* Jitter bandwidth for var after resample */
    float reg_scale_min;
    float reg_scale_max;
    float last_ess;
    
    /*========================================================================
     * PRECOMPUTED
     *======================================================================*/
    float uniform_weight;     /* 1/n */
    float inv_n;
    
} RBPF_KSC;

/**
 * Output from rbpf_ksc_step()
 */
typedef struct {
    /* State estimates */
    float vol_mean;           /* E[exp(ℓ)] */
    float log_vol_mean;       /* E[ℓ] */
    float log_vol_var;        /* Var[ℓ] (includes Kalman uncertainty) */
    float ess;                /* Effective sample size */
    
    /* Regime */
    float regime_probs[RBPF_MAX_REGIMES];
    int dominant_regime;
    
    /* Self-aware signals (Phase 1) */
    float marginal_lik;       /* p(y_t | y_{1:t-1}) - EXACT from Kalman */
    float surprise;           /* -log(marginal_lik) */
    float vol_ratio;          /* vol_mean / vol_ema */
    float regime_entropy;     /* -Σ p·log(p) */
    
    /* Detection flags */
    int regime_changed;       /* 0 or 1 */
    int change_type;          /* 0=none, 1=structural, 2=vol_shock, 3=surprise */
    
    /* Diagnostics */
    int resampled;
} RBPF_KSC_Output;

/*─────────────────────────────────────────────────────────────────────────────
 * API
 *───────────────────────────────────────────────────────────────────────────*/

/* Create/destroy */
RBPF_KSC* rbpf_ksc_create(int n_particles, int n_regimes);
void rbpf_ksc_destroy(RBPF_KSC *rbpf);

/* Configuration */
void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                 float theta, float mu_vol, float sigma_vol);
void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const float *trans_matrix);
void rbpf_ksc_set_regularization(RBPF_KSC *rbpf, float h_mu, float h_var);

/* Initialize */
void rbpf_ksc_init(RBPF_KSC *rbpf, float mu0, float var0);

/* Main update - THE HOT PATH */
void rbpf_ksc_step(RBPF_KSC *rbpf, float obs, RBPF_KSC_Output *output);

/* Warmup (call once before trading) */
void rbpf_ksc_warmup(RBPF_KSC *rbpf);

/* Debug */
void rbpf_ksc_print_config(const RBPF_KSC *rbpf);

#ifdef __cplusplus
}
#endif

#endif /* RBPF_KSC_H */
