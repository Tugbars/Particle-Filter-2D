/**
 * @file rbpf_ksc.c
 * @brief RBPF with Kim-Shephard-Chib (1998) - Optimized Implementation
 *
 * Key optimizations:
 *   - Zero malloc in hot path (all buffers preallocated)
 *   - Pointer swap instead of memcpy for resampling
 *   - PCG32 RNG (fast, good quality)
 *   - Transition LUT (no cumsum search)
 *   - Regularization after resample (prevents Kalman state degeneracy)
 *   - Self-aware detection signals (no external model)
 *
 * Latency target: <15μs for 1000 particles
 */

#include "rbpf_ksc.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>

/*─────────────────────────────────────────────────────────────────────────────
 * KIM-SHEPHARD-CHIB (1998) MIXTURE PARAMETERS
 *
 * Approximation of log(χ²(1)) as mixture of 7 Gaussians:
 * p(log(ε²)) ≈ Σ_k π_k × N(m_k, v_k²)
 *───────────────────────────────────────────────────────────────────────────*/

static const float KSC_PROB[KSC_N_COMPONENTS] = {
    0.00730f, 0.10556f, 0.00002f, 0.04395f, 0.34001f, 0.24566f, 0.25750f};

static const float KSC_MEAN[KSC_N_COMPONENTS] = {
    -10.12999f, -3.97281f, -8.56686f, 2.77786f, 0.61942f, 1.79518f, -1.08819f};

static const float KSC_VAR[KSC_N_COMPONENTS] = {
    5.79596f, 2.61369f, 5.17950f, 0.16735f, 0.64009f, 0.34023f, 1.26261f};

/* Precomputed: -0.5 * log(2π) = -0.9189385332 */
static const float LOG_2PI_HALF = -0.9189385332f;

/*─────────────────────────────────────────────────────────────────────────────
 * HELPERS
 *───────────────────────────────────────────────────────────────────────────*/

static inline float *aligned_alloc_float(int n)
{
    return (float *)mkl_malloc(n * sizeof(float), RBPF_ALIGN);
}

static inline int *aligned_alloc_int(int n)
{
    return (int *)mkl_malloc(n * sizeof(int), RBPF_ALIGN);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CREATE / DESTROY
 *───────────────────────────────────────────────────────────────────────────*/

RBPF_KSC *rbpf_ksc_create(int n_particles, int n_regimes)
{
    RBPF_KSC *rbpf = (RBPF_KSC *)mkl_calloc(1, sizeof(RBPF_KSC), RBPF_ALIGN);
    if (!rbpf)
        return NULL;

    rbpf->n_particles = n_particles;
    rbpf->n_regimes = n_regimes < RBPF_MAX_REGIMES ? n_regimes : RBPF_MAX_REGIMES;
    rbpf->uniform_weight = 1.0f / n_particles;
    rbpf->inv_n = 1.0f / n_particles;

    rbpf->n_threads = omp_get_max_threads();
    if (rbpf->n_threads > RBPF_MAX_THREADS)
        rbpf->n_threads = RBPF_MAX_THREADS;

    int n = n_particles;

    /* Particle state */
    rbpf->mu = aligned_alloc_float(n);
    rbpf->var = aligned_alloc_float(n);
    rbpf->regime = aligned_alloc_int(n);
    rbpf->log_weight = aligned_alloc_float(n);

    /* Double buffers */
    rbpf->mu_tmp = aligned_alloc_float(n);
    rbpf->var_tmp = aligned_alloc_float(n);
    rbpf->regime_tmp = aligned_alloc_int(n);

    /* Workspace - ALL preallocated */
    rbpf->mu_pred = aligned_alloc_float(n);
    rbpf->var_pred = aligned_alloc_float(n);
    rbpf->theta_arr = aligned_alloc_float(n);
    rbpf->mu_vol_arr = aligned_alloc_float(n);
    rbpf->q_arr = aligned_alloc_float(n);
    rbpf->lik_total = aligned_alloc_float(n);
    rbpf->lik_comp = aligned_alloc_float(n);
    rbpf->innov = aligned_alloc_float(n);
    rbpf->S = aligned_alloc_float(n);
    rbpf->K = aligned_alloc_float(n);
    rbpf->w_norm = aligned_alloc_float(n);
    rbpf->cumsum = aligned_alloc_float(n);
    rbpf->mu_accum = aligned_alloc_float(n);
    rbpf->var_accum = aligned_alloc_float(n);
    rbpf->scratch1 = aligned_alloc_float(n);
    rbpf->scratch2 = aligned_alloc_float(n);
    rbpf->indices = aligned_alloc_int(n);

    /* Check allocations */
    if (!rbpf->mu || !rbpf->var || !rbpf->regime || !rbpf->log_weight ||
        !rbpf->mu_tmp || !rbpf->var_tmp || !rbpf->regime_tmp ||
        !rbpf->mu_pred || !rbpf->var_pred || !rbpf->theta_arr ||
        !rbpf->mu_vol_arr || !rbpf->q_arr || !rbpf->lik_total ||
        !rbpf->lik_comp || !rbpf->innov || !rbpf->S || !rbpf->K ||
        !rbpf->w_norm || !rbpf->cumsum || !rbpf->mu_accum || !rbpf->var_accum ||
        !rbpf->scratch1 || !rbpf->scratch2 || !rbpf->indices)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }

    /* Initialize RNG */
    for (int t = 0; t < rbpf->n_threads; t++)
    {
        rbpf_pcg32_seed(&rbpf->pcg[t], 42 + t * 12345, t * 67890);
        vslNewStream(&rbpf->mkl_rng[t], VSL_BRNG_SFMT19937, 42 + t * 8192);
    }

    /* Default regime parameters */
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf->params[r].theta = 0.05f;
        rbpf->params[r].mu_vol = logf(0.01f); /* 1% daily vol */
        rbpf->params[r].sigma_vol = 0.1f;
        rbpf->params[r].q = 0.01f;
    }

    /* Regularization defaults */
    rbpf->reg_bandwidth_mu = 0.02f;   /* ~2% jitter on log-vol */
    rbpf->reg_bandwidth_var = 0.001f; /* Small jitter on variance */
    rbpf->reg_scale_min = 0.1f;
    rbpf->reg_scale_max = 0.5f;
    rbpf->last_ess = (float)n;

    /* Detection state */
    rbpf->detection.vol_ema_short = 0.01f;
    rbpf->detection.vol_ema_long = 0.01f;
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* MKL fast math mode */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    return rbpf;
}

void rbpf_ksc_destroy(RBPF_KSC *rbpf)
{
    if (!rbpf)
        return;

    for (int t = 0; t < rbpf->n_threads; t++)
    {
        if (rbpf->mkl_rng[t])
            vslDeleteStream(&rbpf->mkl_rng[t]);
    }

    mkl_free(rbpf->mu);
    mkl_free(rbpf->var);
    mkl_free(rbpf->regime);
    mkl_free(rbpf->log_weight);
    mkl_free(rbpf->mu_tmp);
    mkl_free(rbpf->var_tmp);
    mkl_free(rbpf->regime_tmp);
    mkl_free(rbpf->mu_pred);
    mkl_free(rbpf->var_pred);
    mkl_free(rbpf->theta_arr);
    mkl_free(rbpf->mu_vol_arr);
    mkl_free(rbpf->q_arr);
    mkl_free(rbpf->lik_total);
    mkl_free(rbpf->lik_comp);
    mkl_free(rbpf->innov);
    mkl_free(rbpf->S);
    mkl_free(rbpf->K);
    mkl_free(rbpf->w_norm);
    mkl_free(rbpf->cumsum);
    mkl_free(rbpf->mu_accum);
    mkl_free(rbpf->var_accum);
    mkl_free(rbpf->scratch1);
    mkl_free(rbpf->scratch2);
    mkl_free(rbpf->indices);
    mkl_free(rbpf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                float theta, float mu_vol, float sigma_vol)
{
    if (r < 0 || r >= RBPF_MAX_REGIMES)
        return;
    rbpf->params[r].theta = theta;
    rbpf->params[r].mu_vol = mu_vol;
    rbpf->params[r].sigma_vol = sigma_vol;
    rbpf->params[r].q = sigma_vol * sigma_vol;
}

void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const float *trans_matrix)
{
    /* Build LUT for each regime: uniform → next regime */
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        float cumsum[RBPF_MAX_REGIMES];
        cumsum[0] = trans_matrix[r * rbpf->n_regimes + 0];
        for (int j = 1; j < rbpf->n_regimes; j++)
        {
            cumsum[j] = cumsum[j - 1] + trans_matrix[r * rbpf->n_regimes + j];
        }

        for (int i = 0; i < 1024; i++)
        {
            float u = (float)i / 1024.0f;
            int next = rbpf->n_regimes - 1;
            for (int j = 0; j < rbpf->n_regimes - 1; j++)
            {
                if (u < cumsum[j])
                {
                    next = j;
                    break;
                }
            }
            rbpf->trans_lut[r][i] = (uint8_t)next;
        }
    }
}

void rbpf_ksc_set_regularization(RBPF_KSC *rbpf, float h_mu, float h_var)
{
    rbpf->reg_bandwidth_mu = h_mu;
    rbpf->reg_bandwidth_var = h_var;
}

/*─────────────────────────────────────────────────────────────────────────────
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_init(RBPF_KSC *rbpf, float mu0, float var0)
{
    int n = rbpf->n_particles;

    /* Spread particles slightly for diversity */
    for (int i = 0; i < n; i++)
    {
        float noise = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * 0.1f;
        rbpf->mu[i] = mu0 + noise;
        rbpf->var[i] = var0;
        rbpf->regime[i] = i % rbpf->n_regimes;
        rbpf->log_weight[i] = 0.0f; /* log(1) = 0 */
    }

    /* Reset detection */
    rbpf->detection.vol_ema_short = expf(mu0);
    rbpf->detection.vol_ema_long = expf(mu0);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;
}

/*─────────────────────────────────────────────────────────────────────────────
 * PREDICT STEP (optimized)
 *
 * ℓ_t = (1-θ)ℓ_{t-1} + θμ + η_t,  η_t ~ N(0, q)
 *
 * Kalman predict:
 *   μ_pred = (1-θ)μ + θμ_vol
 *   P_pred = (1-θ)²P + q
 *
 * Optimizations:
 *   - Unrolled regime gather (branch prediction friendly for stable regimes)
 *   - Fused arithmetic to reduce VML calls
 *───────────────────────────────────────────────────────────────────────────*/

static void rbpf_ksc_predict(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;
    const RBPF_RegimeParams *params = rbpf->params;
    const int n_regimes = rbpf->n_regimes;

    float *restrict mu = rbpf->mu;
    float *restrict var = rbpf->var;
    const int *restrict regime = rbpf->regime;
    float *restrict mu_pred = rbpf->mu_pred;
    float *restrict var_pred = rbpf->var_pred;

    /* Preload regime params into registers (max 8 regimes) */
    float theta_r[RBPF_MAX_REGIMES];
    float mu_vol_r[RBPF_MAX_REGIMES];
    float q_r[RBPF_MAX_REGIMES];
    float one_minus_theta_r[RBPF_MAX_REGIMES];
    float one_minus_theta_sq_r[RBPF_MAX_REGIMES];

    for (int r = 0; r < n_regimes; r++)
    {
        theta_r[r] = params[r].theta;
        mu_vol_r[r] = params[r].mu_vol;
        q_r[r] = params[r].q;
        one_minus_theta_r[r] = 1.0f - theta_r[r];
        one_minus_theta_sq_r[r] = one_minus_theta_r[r] * one_minus_theta_r[r];
    }

    /* Fused predict: single pass, no VML calls for small n
     * mu_pred = (1-θ)*μ + θ*μ_vol
     * var_pred = (1-θ)²*P + q
     */
    for (int i = 0; i < n; i++)
    {
        int r = regime[i];
        float omt = one_minus_theta_r[r];
        float omt2 = one_minus_theta_sq_r[r];
        float th = theta_r[r];
        float mv = mu_vol_r[r];
        float q = q_r[r];

        mu_pred[i] = omt * mu[i] + th * mv;
        var_pred[i] = omt2 * var[i] + q;
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * UPDATE STEP (optimized 7-component mixture Kalman)
 *
 * Observation: y = log(r²) = 2ℓ + log(ε²)
 * Linear: y - m_k = H*ℓ + (log(ε²) - m_k), H = 2
 *
 * Optimizations:
 *   - Fused scalar loops for small n (avoids VML dispatch overhead)
 *   - Precomputed constants (H², log(π_k), etc.)
 *   - Single pass accumulation
 *───────────────────────────────────────────────────────────────────────────*/

/* Precomputed: log(π_k) for each component */
static const float KSC_LOG_PROB[KSC_N_COMPONENTS] = {
    -4.920f,  /* log(0.00730) */
    -2.248f,  /* log(0.10556) */
    -10.820f, /* log(0.00002) */
    -3.125f,  /* log(0.04395) */
    -1.079f,  /* log(0.34001) */
    -1.404f,  /* log(0.24566) */
    -1.356f   /* log(0.25750) */
};

static float rbpf_ksc_update(RBPF_KSC *rbpf, float y)
{
    const int n = rbpf->n_particles;
    const float H = 2.0f;
    const float H2 = 4.0f;
    const float NEG_HALF = -0.5f;

    float *restrict mu_pred = rbpf->mu_pred;
    float *restrict var_pred = rbpf->var_pred;
    float *restrict mu = rbpf->mu;
    float *restrict var = rbpf->var;
    float *restrict log_weight = rbpf->log_weight;
    float *restrict lik_total = rbpf->lik_total;
    float *restrict mu_accum = rbpf->mu_accum;
    float *restrict var_accum = rbpf->var_accum;

    /* Zero accumulators */
    memset(lik_total, 0, n * sizeof(float));
    memset(mu_accum, 0, n * sizeof(float));
    memset(var_accum, 0, n * sizeof(float));

    /* Process each mixture component - fused scalar loop */
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        const float m_k = KSC_MEAN[k];
        const float v2_k = KSC_VAR[k];
        const float log_pi_k = KSC_LOG_PROB[k];
        const float y_adj = y - m_k;

        /* Fused loop: compute everything per particle */
        for (int i = 0; i < n; i++)
        {
            /* Innovation */
            float innov = y_adj - H * mu_pred[i];

            /* Innovation variance */
            float S = H2 * var_pred[i] + v2_k;

            /* Kalman gain */
            float K = H * var_pred[i] / S;

            /* Log-likelihood: -0.5*(log(S) + innov²/S) + log(π_k) */
            float innov2_S = innov * innov / S;
            float log_lik = NEG_HALF * (logf(S) + innov2_S) + log_pi_k;
            float lik = expf(log_lik);

            /* Accumulate */
            lik_total[i] += lik;

            /* Updated state for this component */
            float mu_k = mu_pred[i] + K * innov;
            float var_k = (1.0f - K * H) * var_pred[i];

            mu_accum[i] += lik * mu_k;
            var_accum[i] += lik * var_k;
        }
    }

    /* Normalize and update weights */
    float total_marginal = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float inv_lik = 1.0f / (lik_total[i] + 1e-30f);
        mu[i] = mu_accum[i] * inv_lik;
        var[i] = var_accum[i] * inv_lik;

        /* Floor variance */
        if (var[i] < 1e-6f)
            var[i] = 1e-6f;

        /* Update log-weight */
        log_weight[i] += logf(lik_total[i] + 1e-30f);

        total_marginal += lik_total[i];
    }

    return total_marginal / n;
}

/*─────────────────────────────────────────────────────────────────────────────
 * REGIME TRANSITION (LUT-based, no cumsum search)
 *───────────────────────────────────────────────────────────────────────────*/

static void rbpf_ksc_transition(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;
    int *regime = rbpf->regime;
    rbpf_pcg32_t *rng = &rbpf->pcg[0];

    for (int i = 0; i < n; i++)
    {
        int r_old = regime[i];
        float u = rbpf_pcg32_uniform(rng);
        int lut_idx = (int)(u * 1023.0f);
        regime[i] = rbpf->trans_lut[r_old][lut_idx];
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * RESAMPLE (systematic + regularization)
 *───────────────────────────────────────────────────────────────────────────*/

static int rbpf_ksc_resample(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;

    float *log_weight = rbpf->log_weight;
    float *w_norm = rbpf->w_norm;
    float *cumsum = rbpf->cumsum;
    int *indices = rbpf->indices;

    /* Find max log-weight for numerical stability */
    float max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    /* Normalize: w = exp(lw - max) / sum */
    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    vsExp(n, w_norm, w_norm);

    float sum_w = cblas_sasum(n, w_norm, 1);
    if (sum_w < 1e-30f)
    {
        /* All weights collapsed - reset to uniform */
        float uw = rbpf->uniform_weight;
        for (int i = 0; i < n; i++)
        {
            w_norm[i] = uw;
        }
        sum_w = 1.0f;
    }
    cblas_sscal(n, 1.0f / sum_w, w_norm, 1);

    /* Compute ESS */
    float sum_w2 = cblas_sdot(n, w_norm, 1, w_norm, 1);
    float ess = 1.0f / sum_w2;
    rbpf->last_ess = ess;

    /* Only resample if ESS < n/2 */
    if (ess > n * 0.5f)
    {
        return 0;
    }

    /* Cumulative sum */
    cumsum[0] = w_norm[0];
    for (int i = 1; i < n; i++)
    {
        cumsum[i] = cumsum[i - 1] + w_norm[i];
    }

    /* Systematic resampling */
    float u0 = rbpf_pcg32_uniform(&rbpf->pcg[0]) * rbpf->inv_n;
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        float u = u0 + (float)i * rbpf->inv_n;
        while (j < n - 1 && cumsum[j] < u)
            j++;
        indices[i] = j;
    }

    /* Gather into tmp buffers */
    float *mu = rbpf->mu;
    float *var = rbpf->var;
    int *regime = rbpf->regime;
    float *mu_tmp = rbpf->mu_tmp;
    float *var_tmp = rbpf->var_tmp;
    int *regime_tmp = rbpf->regime_tmp;

    for (int i = 0; i < n; i++)
    {
        int idx = indices[i];
        mu_tmp[i] = mu[idx];
        var_tmp[i] = var[idx];
        regime_tmp[i] = regime[idx];
    }

    /* Pointer swap (no memcpy!) */
    rbpf->mu = mu_tmp;
    rbpf->mu_tmp = mu;
    rbpf->var = var_tmp;
    rbpf->var_tmp = var;
    rbpf->regime = regime_tmp;
    rbpf->regime_tmp = regime;

    /* Reset log-weights to 0 */
    memset(rbpf->log_weight, 0, n * sizeof(float));

    /* Apply regularization (kernel jitter) */
    float ess_ratio = ess / (float)n;
    float scale = rbpf->reg_scale_max -
                  (rbpf->reg_scale_max - rbpf->reg_scale_min) * ess_ratio;
    if (scale < rbpf->reg_scale_min)
        scale = rbpf->reg_scale_min;
    if (scale > rbpf->reg_scale_max)
        scale = rbpf->reg_scale_max;

    float h_mu = rbpf->reg_bandwidth_mu * scale;
    float h_var = rbpf->reg_bandwidth_var * scale;

    /* Add jitter to break duplicates */
    mu = rbpf->mu;
    var = rbpf->var;
    rbpf_pcg32_t *rng = &rbpf->pcg[0];

    for (int i = 0; i < n; i++)
    {
        mu[i] += h_mu * rbpf_pcg32_gaussian(rng);
        var[i] += h_var * fabsf(rbpf_pcg32_gaussian(rng));
        if (var[i] < 1e-6f)
            var[i] = 1e-6f;
    }

    return 1;
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE OUTPUTS
 *───────────────────────────────────────────────────────────────────────────*/

static void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, float marginal_lik,
                                     RBPF_KSC_Output *out)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;

    float *log_weight = rbpf->log_weight;
    float *mu = rbpf->mu;
    float *var = rbpf->var;
    int *regime = rbpf->regime;
    float *w_norm = rbpf->w_norm;

    /* Normalize weights */
    float max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    vsExp(n, w_norm, w_norm);

    float sum_w = cblas_sasum(n, w_norm, 1);
    if (sum_w < 1e-30f)
        sum_w = 1.0f;
    cblas_sscal(n, 1.0f / sum_w, w_norm, 1);

    /* Log-vol mean and variance (using law of total variance) */
    float log_vol_mean = 0.0f;
    for (int i = 0; i < n; i++)
    {
        log_vol_mean += w_norm[i] * mu[i];
    }

    float log_vol_var = 0.0f;
    for (int i = 0; i < n; i++)
    {
        float diff = mu[i] - log_vol_mean;
        /* Var[X] = E[Var[X|particle]] + Var[E[X|particle]] */
        log_vol_var += w_norm[i] * (var[i] + diff * diff);
    }

    out->log_vol_mean = log_vol_mean;
    out->log_vol_var = log_vol_var;

    /* Vol mean: E[exp(ℓ)] ≈ exp(E[ℓ] + 0.5*Var[ℓ]) */
    out->vol_mean = expf(log_vol_mean + 0.5f * log_vol_var);

    /* ESS */
    float sum_w2 = cblas_sdot(n, w_norm, 1, w_norm, 1);
    out->ess = 1.0f / sum_w2;

    /* Regime probabilities */
    memset(out->regime_probs, 0, sizeof(out->regime_probs));
    for (int i = 0; i < n; i++)
    {
        out->regime_probs[regime[i]] += w_norm[i];
    }

    /* Dominant regime */
    int dom = 0;
    float max_prob = out->regime_probs[0];
    for (int r = 1; r < n_regimes; r++)
    {
        if (out->regime_probs[r] > max_prob)
        {
            max_prob = out->regime_probs[r];
            dom = r;
        }
    }
    out->dominant_regime = dom;

    /* Self-aware signals */
    out->marginal_lik = marginal_lik;
    out->surprise = -logf(marginal_lik + 1e-30f);

    /* Regime entropy: -Σ p*log(p) */
    float entropy = 0.0f;
    for (int r = 0; r < n_regimes; r++)
    {
        float p = out->regime_probs[r];
        if (p > 1e-10f)
        {
            entropy -= p * logf(p);
        }
    }
    out->regime_entropy = entropy;

    /* Vol ratio (vs EMA) */
    RBPF_Detection *det = &rbpf->detection;
    det->vol_ema_short = 0.1f * out->vol_mean + 0.9f * det->vol_ema_short;
    det->vol_ema_long = 0.01f * out->vol_mean + 0.99f * det->vol_ema_long;
    out->vol_ratio = det->vol_ema_short / (det->vol_ema_long + 1e-10f);

    /* Regime change detection */
    out->regime_changed = 0;
    out->change_type = 0;

    if (det->cooldown > 0)
    {
        det->cooldown--;
    }
    else
    {
        /* Structural: regime flipped with high confidence */
        int structural = (dom != det->prev_regime) && (max_prob > 0.7f);

        /* Vol shock: >80% increase or >50% decrease */
        int vol_shock = (out->vol_ratio > 1.8f) || (out->vol_ratio < 0.5f);

        /* Surprise: observation unlikely under model */
        int surprised = (out->surprise > 5.0f);

        if (structural || vol_shock || surprised)
        {
            out->regime_changed = 1;
            out->change_type = structural ? 1 : (vol_shock ? 2 : 3);
            det->cooldown = 20; /* Suppress for 20 ticks */
        }
    }

    det->prev_regime = dom;
}

/*─────────────────────────────────────────────────────────────────────────────
 * MAIN UPDATE - THE HOT PATH
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_step(RBPF_KSC *rbpf, float obs, RBPF_KSC_Output *output)
{
    /* Transform observation: y = log(r²) */
    float y;
    if (fabsf(obs) < 1e-10f)
    {
        y = -23.0f; /* Floor at ~log(1e-10²) */
    }
    else
    {
        y = logf(obs * obs);
    }

    /* 1. Regime transition */
    rbpf_ksc_transition(rbpf);

    /* 2. Kalman predict */
    rbpf_ksc_predict(rbpf);

    /* 3. Mixture Kalman update */
    float marginal_lik = rbpf_ksc_update(rbpf, y);

    /* 4. Compute outputs (before resample) */
    rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);

    /* 5. Resample if needed */
    output->resampled = rbpf_ksc_resample(rbpf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * WARMUP
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_warmup(RBPF_KSC *rbpf)
{
    int n = rbpf->n_particles;

/* Force OpenMP thread creation */
#pragma omp parallel
    {
        volatile int tid = omp_get_thread_num();
        (void)tid;
    }

    /* Warmup MKL VML */
    vsExp(n, rbpf->mu, rbpf->scratch1);
    vsLn(n, rbpf->var, rbpf->scratch2);

    /* Warmup BLAS */
    volatile float sum = cblas_sasum(n, rbpf->w_norm, 1);
    (void)sum;
}

/*─────────────────────────────────────────────────────────────────────────────
 * DEBUG
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_print_config(const RBPF_KSC *rbpf)
{
    printf("RBPF-KSC Configuration:\n");
    printf("  Particles:     %d\n", rbpf->n_particles);
    printf("  Regimes:       %d\n", rbpf->n_regimes);
    printf("  Threads:       %d\n", rbpf->n_threads);
    printf("  Reg bandwidth: mu=%.4f, var=%.4f\n",
           rbpf->reg_bandwidth_mu, rbpf->reg_bandwidth_var);

    printf("\n  Per-regime parameters:\n");
    printf("  %-8s %8s %8s %8s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        const RBPF_RegimeParams *p = &rbpf->params[r];
        printf("  %-8d %8.4f %8.4f %8.4f\n",
               r, p->theta, p->mu_vol, p->sigma_vol);
    }
}