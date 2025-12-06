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
#include <mkl_vml.h>

/*─────────────────────────────────────────────────────────────────────────────
 * KIM-SHEPHARD-CHIB (1998) MIXTURE PARAMETERS
 *
 * Approximation of log(χ²(1)) as mixture of 7 Gaussians:
 * p(log(ε²)) ≈ Σ_k π_k × N(m_k, v_k²)
 *───────────────────────────────────────────────────────────────────────────*/

static const rbpf_real_t KSC_PROB[KSC_N_COMPONENTS] = {
    RBPF_REAL(0.00730), RBPF_REAL(0.10556), RBPF_REAL(0.00002), RBPF_REAL(0.04395),
    RBPF_REAL(0.34001), RBPF_REAL(0.24566), RBPF_REAL(0.25750)};

static const rbpf_real_t KSC_MEAN[KSC_N_COMPONENTS] = {
    RBPF_REAL(-10.12999), RBPF_REAL(-3.97281), RBPF_REAL(-8.56686), RBPF_REAL(2.77786),
    RBPF_REAL(0.61942), RBPF_REAL(1.79518), RBPF_REAL(-1.08819)};

static const rbpf_real_t KSC_VAR[KSC_N_COMPONENTS] = {
    RBPF_REAL(5.79596), RBPF_REAL(2.61369), RBPF_REAL(5.17950), RBPF_REAL(0.16735),
    RBPF_REAL(0.64009), RBPF_REAL(0.34023), RBPF_REAL(1.26261)};

/* Precomputed: -0.5 * log(2π) = -0.9189385332 */
static const rbpf_real_t LOG_2PI_HALF = RBPF_REAL(-0.9189385332);

/*─────────────────────────────────────────────────────────────────────────────
 * HELPERS
 *───────────────────────────────────────────────────────────────────────────*/

static inline rbpf_real_t *aligned_alloc_real(int n)
{
    return (rbpf_real_t *)mkl_malloc(n * sizeof(rbpf_real_t), RBPF_ALIGN);
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
    rbpf->uniform_weight = RBPF_REAL(1.0) / n_particles;
    rbpf->inv_n = RBPF_REAL(1.0) / n_particles;

    rbpf->n_threads = omp_get_max_threads();
    if (rbpf->n_threads > RBPF_MAX_THREADS)
        rbpf->n_threads = RBPF_MAX_THREADS;

    int n = n_particles;

    /* Particle state */
    rbpf->mu = aligned_alloc_real(n);
    rbpf->var = aligned_alloc_real(n);
    rbpf->regime = aligned_alloc_int(n);
    rbpf->log_weight = aligned_alloc_real(n);

    /* Double buffers */
    rbpf->mu_tmp = aligned_alloc_real(n);
    rbpf->var_tmp = aligned_alloc_real(n);
    rbpf->regime_tmp = aligned_alloc_int(n);

    /* Workspace - ALL preallocated */
    rbpf->mu_pred = aligned_alloc_real(n);
    rbpf->var_pred = aligned_alloc_real(n);
    rbpf->theta_arr = aligned_alloc_real(n);
    rbpf->mu_vol_arr = aligned_alloc_real(n);
    rbpf->q_arr = aligned_alloc_real(n);
    rbpf->lik_total = aligned_alloc_real(n);
    rbpf->lik_comp = aligned_alloc_real(n);
    rbpf->innov = aligned_alloc_real(n);
    rbpf->S = aligned_alloc_real(n);
    rbpf->K = aligned_alloc_real(n);
    rbpf->w_norm = aligned_alloc_real(n);
    rbpf->cumsum = aligned_alloc_real(n);
    rbpf->mu_accum = aligned_alloc_real(n);
    rbpf->var_accum = aligned_alloc_real(n);
    rbpf->scratch1 = aligned_alloc_real(n);
    rbpf->scratch2 = aligned_alloc_real(n);
    rbpf->indices = aligned_alloc_int(n);

    /* Log-sum-exp buffers for numerical stability in K-mixture */
    rbpf->log_lik_buffer = aligned_alloc_real(KSC_N_COMPONENTS * n);
    rbpf->max_log_lik = aligned_alloc_real(n);

    /* Pre-generated Gaussian buffer for jitter (MKL ICDF) */
    rbpf->rng_gaussian = aligned_alloc_real(2 * n); /* 2n for mu and var jitter */
    rbpf->rng_buffer_size = 2 * n;

    /* Check allocations */
    if (!rbpf->mu || !rbpf->var || !rbpf->regime || !rbpf->log_weight ||
        !rbpf->mu_tmp || !rbpf->var_tmp || !rbpf->regime_tmp ||
        !rbpf->mu_pred || !rbpf->var_pred || !rbpf->theta_arr ||
        !rbpf->mu_vol_arr || !rbpf->q_arr || !rbpf->lik_total ||
        !rbpf->lik_comp || !rbpf->innov || !rbpf->S || !rbpf->K ||
        !rbpf->w_norm || !rbpf->cumsum || !rbpf->mu_accum || !rbpf->var_accum ||
        !rbpf->scratch1 || !rbpf->scratch2 || !rbpf->indices ||
        !rbpf->log_lik_buffer || !rbpf->max_log_lik || !rbpf->rng_gaussian)
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
        rbpf->params[r].theta = RBPF_REAL(0.05);
        rbpf->params[r].mu_vol = rbpf_log(RBPF_REAL(0.01)); /* 1% daily vol */
        rbpf->params[r].sigma_vol = RBPF_REAL(0.1);
        rbpf->params[r].q = RBPF_REAL(0.01);
    }

    /* Regularization defaults
     *
     * With correct law-of-total-variance calculation, variance estimates
     * are more accurate. We can use lighter regularization:
     * - h_mu: jitter on state to prevent particle collapse
     * - h_var: lighter jitter on covariance (optional, mainly for robustness)
     */
    rbpf->reg_bandwidth_mu = RBPF_REAL(0.02);    /* ~2% jitter on log-vol */
    rbpf->reg_bandwidth_var = RBPF_REAL(0.0005); /* Reduced: correct variance calc */
    rbpf->reg_scale_min = RBPF_REAL(0.1);
    rbpf->reg_scale_max = RBPF_REAL(0.5);
    rbpf->last_ess = (rbpf_real_t)n;

    /* Regime diversity: prevent particle collapse to single regime
     * Without this, resampling can kill minority regimes, leaving
     * no particles to respond to sudden regime changes. */
    rbpf->min_particles_per_regime = n / (4 * n_regimes); /* ~6% per regime */
    if (rbpf->min_particles_per_regime < 2)
        rbpf->min_particles_per_regime = 2;
    rbpf->regime_mutation_prob = RBPF_REAL(0.02); /* 2% mutation rate */

    /* Detection state */
    rbpf->detection.vol_ema_short = RBPF_REAL(0.01);
    rbpf->detection.vol_ema_long = RBPF_REAL(0.01);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* Liu-West parameter learning (Phase 3) */
    int n_params = n * n_regimes;
    rbpf->particle_mu_vol = aligned_alloc_real(n_params);
    rbpf->particle_sigma_vol = aligned_alloc_real(n_params);
    rbpf->particle_mu_vol_tmp = aligned_alloc_real(n_params);
    rbpf->particle_sigma_vol_tmp = aligned_alloc_real(n_params);

    if (!rbpf->particle_mu_vol || !rbpf->particle_sigma_vol ||
        !rbpf->particle_mu_vol_tmp || !rbpf->particle_sigma_vol_tmp)
    {
        rbpf_ksc_destroy(rbpf);
        return NULL;
    }

    /* Liu-West defaults (disabled) */
    rbpf->liu_west.enabled = 0;
    rbpf->liu_west.shrinkage = RBPF_REAL(0.98);
    rbpf->liu_west.min_mu_vol = rbpf_log(0.001f); /* 0.1% vol floor */
    rbpf->liu_west.max_mu_vol = rbpf_log(0.5f);   /* 50% vol ceiling */
    rbpf->liu_west.min_sigma_vol = RBPF_REAL(0.01);
    rbpf->liu_west.max_sigma_vol = RBPF_REAL(1.0);
    rbpf->liu_west.learn_mu_vol = 1;
    rbpf->liu_west.learn_sigma_vol = 0; /* Off by default */
    rbpf->liu_west.warmup_ticks = 100;
    rbpf->liu_west.tick_count = 0;

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

    /* Log-sum-exp and RNG buffers */
    mkl_free(rbpf->log_lik_buffer);
    mkl_free(rbpf->max_log_lik);
    mkl_free(rbpf->rng_gaussian);

    /* Liu-West arrays */
    mkl_free(rbpf->particle_mu_vol);
    mkl_free(rbpf->particle_sigma_vol);
    mkl_free(rbpf->particle_mu_vol_tmp);
    mkl_free(rbpf->particle_sigma_vol_tmp);

    mkl_free(rbpf);
}

/*─────────────────────────────────────────────────────────────────────────────
 * CONFIGURATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_set_regime_params(RBPF_KSC *rbpf, int r,
                                rbpf_real_t theta, rbpf_real_t mu_vol, rbpf_real_t sigma_vol)
{
    if (r < 0 || r >= RBPF_MAX_REGIMES)
        return;
    rbpf->params[r].theta = theta;
    rbpf->params[r].mu_vol = mu_vol;
    rbpf->params[r].sigma_vol = sigma_vol;
    rbpf->params[r].q = sigma_vol * sigma_vol;
}

void rbpf_ksc_build_transition_lut(RBPF_KSC *rbpf, const rbpf_real_t *trans_matrix)
{
    /* Build LUT for each regime: uniform → next regime */
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        rbpf_real_t cumsum[RBPF_MAX_REGIMES];
        cumsum[0] = trans_matrix[r * rbpf->n_regimes + 0];
        for (int j = 1; j < rbpf->n_regimes; j++)
        {
            cumsum[j] = cumsum[j - 1] + trans_matrix[r * rbpf->n_regimes + j];
        }

        for (int i = 0; i < 1024; i++)
        {
            rbpf_real_t u = (rbpf_real_t)i / RBPF_REAL(1024.0);
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

void rbpf_ksc_set_regularization(RBPF_KSC *rbpf, rbpf_real_t h_mu, rbpf_real_t h_var)
{
    rbpf->reg_bandwidth_mu = h_mu;
    rbpf->reg_bandwidth_var = h_var;
}

void rbpf_ksc_set_regime_diversity(RBPF_KSC *rbpf, int min_per_regime, rbpf_real_t mutation_prob)
{
    rbpf->min_particles_per_regime = min_per_regime;
    /* Clamp mutation probability to [0, 0.2] for stability */
    if (mutation_prob < RBPF_REAL(0.0))
        mutation_prob = RBPF_REAL(0.0);
    if (mutation_prob > RBPF_REAL(0.2))
        mutation_prob = RBPF_REAL(0.2);
    rbpf->regime_mutation_prob = mutation_prob;
}

/*─────────────────────────────────────────────────────────────────────────────
 * LIU-WEST PARAMETER LEARNING (Phase 3)
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_enable_liu_west(RBPF_KSC *rbpf, rbpf_real_t shrinkage, int warmup_ticks)
{
    rbpf->liu_west.enabled = 1;
    rbpf->liu_west.shrinkage = shrinkage;
    rbpf->liu_west.warmup_ticks = warmup_ticks;
    rbpf->liu_west.tick_count = 0;
    rbpf->liu_west.learn_mu_vol = 1;
    rbpf->liu_west.learn_sigma_vol = 0; /* Conservative default */

    /* Aggressive resampling for learning mode
     *
     * CRITICAL: Liu-West only updates during resample!
     * With normal ESS threshold (0.5), we rarely resample when filter is confident.
     * This kills learning. Solution: higher threshold + forced periodic resample.
     */
    rbpf->liu_west.resample_threshold = RBPF_REAL(0.85); /* Resample when ESS < 85% */
    rbpf->liu_west.max_ticks_no_resample = 5;            /* Force resample every 5 ticks (was 10) */
    rbpf->liu_west.ticks_since_resample = 0;
}

void rbpf_ksc_disable_liu_west(RBPF_KSC *rbpf)
{
    rbpf->liu_west.enabled = 0;
    rbpf->liu_west.resample_threshold = RBPF_REAL(0.5); /* Back to normal */
}

/* Fine-tune Liu-West learning behavior */
void rbpf_ksc_set_liu_west_resample(RBPF_KSC *rbpf,
                                    rbpf_real_t ess_threshold,
                                    int max_ticks_no_resample)
{
    rbpf->liu_west.resample_threshold = ess_threshold;
    rbpf->liu_west.max_ticks_no_resample = max_ticks_no_resample;
}

void rbpf_ksc_set_liu_west_bounds(RBPF_KSC *rbpf,
                                  rbpf_real_t min_mu_vol, rbpf_real_t max_mu_vol,
                                  rbpf_real_t min_sigma_vol, rbpf_real_t max_sigma_vol)
{
    rbpf->liu_west.min_mu_vol = min_mu_vol;
    rbpf->liu_west.max_mu_vol = max_mu_vol;
    rbpf->liu_west.min_sigma_vol = min_sigma_vol;
    rbpf->liu_west.max_sigma_vol = max_sigma_vol;
}

void rbpf_ksc_get_learned_params(const RBPF_KSC *rbpf, int regime,
                                 rbpf_real_t *mu_vol_out, rbpf_real_t *sigma_vol_out)
{
    if (regime < 0 || regime >= rbpf->n_regimes)
    {
        *mu_vol_out = rbpf->params[0].mu_vol;
        *sigma_vol_out = rbpf->params[0].sigma_vol;
        return;
    }

    /* Compute weighted average across particles */
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;
    const rbpf_real_t *log_w = rbpf->log_weight;

    /* Find max log weight for numerical stability */
    rbpf_real_t max_log_w = log_w[0];
    for (int i = 1; i < n; i++)
    {
        if (log_w[i] > max_log_w)
            max_log_w = log_w[i];
    }

    rbpf_real_t sum_mu = RBPF_REAL(0.0);
    rbpf_real_t sum_sigma = RBPF_REAL(0.0);
    rbpf_real_t sum_w = RBPF_REAL(0.0);

    for (int i = 0; i < n; i++)
    {
        rbpf_real_t w = rbpf_exp(log_w[i] - max_log_w);
        int idx = i * n_regimes + regime;
        sum_mu += w * rbpf->particle_mu_vol[idx];
        sum_sigma += w * rbpf->particle_sigma_vol[idx];
        sum_w += w;
    }

    *mu_vol_out = sum_mu / sum_w;
    *sigma_vol_out = sum_sigma / sum_w;
}

/**
 * Internal: Compute Liu-West sufficient statistics (weighted mean/var)
 */
static void rbpf_ksc_liu_west_compute_stats(RBPF_KSC *rbpf)
{
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;
    const rbpf_real_t *log_w = rbpf->log_weight;

    /* Find max log weight for numerical stability */
    rbpf_real_t max_log_w = log_w[0];
    for (int i = 1; i < n; i++)
    {
        if (log_w[i] > max_log_w)
            max_log_w = log_w[i];
    }

    /* Compute normalized weights */
    rbpf_real_t sum_w = RBPF_REAL(0.0);
    rbpf_real_t *w = rbpf->scratch1;
    for (int i = 0; i < n; i++)
    {
        w[i] = rbpf_exp(log_w[i] - max_log_w);
        sum_w += w[i];
    }
    rbpf_real_t inv_sum = RBPF_REAL(1.0) / sum_w;
    for (int i = 0; i < n; i++)
    {
        w[i] *= inv_sum;
    }

    /* Compute weighted mean and variance for each regime */
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf_real_t sum_mu = RBPF_REAL(0.0), sum_mu2 = RBPF_REAL(0.0);
        rbpf_real_t sum_sigma = RBPF_REAL(0.0), sum_sigma2 = RBPF_REAL(0.0);

        for (int i = 0; i < n; i++)
        {
            int idx = i * n_regimes + r;
            rbpf_real_t mu_v = rbpf->particle_mu_vol[idx];
            rbpf_real_t sigma_v = rbpf->particle_sigma_vol[idx];

            sum_mu += w[i] * mu_v;
            sum_mu2 += w[i] * mu_v * mu_v;
            sum_sigma += w[i] * sigma_v;
            sum_sigma2 += w[i] * sigma_v * sigma_v;
        }

        rbpf->lw_mu_vol_mean[r] = sum_mu;
        rbpf->lw_mu_vol_var[r] = sum_mu2 - sum_mu * sum_mu;
        if (rbpf->lw_mu_vol_var[r] < RBPF_REAL(1e-10))
            rbpf->lw_mu_vol_var[r] = RBPF_REAL(1e-10);

        rbpf->lw_sigma_vol_mean[r] = sum_sigma;
        rbpf->lw_sigma_vol_var[r] = sum_sigma2 - sum_sigma * sum_sigma;
        if (rbpf->lw_sigma_vol_var[r] < RBPF_REAL(1e-10))
            rbpf->lw_sigma_vol_var[r] = RBPF_REAL(1e-10);
    }
}

/**
 * Internal: Apply Liu-West shrinkage + jitter after resample
 *
 * For each parameter θ:
 *   θ_new = a × θ_parent + (1-a) × θ_mean + h × randn()
 *
 * where h = sqrt(1 - a²) × sqrt(Var(θ))
 */
static void rbpf_ksc_liu_west_resample(RBPF_KSC *rbpf, const int *indices)
{
    if (!rbpf->liu_west.enabled)
        return;
    if (rbpf->liu_west.tick_count < rbpf->liu_west.warmup_ticks)
        return;

    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;
    rbpf_real_t a = rbpf->liu_west.shrinkage;
    rbpf_real_t one_minus_a = RBPF_REAL(1.0) - a;
    rbpf_real_t h_scale = rbpf_sqrt(RBPF_REAL(1.0) - a * a); /* Jitter scale */

    RBPF_LiuWest *lw = &rbpf->liu_west;
    rbpf_pcg32_t *rng = &rbpf->pcg[0];

    /* Gather parameters from parents into tmp buffers */
    for (int i = 0; i < n; i++)
    {
        int parent = indices[i];
        for (int r = 0; r < n_regimes; r++)
        {
            int idx_new = i * n_regimes + r;
            int idx_parent = parent * n_regimes + r;

            /* Liu-West update for μ_vol */
            if (lw->learn_mu_vol)
            {
                rbpf_real_t parent_val = rbpf->particle_mu_vol[idx_parent];
                rbpf_real_t mean_val = rbpf->lw_mu_vol_mean[r];
                rbpf_real_t h = h_scale * rbpf_sqrt(rbpf->lw_mu_vol_var[r]);
                rbpf_real_t noise = rbpf_pcg32_gaussian(rng);

                rbpf_real_t new_val = a * parent_val + one_minus_a * mean_val + h * noise;

                /* Clamp to bounds */
                if (new_val < lw->min_mu_vol)
                    new_val = lw->min_mu_vol;
                if (new_val > lw->max_mu_vol)
                    new_val = lw->max_mu_vol;

                rbpf->particle_mu_vol_tmp[idx_new] = new_val;
            }
            else
            {
                rbpf->particle_mu_vol_tmp[idx_new] = rbpf->particle_mu_vol[idx_parent];
            }

            /* Liu-West update for σ_vol (if enabled) */
            if (lw->learn_sigma_vol)
            {
                rbpf_real_t parent_val = rbpf->particle_sigma_vol[idx_parent];
                rbpf_real_t mean_val = rbpf->lw_sigma_vol_mean[r];
                rbpf_real_t h = h_scale * rbpf_sqrt(rbpf->lw_sigma_vol_var[r]);
                rbpf_real_t noise = rbpf_pcg32_gaussian(rng);

                rbpf_real_t new_val = a * parent_val + one_minus_a * mean_val + h * rbpf_fabs(noise);

                /* Clamp to bounds (σ_vol must be positive) */
                if (new_val < lw->min_sigma_vol)
                    new_val = lw->min_sigma_vol;
                if (new_val > lw->max_sigma_vol)
                    new_val = lw->max_sigma_vol;

                rbpf->particle_sigma_vol_tmp[idx_new] = new_val;
            }
            else
            {
                rbpf->particle_sigma_vol_tmp[idx_new] = rbpf->particle_sigma_vol[idx_parent];
            }
        }
    }

    /* Pointer swap */
    rbpf_real_t *tmp_mu = rbpf->particle_mu_vol;
    rbpf->particle_mu_vol = rbpf->particle_mu_vol_tmp;
    rbpf->particle_mu_vol_tmp = tmp_mu;

    rbpf_real_t *tmp_sigma = rbpf->particle_sigma_vol;
    rbpf->particle_sigma_vol = rbpf->particle_sigma_vol_tmp;
    rbpf->particle_sigma_vol_tmp = tmp_sigma;
}

/*─────────────────────────────────────────────────────────────────────────────
 * INITIALIZATION
 *───────────────────────────────────────────────────────────────────────────*/

void rbpf_ksc_init(RBPF_KSC *rbpf, rbpf_real_t mu0, rbpf_real_t var0)
{
    int n = rbpf->n_particles;
    int n_regimes = rbpf->n_regimes;

    /* Spread particles for state diversity
     * Wider spread when Liu-West is enabled to explore parameter space */
    rbpf_real_t state_spread = rbpf->liu_west.enabled ? RBPF_REAL(0.5) : RBPF_REAL(0.1);

    for (int i = 0; i < n; i++)
    {
        rbpf_real_t noise = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * state_spread;
        rbpf->mu[i] = mu0 + noise;
        rbpf->var[i] = var0;
        rbpf->regime[i] = i % n_regimes;
        rbpf->log_weight[i] = RBPF_REAL(0.0); /* log(1) = 0 */
    }

    /* Initialize per-particle Liu-West parameters from global params
     *
     * CRITICAL: If initial params are wrong, particles must spread widely
     * to "find" the true values. With tight spread (0.05), particles can't
     * explore and get stuck near the wrong initial guess.
     *
     * Spread should cover the expected error in initial params.
     * Log-scale spread of 1.0 covers ~3x range in volatility.
     */
    rbpf_real_t param_spread_mu = RBPF_REAL(0.8);    /* Wide spread on μ_vol */
    rbpf_real_t param_spread_sigma = RBPF_REAL(0.1); /* Moderate spread on σ_vol */

    for (int i = 0; i < n; i++)
    {
        for (int r = 0; r < n_regimes; r++)
        {
            int idx = i * n_regimes + r;

            /* Wide jitter to explore parameter space */
            rbpf_real_t jitter_mu = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_mu;
            rbpf_real_t jitter_sigma = rbpf_pcg32_gaussian(&rbpf->pcg[0]) * param_spread_sigma;

            rbpf_real_t mu_vol = rbpf->params[r].mu_vol + jitter_mu;
            rbpf_real_t sigma_vol = rbpf->params[r].sigma_vol + rbpf_fabs(jitter_sigma);

            /* Clamp to bounds if Liu-West is configured */
            if (rbpf->liu_west.enabled)
            {
                if (mu_vol < rbpf->liu_west.min_mu_vol)
                    mu_vol = rbpf->liu_west.min_mu_vol;
                if (mu_vol > rbpf->liu_west.max_mu_vol)
                    mu_vol = rbpf->liu_west.max_mu_vol;
                if (sigma_vol < rbpf->liu_west.min_sigma_vol)
                    sigma_vol = rbpf->liu_west.min_sigma_vol;
                if (sigma_vol > rbpf->liu_west.max_sigma_vol)
                    sigma_vol = rbpf->liu_west.max_sigma_vol;
            }

            rbpf->particle_mu_vol[idx] = mu_vol;
            rbpf->particle_sigma_vol[idx] = sigma_vol;
        }
    }

    /* Reset detection */
    rbpf->detection.vol_ema_short = rbpf_exp(mu0);
    rbpf->detection.vol_ema_long = rbpf_exp(mu0);
    rbpf->detection.prev_regime = 0;
    rbpf->detection.cooldown = 0;

    /* Reset Liu-West tick counter */
    rbpf->liu_west.tick_count = 0;
    rbpf->liu_west.ticks_since_resample = 0;
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
 *   - Liu-West: uses per-particle parameters when enabled
 *───────────────────────────────────────────────────────────────────────────*/

static void rbpf_ksc_predict(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;
    const RBPF_RegimeParams *params = rbpf->params;
    const int n_regimes = rbpf->n_regimes;
    const int lw_enabled = rbpf->liu_west.enabled &&
                           (rbpf->liu_west.tick_count >= rbpf->liu_west.warmup_ticks);

    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    const int *restrict regime = rbpf->regime;
    rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
    rbpf_real_t *restrict var_pred = rbpf->var_pred;

    if (lw_enabled)
    {
        /* Use per-particle learned parameters */
        const rbpf_real_t *particle_mu_vol = rbpf->particle_mu_vol;
        const rbpf_real_t *particle_sigma_vol = rbpf->particle_sigma_vol;

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];
            rbpf_real_t theta = params[r].theta; /* θ still from global (not learned) */
            rbpf_real_t omt = RBPF_REAL(1.0) - theta;
            rbpf_real_t omt2 = omt * omt;

            /* Per-particle μ_vol and σ_vol */
            int idx = i * n_regimes + r;
            rbpf_real_t mv = particle_mu_vol[idx];
            rbpf_real_t sigma_vol = particle_sigma_vol[idx];
            rbpf_real_t q = sigma_vol * sigma_vol;

            mu_pred[i] = omt * mu[i] + theta * mv;
            var_pred[i] = omt2 * var[i] + q;
        }
    }
    else
    {
        /* Use global regime parameters (original behavior) */
        rbpf_real_t theta_r[RBPF_MAX_REGIMES];
        rbpf_real_t mu_vol_r[RBPF_MAX_REGIMES];
        rbpf_real_t q_r[RBPF_MAX_REGIMES];
        rbpf_real_t one_minus_theta_r[RBPF_MAX_REGIMES];
        rbpf_real_t one_minus_theta_sq_r[RBPF_MAX_REGIMES];

        for (int r = 0; r < n_regimes; r++)
        {
            theta_r[r] = params[r].theta;
            mu_vol_r[r] = params[r].mu_vol;
            q_r[r] = params[r].q;
            one_minus_theta_r[r] = RBPF_REAL(1.0) - theta_r[r];
            one_minus_theta_sq_r[r] = one_minus_theta_r[r] * one_minus_theta_r[r];
        }

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];
            rbpf_real_t omt = one_minus_theta_r[r];
            rbpf_real_t omt2 = one_minus_theta_sq_r[r];
            rbpf_real_t th = theta_r[r];
            rbpf_real_t mv = mu_vol_r[r];
            rbpf_real_t q = q_r[r];

            mu_pred[i] = omt * mu[i] + th * mv;
            var_pred[i] = omt2 * var[i] + q;
        }
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
static const rbpf_real_t KSC_LOG_PROB[KSC_N_COMPONENTS] = {
    -4.920f,  /* log(0.00730) */
    -2.248f,  /* log(0.10556) */
    -10.820f, /* log(0.00002) */
    -3.125f,  /* log(0.04395) */
    -1.079f,  /* log(0.34001) */
    -1.404f,  /* log(0.24566) */
    -1.356f   /* log(0.25750) */
};

static rbpf_real_t rbpf_ksc_update(RBPF_KSC *rbpf, rbpf_real_t y)
{
    const int n = rbpf->n_particles;
    const rbpf_real_t H = RBPF_REAL(2.0);
    const rbpf_real_t H2 = RBPF_REAL(4.0);
    const rbpf_real_t NEG_HALF = RBPF_REAL(-0.5);

    rbpf_real_t *restrict mu_pred = rbpf->mu_pred;
    rbpf_real_t *restrict var_pred = rbpf->var_pred;
    rbpf_real_t *restrict mu = rbpf->mu;
    rbpf_real_t *restrict var = rbpf->var;
    rbpf_real_t *restrict log_weight = rbpf->log_weight;
    rbpf_real_t *restrict lik_total = rbpf->lik_total;
    rbpf_real_t *restrict mu_accum = rbpf->mu_accum;
    rbpf_real_t *restrict var_accum = rbpf->var_accum;
    rbpf_real_t *restrict log_lik_buf = rbpf->log_lik_buffer;
    rbpf_real_t *restrict max_ll = rbpf->max_log_lik;

    /*
     * Log-Sum-Exp approach for numerical stability:
     * 1. Compute log_lik for all K components, store in buffer
     * 2. Find max log_lik per particle
     * 3. Sum exp(log_lik - max) to avoid underflow
     * 4. log(sum) + max = log(total_lik)
     *
     * Variance calculation uses law of total variance:
     *   Var[X] = E[Var[X|K]] + Var[E[X|K]]
     *          = E[X²] - E[X]²
     *
     * We accumulate: E[X²] = Σ wₖ (σₖ² + μₖ²)
     * Then compute:  Var = E[X²] - (E[X])²
     */

    /* Initialize max to very negative */
    for (int i = 0; i < n; i++)
    {
        max_ll[i] = RBPF_REAL(-1e30);
    }

    /* Pass 1: Compute log-likelihoods for all components */
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        const rbpf_real_t m_k = KSC_MEAN[k];
        const rbpf_real_t v2_k = KSC_VAR[k];
        const rbpf_real_t log_pi_k = KSC_LOG_PROB[k];
        const rbpf_real_t y_adj = y - m_k;

        rbpf_real_t *log_lik_k = log_lik_buf + k * n; /* Pointer to component k's buffer */

        RBPF_PRAGMA_SIMD
        for (int i = 0; i < n; i++)
        {
            /* Innovation */
            rbpf_real_t innov = y_adj - H * mu_pred[i];

            /* Innovation variance */
            rbpf_real_t S = H2 * var_pred[i] + v2_k;

            /* Log-likelihood: -0.5*(log(S) + innov²/S) + log(π_k) */
            rbpf_real_t innov2_S = innov * innov / S;
            rbpf_real_t log_lik = NEG_HALF * (rbpf_log(S) + innov2_S) + log_pi_k;

            log_lik_k[i] = log_lik;

            /* Track max for log-sum-exp */
            if (log_lik > max_ll[i])
                max_ll[i] = log_lik;
        }
    }

    /* Zero accumulators */
    memset(lik_total, 0, n * sizeof(rbpf_real_t));
    memset(mu_accum, 0, n * sizeof(rbpf_real_t));
    memset(var_accum, 0, n * sizeof(rbpf_real_t)); /* Now holds E[X²] = Σ wₖ(σₖ² + μₖ²) */

    /* Pass 2: Compute normalized likelihoods and accumulate */
    for (int k = 0; k < KSC_N_COMPONENTS; k++)
    {
        const rbpf_real_t m_k = KSC_MEAN[k];
        const rbpf_real_t v2_k = KSC_VAR[k];
        const rbpf_real_t y_adj = y - m_k;

        rbpf_real_t *log_lik_k = log_lik_buf + k * n;

        RBPF_PRAGMA_SIMD
        for (int i = 0; i < n; i++)
        {
            /* Stable exponential: exp(log_lik - max) */
            rbpf_real_t lik = rbpf_exp(log_lik_k[i] - max_ll[i]);

            /* Accumulate total likelihood */
            lik_total[i] += lik;

            /* Recompute Kalman update for this component */
            rbpf_real_t innov = y_adj - H * mu_pred[i];
            rbpf_real_t S = H2 * var_pred[i] + v2_k;
            rbpf_real_t K = H * var_pred[i] / S;
            rbpf_real_t mu_k = mu_pred[i] + K * innov;
            rbpf_real_t var_k = (RBPF_REAL(1.0) - K * H) * var_pred[i];

            /* Accumulate E[X] = Σ wₖ μₖ */
            mu_accum[i] += lik * mu_k;

            /* Accumulate E[X²] = Σ wₖ (σₖ² + μₖ²)
             * This captures BOTH the average variance AND the spread of means
             * (Law of total variance: Var = E[Var|K] + Var[E|K] = E[X²] - E[X]²) */
            var_accum[i] += lik * (var_k + mu_k * mu_k);
        }
    }

    /* Normalize and update weights */
    rbpf_real_t total_marginal = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t inv_lik = RBPF_REAL(1.0) / (lik_total[i] + RBPF_REAL(1e-30));

        /* E[X] = Σ wₖ μₖ / Σ wₖ */
        rbpf_real_t mean_final = mu_accum[i] * inv_lik;

        /* E[X²] = Σ wₖ (σₖ² + μₖ²) / Σ wₖ */
        rbpf_real_t E_X2 = var_accum[i] * inv_lik;

        /* Var[X] = E[X²] - E[X]² (law of total variance) */
        rbpf_real_t var_final = E_X2 - mean_final * mean_final;

        mu[i] = mean_final;
        var[i] = var_final;

        /* Floor variance (should rarely trigger now with correct calculation) */
        if (var[i] < RBPF_REAL(1e-6))
            var[i] = RBPF_REAL(1e-6);

        /* Update log-weight: log(sum * exp(max)) = log(sum) + max */
        log_weight[i] += rbpf_log(lik_total[i] + RBPF_REAL(1e-30)) + max_ll[i];

        /* Marginal uses un-normalized likelihood */
        total_marginal += lik_total[i] * rbpf_exp(max_ll[i]);
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
        rbpf_real_t u = rbpf_pcg32_uniform(rng);
        int lut_idx = (int)(u * RBPF_REAL(1023.0));
        regime[i] = rbpf->trans_lut[r_old][lut_idx];
    }
}

/*─────────────────────────────────────────────────────────────────────────────
 * RESAMPLE (systematic + regularization)
 *───────────────────────────────────────────────────────────────────────────*/

static int rbpf_ksc_resample(RBPF_KSC *rbpf)
{
    const int n = rbpf->n_particles;

    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *w_norm = rbpf->w_norm;
    rbpf_real_t *cumsum = rbpf->cumsum;
    int *indices = rbpf->indices;

    /* Find max log-weight for numerical stability */
    rbpf_real_t max_lw = log_weight[0];
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
    rbpf_vsExp(n, w_norm, w_norm);

    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
    {
        /* All weights collapsed - reset to uniform */
        rbpf_real_t uw = rbpf->uniform_weight;
        for (int i = 0; i < n; i++)
        {
            w_norm[i] = uw;
        }
        sum_w = RBPF_REAL(1.0);
    }
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /* Compute ESS */
    rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
    rbpf_real_t ess = RBPF_REAL(1.0) / sum_w2;
    rbpf->last_ess = ess;

    /* Adaptive resampling threshold
     *
     * Normal mode: resample when ESS < 50% (standard)
     * Learning mode: resample when ESS < 85% OR forced every N ticks
     *
     * CRITICAL: Liu-West only learns during resample!
     * High ESS = no resample = no learning = parameters stuck
     */
    rbpf_real_t threshold;
    int force_resample = 0;

    if (rbpf->liu_west.enabled)
    {
        threshold = rbpf->liu_west.resample_threshold;
        rbpf->liu_west.ticks_since_resample++;

        /* Force resample periodically to ensure learning happens */
        if (rbpf->liu_west.ticks_since_resample >= rbpf->liu_west.max_ticks_no_resample)
        {
            force_resample = 1;
        }
    }
    else
    {
        threshold = RBPF_REAL(0.5);
    }

    /* Skip resample if ESS is high enough AND not forced */
    if (ess > n * threshold && !force_resample)
    {
        return 0;
    }

    /* Reset counter on resample */
    if (rbpf->liu_west.enabled)
    {
        rbpf->liu_west.ticks_since_resample = 0;
    }

    /* Cumulative sum */
    cumsum[0] = w_norm[0];
    for (int i = 1; i < n; i++)
    {
        cumsum[i] = cumsum[i - 1] + w_norm[i];
    }

    /* Fused systematic resampling + data copy
     * - Single pass: generate index and copy immediately
     * - Keeps source data in cache if selected multiple times */
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    int *regime = rbpf->regime;
    rbpf_real_t *mu_tmp = rbpf->mu_tmp;
    rbpf_real_t *var_tmp = rbpf->var_tmp;
    int *regime_tmp = rbpf->regime_tmp;

    rbpf_real_t u0 = rbpf_pcg32_uniform(&rbpf->pcg[0]) * rbpf->inv_n;
    int j = 0;
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t u = u0 + (rbpf_real_t)i * rbpf->inv_n;
        while (j < n - 1 && cumsum[j] < u)
            j++;

        /* Store index (still needed for Liu-West) */
        indices[i] = j;

        /* Copy immediately - keeps mu[j] hot in cache */
        mu_tmp[i] = mu[j];
        var_tmp[i] = var[j];
        regime_tmp[i] = regime[j];
    }

    /* Pointer swap (no memcpy!) */
    rbpf->mu = mu_tmp;
    rbpf->mu_tmp = mu;
    rbpf->var = var_tmp;
    rbpf->var_tmp = var;
    rbpf->regime = regime_tmp;
    rbpf->regime_tmp = regime;

    /* Reset log-weights to 0 */
    memset(rbpf->log_weight, 0, n * sizeof(rbpf_real_t));

    /* Apply regularization (kernel jitter) */
    rbpf_real_t ess_ratio = ess / (rbpf_real_t)n;
    rbpf_real_t scale = rbpf->reg_scale_max -
                        (rbpf->reg_scale_max - rbpf->reg_scale_min) * ess_ratio;
    if (scale < rbpf->reg_scale_min)
        scale = rbpf->reg_scale_min;
    if (scale > rbpf->reg_scale_max)
        scale = rbpf->reg_scale_max;

    rbpf_real_t h_mu = rbpf->reg_bandwidth_mu * scale;
    rbpf_real_t h_var = rbpf->reg_bandwidth_var * scale;

    /* Generate Gaussian randoms in batch using MKL (ICDF method)
     * Much faster than scalar PCG32 calls in loop */
    rbpf_real_t *gauss = rbpf->rng_gaussian;
    RBPF_VSL_RNG_GAUSSIAN(VSL_RNG_METHOD_GAUSSIAN_ICDF, rbpf->mkl_rng[0],
                          2 * n, gauss, RBPF_REAL(0.0), RBPF_REAL(1.0));

    /* Apply jitter: first n randoms for mu, next n for var */
    mu = rbpf->mu;
    var = rbpf->var;
    regime = rbpf->regime;

    RBPF_PRAGMA_SIMD
    for (int i = 0; i < n; i++)
    {
        mu[i] += h_mu * gauss[i];
        var[i] += h_var * rbpf_fabs(gauss[n + i]);
        if (var[i] < RBPF_REAL(1e-6))
            var[i] = RBPF_REAL(1e-6);
    }

    /* Regime diversity preservation
     *
     * Problem: Standard resampling can kill minority regimes. When volatility
     * is calm, regime 3 (crisis) particles get low weight and die out. Later,
     * when a crisis hits, there are no regime 3 particles to respond!
     *
     * Solution: Ensure minimum particles per regime through two mechanisms:
     * 1. Random mutation: Some particles randomly switch regime
     * 2. Stratification: Force minimum count per regime (if needed)
     */
    if (rbpf->regime_mutation_prob > RBPF_REAL(0.0))
    {
        int n_regimes = rbpf->n_regimes;
        rbpf_pcg32_t *rng_mut = &rbpf->pcg[0];

        /* Count current regime distribution */
        int regime_count[RBPF_MAX_REGIMES] = {0};
        for (int i = 0; i < n; i++)
        {
            regime_count[regime[i]]++;
        }

        /* Find regimes that need more particles */
        int min_count = rbpf->min_particles_per_regime;

        for (int i = 0; i < n; i++)
        {
            int r = regime[i];

            /* Only mutate particles from over-represented regimes */
            if (regime_count[r] > min_count * 2)
            {
                if (rbpf_pcg32_uniform(rng_mut) < rbpf->regime_mutation_prob)
                {
                    /* Find under-represented regime */
                    for (int r_new = 0; r_new < n_regimes; r_new++)
                    {
                        if (regime_count[r_new] < min_count)
                        {
                            /* Mutate to new regime */
                            regime_count[r]--;
                            regime_count[r_new]++;
                            regime[i] = r_new;

                            /* Adapt state toward new regime's mu_vol */
                            rbpf_real_t mu_new = rbpf->params[r_new].mu_vol;
                            mu[i] = RBPF_REAL(0.7) * mu[i] + RBPF_REAL(0.3) * mu_new;
                            break;
                        }
                    }
                }
            }
        }
    }

    /* Liu-West parameter learning: compute stats then apply shrinkage */
    if (rbpf->liu_west.enabled)
    {
        rbpf_ksc_liu_west_compute_stats(rbpf);
        rbpf_ksc_liu_west_resample(rbpf, indices);
    }

    return 1;
}

/*─────────────────────────────────────────────────────────────────────────────
 * COMPUTE OUTPUTS
 *───────────────────────────────────────────────────────────────────────────*/

static void rbpf_ksc_compute_outputs(RBPF_KSC *rbpf, rbpf_real_t marginal_lik,
                                     RBPF_KSC_Output *out)
{
    const int n = rbpf->n_particles;
    const int n_regimes = rbpf->n_regimes;

    rbpf_real_t *log_weight = rbpf->log_weight;
    rbpf_real_t *mu = rbpf->mu;
    rbpf_real_t *var = rbpf->var;
    int *regime = rbpf->regime;
    rbpf_real_t *w_norm = rbpf->w_norm;

    /* Normalize weights */
    rbpf_real_t max_lw = log_weight[0];
    for (int i = 1; i < n; i++)
    {
        if (log_weight[i] > max_lw)
            max_lw = log_weight[i];
    }

    for (int i = 0; i < n; i++)
    {
        w_norm[i] = log_weight[i] - max_lw;
    }
    rbpf_vsExp(n, w_norm, w_norm);

    rbpf_real_t sum_w = rbpf_cblas_asum(n, w_norm, 1);
    if (sum_w < RBPF_REAL(1e-30))
        sum_w = RBPF_REAL(1.0);
    rbpf_cblas_scal(n, RBPF_REAL(1.0) / sum_w, w_norm, 1);

    /* Log-vol mean and variance (using law of total variance) */
    rbpf_real_t log_vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        log_vol_mean += w_norm[i] * mu[i];
    }

    rbpf_real_t log_vol_var = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        rbpf_real_t diff = mu[i] - log_vol_mean;
        /* Var[X] = E[Var[X|particle]] + Var[E[X|particle]] */
        log_vol_var += w_norm[i] * (var[i] + diff * diff);
    }

    out->log_vol_mean = log_vol_mean;
    out->log_vol_var = log_vol_var;

    /* Vol mean: TRUE Monte Carlo estimate over particle mixture
     *
     * Each particle i represents a Gaussian: ℓ ~ N(μ_i, σ²_i)
     * For a log-normal: E[exp(ℓ)|particle i] = exp(μ_i + ½σ²_i)
     *
     * True mixture expectation:
     *   E[exp(ℓ)] = Σ_i w_i × E[exp(ℓ)|i] = Σ_i w_i × exp(μ_i + ½var_i)
     *
     * This is more accurate than the single-Gaussian approximation:
     *   exp(E[ℓ] + ½Var[ℓ])  ← WRONG for mixtures
     */
    rbpf_real_t vol_mean = RBPF_REAL(0.0);
    for (int i = 0; i < n; i++)
    {
        vol_mean += w_norm[i] * rbpf_exp(mu[i] + RBPF_REAL(0.5) * var[i]);
    }
    out->vol_mean = vol_mean;

    /* ESS */
    rbpf_real_t sum_w2 = rbpf_cblas_dot(n, w_norm, 1, w_norm, 1);
    out->ess = RBPF_REAL(1.0) / sum_w2;

    /* Regime probabilities */
    memset(out->regime_probs, 0, sizeof(out->regime_probs));
    for (int i = 0; i < n; i++)
    {
        out->regime_probs[regime[i]] += w_norm[i];
    }

    /* Dominant regime */
    int dom = 0;
    rbpf_real_t max_prob = out->regime_probs[0];
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
    out->surprise = -rbpf_log(marginal_lik + RBPF_REAL(1e-30));

    /* Regime entropy: -Σ p*log(p) */
    rbpf_real_t entropy = RBPF_REAL(0.0);
    for (int r = 0; r < n_regimes; r++)
    {
        rbpf_real_t p = out->regime_probs[r];
        if (p > RBPF_REAL(1e-10))
        {
            entropy -= p * rbpf_log(p);
        }
    }
    out->regime_entropy = entropy;

    /* Vol ratio (vs EMA) */
    RBPF_Detection *det = &rbpf->detection;
    det->vol_ema_short = RBPF_REAL(0.1) * out->vol_mean + 0.9f * det->vol_ema_short;
    det->vol_ema_long = RBPF_REAL(0.01) * out->vol_mean + 0.99f * det->vol_ema_long;
    out->vol_ratio = det->vol_ema_short / (det->vol_ema_long + RBPF_REAL(1e-10));

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
        int vol_shock = (out->vol_ratio > 1.8f) || (out->vol_ratio < RBPF_REAL(0.5));

        /* Surprise: observation unlikely under model */
        int surprised = (out->surprise > RBPF_REAL(5.0));

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

void rbpf_ksc_step(RBPF_KSC *rbpf, rbpf_real_t obs, RBPF_KSC_Output *output)
{
    /* Transform observation: y = log(r²) */
    rbpf_real_t y;
    if (rbpf_fabs(obs) < RBPF_REAL(1e-10))
    {
        y = RBPF_REAL(-23.0); /* Floor at ~log(1e-10²) */
    }
    else
    {
        y = rbpf_log(obs * obs);
    }

    /* 1. Regime transition */
    rbpf_ksc_transition(rbpf);

    /* 2. Kalman predict */
    rbpf_ksc_predict(rbpf);

    /* 3. Mixture Kalman update */
    rbpf_real_t marginal_lik = rbpf_ksc_update(rbpf, y);

    /* 4. Compute outputs (before resample) */
    rbpf_ksc_compute_outputs(rbpf, marginal_lik, output);

    /* 5. Resample if needed (includes Liu-West update) */
    output->resampled = rbpf_ksc_resample(rbpf);

    /* 6. Liu-West: increment tick counter and populate learned params */
    if (rbpf->liu_west.enabled)
    {
        rbpf->liu_west.tick_count++;

        /* Output current learned parameters (weighted averages) */
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            rbpf_ksc_get_learned_params(rbpf, r,
                                        &output->learned_mu_vol[r],
                                        &output->learned_sigma_vol[r]);
        }
    }
    else
    {
        /* Not learning - just copy global params */
        for (int r = 0; r < rbpf->n_regimes; r++)
        {
            output->learned_mu_vol[r] = rbpf->params[r].mu_vol;
            output->learned_sigma_vol[r] = rbpf->params[r].sigma_vol;
        }
    }
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
    rbpf_vsExp(n, rbpf->mu, rbpf->scratch1);
    rbpf_vsLn(n, rbpf->var, rbpf->scratch2);

    /* Warmup BLAS */
    volatile rbpf_real_t sum = rbpf_cblas_asum(n, rbpf->w_norm, 1);
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

    printf("\n  Liu-West Parameter Learning:\n");
    printf("    Enabled:     %s\n", rbpf->liu_west.enabled ? "YES" : "NO");
    if (rbpf->liu_west.enabled)
    {
        printf("    Shrinkage:   %.4f\n", rbpf->liu_west.shrinkage);
        printf("    Warmup:      %d ticks\n", rbpf->liu_west.warmup_ticks);
        printf("    Learn μ_vol: %s\n", rbpf->liu_west.learn_mu_vol ? "YES" : "NO");
        printf("    Learn σ_vol: %s\n", rbpf->liu_west.learn_sigma_vol ? "YES" : "NO");
        printf("    μ_vol bounds: [%.4f, %.4f]\n",
               rbpf->liu_west.min_mu_vol, rbpf->liu_west.max_mu_vol);
    }

    printf("\n  Per-regime parameters (initial):\n");
    printf("  %-8s %8s %8s %8s\n", "Regime", "theta", "mu_vol", "sigma_vol");
    for (int r = 0; r < rbpf->n_regimes; r++)
    {
        const RBPF_RegimeParams *p = &rbpf->params[r];
        printf("  %-8d %8.4f %8.4f %8.4f\n",
               r, p->theta, p->mu_vol, p->sigma_vol);
    }
}