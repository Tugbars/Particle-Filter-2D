/* particle_filter.c
 * High-performance particle filter with Intel MKL
 * Part of quantitative trading stack: SSA → BOCPD → PF → Kelly
 */
#include "particle_filter.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

/* ========================================================================== */
/* HELPERS                                                                    */
/* ========================================================================== */

static inline pf_real *aligned_alloc_real(int n)
{
    return (pf_real *)mkl_malloc(n * sizeof(pf_real), PF_ALIGN);
}

static inline int *aligned_alloc_int(int n)
{
    return (int *)mkl_malloc(n * sizeof(int), PF_ALIGN);
}

/* ========================================================================== */
/* CREATE / DESTROY                                                           */
/* ========================================================================== */

ParticleFilter *pf_create(int n_particles, int n_regimes)
{
    ParticleFilter *pf = (ParticleFilter *)mkl_calloc(1, sizeof(ParticleFilter), PF_ALIGN);
    if (!pf)
        return NULL;

    pf->n_particles = n_particles;
    pf->n_regimes = n_regimes < PF_MAX_REGIMES ? n_regimes : PF_MAX_REGIMES;
    pf->uniform_weight = (pf_real)1.0 / n_particles;

    /* Get thread count */
    pf->n_threads = omp_get_max_threads();
    if (pf->n_threads > PF_MAX_THREADS)
        pf->n_threads = PF_MAX_THREADS;

    /* Allocate aligned buffers */
    pf->states = aligned_alloc_real(n_particles);
    pf->states_tmp = aligned_alloc_real(n_particles);
    pf->weights = aligned_alloc_real(n_particles);
    pf->log_weights = aligned_alloc_real(n_particles);
    pf->cumsum = aligned_alloc_real(n_particles);
    pf->noise = aligned_alloc_real(n_particles);
    pf->uniform = aligned_alloc_real(n_particles);
    pf->scratch = aligned_alloc_real(n_particles);
    pf->regimes = aligned_alloc_int(n_particles);
    pf->regimes_tmp = aligned_alloc_int(n_particles);

    if (!pf->states || !pf->states_tmp || !pf->weights ||
        !pf->log_weights || !pf->cumsum || !pf->noise ||
        !pf->uniform || !pf->scratch || !pf->regimes || !pf->regimes_tmp)
    {
        pf_destroy(pf);
        return NULL;
    }

    /* Initialize thread-local MKL RNG streams (fallback) */
    for (int t = 0; t < pf->n_threads; t++)
    {
        vslNewStream(&pf->rng[t], VSL_BRNG_SFMT19937, 42 + t * 8192);
    }

    /* Initialize PCG RNG streams (default: enabled for small N) */
    for (int t = 0; t < pf->n_threads; t++)
    {
        pcg32_seed(&pf->pcg[t], 42 + t * 12345, t * 67890);
    }
    pf->use_pcg = (n_particles < PF_BLAS_THRESHOLD) ? 1 : 0;

    /* Set MKL to fastest math mode */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);

    /* Default regime parameters */
    pf_set_regime_params(pf, 0, (pf_real)0.001, (pf_real)0.0, (pf_real)0.0, (pf_real)0.01);
    pf_set_regime_params(pf, 1, (pf_real)0.0, (pf_real)0.0, (pf_real)0.1, (pf_real)0.01);
    pf_set_regime_params(pf, 2, (pf_real)0.0, (pf_real)0.0, (pf_real)0.0, (pf_real)0.05);
    pf_set_regime_params(pf, 3, (pf_real)0.0, (pf_real)0.0, (pf_real)0.0, (pf_real)0.10);

    /* Default observation variance */
    pf_set_observation_variance(pf, (pf_real)0.0001);

    /* Adaptive resampling defaults */
    pf->resample_threshold = PF_RESAMPLE_THRESH_DEFAULT;
    pf->volatility_ema = (pf_real)0.01;
    pf->volatility_baseline = (pf_real)0.01;

    /* Uniform initial weights */
    for (int i = 0; i < n_particles; i++)
    {
        pf->weights[i] = pf->uniform_weight;
    }

    /* Initialize regime LUT to 0 */
    memset(pf->regime_lut, 0, PF_REGIME_LUT_SIZE);

    return pf;
}

void pf_destroy(ParticleFilter *pf)
{
    if (!pf)
        return;

    /* Free all thread-local RNG streams */
    for (int t = 0; t < pf->n_threads; t++)
    {
        if (pf->rng[t])
            vslDeleteStream(&pf->rng[t]);
    }

    if (pf->states)
        mkl_free(pf->states);
    if (pf->states_tmp)
        mkl_free(pf->states_tmp);
    if (pf->weights)
        mkl_free(pf->weights);
    if (pf->log_weights)
        mkl_free(pf->log_weights);
    if (pf->cumsum)
        mkl_free(pf->cumsum);
    if (pf->noise)
        mkl_free(pf->noise);
    if (pf->uniform)
        mkl_free(pf->uniform);
    if (pf->scratch)
        mkl_free(pf->scratch);
    if (pf->regimes)
        mkl_free(pf->regimes);
    if (pf->regimes_tmp)
        mkl_free(pf->regimes_tmp);

    mkl_free(pf);
}

/* ========================================================================== */
/* CONFIGURATION                                                              */
/* ========================================================================== */

void pf_set_observation_variance(ParticleFilter *pf, pf_real var)
{
    pf->obs_variance = var;
    pf->inv_obs_variance = (pf_real)1.0 / var;
    pf->neg_half_inv_var = (pf_real)-0.5 / var;
}

void pf_set_regime_params(ParticleFilter *pf, int regime,
                          pf_real drift, pf_real mean, pf_real theta, pf_real sigma)
{
    if (regime < 0 || regime >= PF_MAX_REGIMES)
        return;
    pf->drift[regime] = drift;
    pf->mean[regime] = mean;
    pf->theta[regime] = theta;
    pf->sigma[regime] = sigma;
}

/* ========================================================================== */
/* PRECOMPUTE (call when SSA refreshes every 50-100 ticks)                    */
/* ========================================================================== */

void pf_precompute(ParticleFilter *pf, const SSAFeatures *ssa)
{
    pf_real trend_factor = (pf_real)1.0 + ssa->trend;
    pf_real vol_factor = (pf_real)1.0 + ssa->volatility;

    for (int r = 0; r < pf->n_regimes; r++)
    {
        pf->pre.drift_scaled[r] = pf->drift[r] * trend_factor;
        pf->pre.sigma_scaled[r] = pf->sigma[r] * vol_factor;
        pf->pre.one_minus_theta[r] = (pf_real)1.0 - pf->theta[r];
        pf->pre.theta_mean[r] = pf->theta[r] * pf->mean[r];
    }
}

void pf_set_regime_probs(RegimeProbs *rp, const pf_real *probs, int n)
{
    rp->n_regimes = n;
    rp->cumprobs[0] = probs[0];
    rp->probs[0] = probs[0];

    for (int i = 1; i < n; i++)
    {
        rp->probs[i] = probs[i];
        rp->cumprobs[i] = rp->cumprobs[i - 1] + probs[i];
    }
    rp->cumprobs[n - 1] = (pf_real)1.0; /* Ensure exact 1.0 */
}

/* Build regime lookup table for O(1) sampling - call after pf_set_regime_probs */
void pf_build_regime_lut(ParticleFilter *pf, const RegimeProbs *rp)
{
    for (int i = 0; i < PF_REGIME_LUT_SIZE; i++)
    {
        pf_real u = (pf_real)i / (pf_real)PF_REGIME_LUT_SIZE;
        int regime = rp->n_regimes - 1;
        for (int r = 0; r < rp->n_regimes - 1; r++)
        {
            if (u < rp->cumprobs[r])
            {
                regime = r;
                break;
            }
        }
        pf->regime_lut[i] = (uint8_t)regime;
    }
}

/* Enable/disable PCG RNG (faster than MKL for small N) */
void pf_enable_pcg(ParticleFilter *pf, int enable)
{
    pf->use_pcg = enable;
}

/* Set adaptive resampling based on volatility */
void pf_set_resample_adaptive(ParticleFilter *pf, pf_real baseline_volatility)
{
    pf->volatility_baseline = baseline_volatility;
    pf->volatility_ema = baseline_volatility;
    pf->resample_threshold = PF_RESAMPLE_THRESH_DEFAULT;
}

/* Update adaptive resample threshold based on current volatility */
static inline void pf_update_resample_threshold(ParticleFilter *pf, pf_real current_variance)
{
    pf_real alpha = (pf_real)0.05; /* Slow adaptation */
    pf_real current_vol = (pf_real)sqrt((double)current_variance);

    /* EMA of volatility */
    pf->volatility_ema = alpha * current_vol + ((pf_real)1.0 - alpha) * pf->volatility_ema;

    /* Ratio to baseline */
    pf_real vol_ratio = pf->volatility_ema / (pf->volatility_baseline + (pf_real)1e-10);

    /* High vol → lower threshold (resample less often, preserve diversity)
     * Low vol → higher threshold (resample more often, track mode) */
    if (vol_ratio > (pf_real)2.0)
    {
        pf->resample_threshold = PF_RESAMPLE_THRESH_MIN;
    }
    else if (vol_ratio < (pf_real)0.5)
    {
        pf->resample_threshold = PF_RESAMPLE_THRESH_MAX;
    }
    else
    {
        /* Linear interpolation */
        pf_real t = (vol_ratio - (pf_real)0.5) / (pf_real)1.5;
        pf->resample_threshold = PF_RESAMPLE_THRESH_MAX - t * (PF_RESAMPLE_THRESH_MAX - PF_RESAMPLE_THRESH_MIN);
    }
}

/* ========================================================================== */
/* INITIALIZE                                                                 */
/* ========================================================================== */

void pf_initialize(ParticleFilter *pf, pf_real x0, pf_real spread)
{
    int n = pf->n_particles;

    if (pf->use_pcg)
    {
/* PCG: each thread uses its own stream for reproducibility */
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            pcg32_random_t *rng = &pf->pcg[tid];

#pragma omp for
            for (int i = 0; i < n; i++)
            {
                pf->states[i] = x0 + spread * pcg32_gaussian(rng);
            }
        }
    }
    else
    {
        /* MKL: bulk Gaussian generation */
        pf_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->rng[0],
                       n, pf->states, x0, spread);
    }

    /* Uniform weights */
    for (int i = 0; i < n; i++)
    {
        pf->weights[i] = pf->uniform_weight;
        pf->regimes[i] = i % pf->n_regimes;
    }
}

/* ========================================================================== */
/* PROPAGATE                                                                  */
/* ========================================================================== */

void pf_propagate(ParticleFilter *pf, const RegimeProbs *rp)
{
    int n = pf->n_particles;
    (void)rp; /* Regime probs now baked into LUT */

    /* Use precomputed dynamics */
    const pf_real *drift_s = pf->pre.drift_scaled;
    const pf_real *sigma_s = pf->pre.sigma_scaled;
    const pf_real *omt = pf->pre.one_minus_theta;
    const pf_real *tm = pf->pre.theta_mean;
    const pf_real *theta = pf->theta;
    const uint8_t *lut = pf->regime_lut;

    pf_real *x = pf->states;
    pf_real *noise = pf->noise;
    pf_real *uniform = pf->uniform;
    int *reg = pf->regimes;

    if (pf->use_pcg)
    {
/* PCG RNG: generate inline, fully parallel, no locking */
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            pcg32_random_t *rng = &pf->pcg[tid];

#pragma omp for
            for (int i = 0; i < n; i++)
            {
                /* Generate random numbers inline */
                pf_real ui = pcg32_uniform(rng);
                pf_real wi = pcg32_gaussian(rng);

                /* O(1) regime lookup - use (SIZE-1) to avoid overflow */
                int lut_idx = (int)(ui * (pf_real)(PF_REGIME_LUT_SIZE - 1));
                int r = lut[lut_idx];
                reg[i] = r;

                /* Apply dynamics */
                pf_real xi = x[i];
                if (theta[r] > (pf_real)0.0)
                {
                    xi = omt[r] * xi + tm[r] + sigma_s[r] * wi;
                }
                else
                {
                    xi = xi + drift_s[r] + sigma_s[r] * wi;
                }
                x[i] = xi;
            }
        }
    }
    else
    {
/* MKL RNG: bulk generation, then propagate */
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (n + nt - 1) / nt;
            int start = tid * chunk;
            int end = start + chunk;
            if (end > n)
                end = n;
            int len = end - start;

            if (len > 0)
            {
                pf_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->rng[tid],
                               len, &noise[start], 0.0, 1.0);
                pf_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, pf->rng[tid],
                              len, &uniform[start], 0.0, 1.0);
            }
        }

/* Propagate particles */
#pragma omp parallel for
        for (int i = 0; i < n; i++)
        {
            int lut_idx = (int)(uniform[i] * (pf_real)(PF_REGIME_LUT_SIZE - 1));
            int r = lut[lut_idx];
            reg[i] = r;

            pf_real xi = x[i];
            pf_real wi = noise[i];

            if (theta[r] > (pf_real)0.0)
            {
                xi = omt[r] * xi + tm[r] + sigma_s[r] * wi;
            }
            else
            {
                xi = xi + drift_s[r] + sigma_s[r] * wi;
            }
            x[i] = xi;
        }
    }
}

/* ========================================================================== */
/* UPDATE WEIGHTS                                                             */
/* ========================================================================== */

void pf_update_weights(ParticleFilter *pf, pf_real observation)
{
    int n = pf->n_particles;
    pf_real *s = pf->states;
    pf_real *lw = pf->log_weights;
    pf_real *w = pf->weights;
    pf_real *scratch = pf->scratch;
    pf_real nhiv = pf->neg_half_inv_var;

    /* For small N, avoid BLAS call overhead with manual loops */
    if (n < PF_BLAS_THRESHOLD)
    {
        /* Fused: compute log-weights and find max */
        pf_real max_lw = (pf_real)(-1e30);
        for (int i = 0; i < n; i++)
        {
            pf_real diff = observation - s[i];
            pf_real logw = nhiv * diff * diff;
            lw[i] = logw;
            if (logw > max_lw)
                max_lw = logw;
        }

        /* Subtract max and exp - use VML for exp, it's still faster */
        for (int i = 0; i < n; i++)
        {
            lw[i] -= max_lw;
        }
        pf_vExp(n, lw, w);

        /* Sum - manual reduction (faster than BLAS for small N) */
        pf_real sum = (pf_real)0.0;
        for (int i = 0; i < n; i++)
        {
            sum += w[i];
        }

        /* Degenerate weight detection: all weights underflowed to 0 */
        if (sum == (pf_real)0.0 || !isfinite(sum))
        {
            pf_real uw = pf->uniform_weight;
            for (int i = 0; i < n; i++)
            {
                w[i] = uw;
            }
            return;
        }

        /* Normalize - manual (faster than BLAS scal for small N) */
        pf_real inv_sum = (pf_real)1.0 / sum;
        for (int i = 0; i < n; i++)
        {
            w[i] *= inv_sum;
        }
    }
    else
    {
        /* Large N: use BLAS/VML for parallelism */

        /* Compute residuals: scratch = observation - states */
        for (int i = 0; i < n; i++)
        {
            scratch[i] = observation - s[i];
        }

        /* lw = scratch^2 */
        pf_vSqr(n, scratch, lw);

        /* lw = neg_half_inv_var * lw */
        pf_cblas_scal(n, nhiv, lw, 1);

        /* Find max using BLAS */
        int max_idx = pf_cblas_iamax(n, lw, 1);
        pf_real max_lw = lw[max_idx];

        /* Subtract max */
        for (int i = 0; i < n; i++)
        {
            lw[i] -= max_lw;
        }

        /* Exponentiate */
        pf_vExp(n, lw, w);

        /* Sum using BLAS (weights are positive) */
        pf_real sum = pf_cblas_asum(n, w, 1);

        /* Degenerate weight detection */
        if (sum == (pf_real)0.0 || !isfinite(sum))
        {
            pf_real uw = pf->uniform_weight;
            for (int i = 0; i < n; i++)
            {
                w[i] = uw;
            }
            return;
        }

        /* Normalize */
        pf_real inv_sum = (pf_real)1.0 / sum;
        pf_cblas_scal(n, inv_sum, w, 1);
    }
}

/* ========================================================================== */
/* EFFECTIVE SAMPLE SIZE                                                      */
/* ========================================================================== */

pf_real pf_effective_sample_size(const ParticleFilter *pf)
{
    /* ESS = 1 / sum(w^2) */
    pf_real sum_sq = pf_cblas_dot(pf->n_particles, pf->weights, 1, pf->weights, 1);
    return (pf_real)1.0 / sum_sq;
}

/* ========================================================================== */
/* RESAMPLE (Systematic)                                                      */
/* ========================================================================== */

/* Binary search for resampling - O(log n) per particle */
static inline int binary_search_cumsum(const pf_real *cs, int n, pf_real u)
{
    int lo = 0, hi = n - 1;
    while (lo < hi)
    {
        int mid = (lo + hi) >> 1;
        if (cs[mid] < u)
            lo = mid + 1;
        else
            hi = mid;
    }
    return lo;
}

void pf_resample(ParticleFilter *pf)
{
    int n = pf->n_particles;
    pf_real *w = pf->weights;
    pf_real *cs = pf->cumsum;

    /* Cumulative sum */
    cs[0] = w[0];
    for (int i = 1; i < n; i++)
    {
        cs[i] = cs[i - 1] + w[i];
    }

    /* Get single random number for systematic resampling */
    pf_real u0;
    if (pf->use_pcg)
    {
        u0 = pcg32_uniform(&pf->pcg[0]);
    }
    else
    {
        pf_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, pf->rng[0], 1, &u0, 0.0, 1.0);
    }
    u0 *= pf->uniform_weight;

    pf_real inv_n = pf->uniform_weight;

    /* Check if weights are highly skewed (ESS < 10% of N) */
    pf_real ess = pf_effective_sample_size(pf);
    int use_binary = (ess < n * 0.1);

    if (use_binary)
    {
        /* Binary search - O(N log N) but safe for skewed weights */
        for (int i = 0; i < n; i++)
        {
            pf_real u = u0 + i * inv_n;
            int idx = binary_search_cumsum(cs, n, u);
            pf->states_tmp[i] = pf->states[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    }
    else
    {
        /* Linear scan - O(N) for well-distributed weights */
        int idx = 0;
        for (int i = 0; i < n; i++)
        {
            pf_real u = u0 + i * inv_n;
            while (cs[idx] < u && idx < n - 1)
                idx++;
            pf->states_tmp[i] = pf->states[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    }

    /* Swap pointers */
    pf_real *tmp_s = pf->states;
    pf->states = pf->states_tmp;
    pf->states_tmp = tmp_s;

    int *tmp_r = pf->regimes;
    pf->regimes = pf->regimes_tmp;
    pf->regimes_tmp = tmp_r;

    /* Reset to uniform weights */
    for (int i = 0; i < n; i++)
    {
        w[i] = inv_n;
    }
}

int pf_resample_if_needed(ParticleFilter *pf, pf_real threshold)
{
    (void)threshold; /* Use adaptive threshold instead */
    pf_real ess = pf_effective_sample_size(pf);
    if (ess < pf->n_particles * pf->resample_threshold)
    {
        pf_resample(pf);
        return 1;
    }
    return 0;
}

/* ========================================================================== */
/* ESTIMATES                                                                  */
/* ========================================================================== */

pf_real pf_mean(const ParticleFilter *pf)
{
    /* mean = sum(w * x) = dot(w, x) */
    return pf_cblas_dot(pf->n_particles, pf->weights, 1, pf->states, 1);
}

pf_real pf_variance(const ParticleFilter *pf)
{
    int n = pf->n_particles;
    pf_real m = pf_mean(pf);
    pf_real sum = (pf_real)0.0;

    for (int i = 0; i < n; i++)
    {
        pf_real diff = pf->states[i] - m;
        sum += pf->weights[i] * diff * diff;
    }
    return sum;
}

void pf_regime_distribution(const ParticleFilter *pf, pf_real *out)
{
    for (int r = 0; r < PF_MAX_REGIMES; r++)
    {
        out[r] = (pf_real)0.0;
    }
    for (int i = 0; i < pf->n_particles; i++)
    {
        out[pf->regimes[i]] += pf->weights[i];
    }
}

/* ========================================================================== */
/* FULL UPDATE                                                                */
/* ========================================================================== */

PFOutput pf_update(ParticleFilter *pf, pf_real observation,
                   const RegimeProbs *regime_probs)
{
    PFOutput out;

    /* 1. Propagate (uses precomputed SSA terms) */
    pf_propagate(pf, regime_probs);

    /* 2. Update weights */
    pf_update_weights(pf, observation);

    /* 3. Compute estimates */
    out.mean = pf_mean(pf);
    out.variance = pf_variance(pf);
    out.ess = pf_effective_sample_size(pf);
    pf_regime_distribution(pf, out.regime_probs);

    /* 4. Update adaptive resample threshold based on variance */
    pf_update_resample_threshold(pf, out.variance);

    /* 5. Resample if needed (uses adaptive threshold) */
    out.resampled = pf_resample_if_needed(pf, (pf_real)0.5);

    return out;
}

/* ========================================================================== */
/* DEBUG                                                                      */
/* ========================================================================== */

void pf_print_config(const ParticleFilter *pf)
{
    printf("Particle Filter Configuration:\n");
    printf("  Precision:     %s (%d bytes)\n",
           PF_REAL_SIZE == 4 ? "float" : "double", PF_REAL_SIZE);
    printf("  Particles:     %d\n", pf->n_particles);
    printf("  Regimes:       %d\n", pf->n_regimes);
    printf("  Obs var:       %g\n", (double)pf->obs_variance);
    printf("  MKL threads:   %d\n", mkl_get_max_threads());
    printf("  OMP threads:   %d\n", pf->n_threads);
    printf("  RNG:           %s\n", pf->use_pcg ? "PCG (fast)" : "MKL SFMT");
    printf("  Regime LUT:    %d entries\n", PF_REGIME_LUT_SIZE);
    printf("  BLAS thresh:   %d (manual loops below)\n", PF_BLAS_THRESHOLD);
    printf("  Resample thresh: %.2f (adaptive: %.2f-%.2f)\n",
           (double)pf->resample_threshold,
           (double)PF_RESAMPLE_THRESH_MIN, (double)PF_RESAMPLE_THRESH_MAX);

    for (int r = 0; r < pf->n_regimes; r++)
    {
        printf("  Regime %d: drift=%.4f mean=%.4f theta=%.4f sigma=%.4f\n",
               r, (double)pf->drift[r], (double)pf->mean[r],
               (double)pf->theta[r], (double)pf->sigma[r]);
    }
}
