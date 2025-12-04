/**
 * @file particle_filter_2d.c
 * @brief 2D Particle Filter with Stochastic Volatility - Implementation
 */

#include "particle_filter_2d.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <omp.h>

#ifdef PF2D_HIGH_N
#include <alloca.h>
#endif

/*============================================================================
 * HELPERS
 *============================================================================*/

static inline pf2d_real* aligned_alloc_real(int n) {
    return (pf2d_real*)mkl_malloc(n * sizeof(pf2d_real), PF2D_ALIGN);
}

static inline int* aligned_alloc_int(int n) {
    return (int*)mkl_malloc(n * sizeof(int), PF2D_ALIGN);
}

static inline int binary_search_cumsum(const pf2d_real* cs, int n, pf2d_real u) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (cs[mid] < u) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/*============================================================================
 * HIGH-N OPTIMIZATIONS (enabled with -DPF2D_HIGH_N)
 *============================================================================*/

#ifdef PF2D_HIGH_N

/**
 * Parallel prefix sum (cumulative sum)
 * 
 * Three-phase algorithm:
 *   1. Each thread computes local cumsum for its chunk
 *   2. Serial scan of chunk totals (tiny - just n_threads elements)
 *   3. Each thread adds offset from previous chunks
 * 
 * Speedup: ~4-8x for N > 50k on many-core systems
 */
static void parallel_cumsum(pf2d_real* out, const pf2d_real* in, int n) {
    int nt = omp_get_max_threads();
    
    /* For small N or single thread, use sequential */
    if (n < PF2D_PARALLEL_CUMSUM_THRESH || nt == 1) {
        out[0] = in[0];
        for (int i = 1; i < n; i++) {
            out[i] = out[i-1] + in[i];
        }
        return;
    }
    
    /* Allocate partial sums on stack (small, thread count) */
    pf2d_real* partial = (pf2d_real*)alloca(nt * sizeof(pf2d_real));
    
    /* Phase 1: Local cumulative sums */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunk = (n + nt - 1) / nt;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > n) end = n;
        
        if (start < n) {
            pf2d_real sum = (pf2d_real)0.0;
            for (int i = start; i < end; i++) {
                sum += in[i];
                out[i] = sum;
            }
            partial[tid] = sum;
        } else {
            partial[tid] = (pf2d_real)0.0;
        }
    }
    
    /* Phase 2: Prefix sum of partial totals (serial, but only nt elements) */
    for (int t = 1; t < nt; t++) {
        partial[t] += partial[t-1];
    }
    
    /* Phase 3: Add offsets to each chunk */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        if (tid > 0) {
            int chunk = (n + nt - 1) / nt;
            int start = tid * chunk;
            int end = start + chunk;
            if (end > n) end = n;
            
            pf2d_real offset = partial[tid - 1];
            for (int i = start; i < end; i++) {
                out[i] += offset;
            }
        }
    }
}

/**
 * Parallel batch binary search
 * 
 * Each thread searches a subset of the query points independently.
 * Effective when number of queries (N) is large.
 */
static void parallel_batch_search(
    const pf2d_real* cumsum,    /* Sorted cumulative weights */
    const pf2d_real* queries,   /* Search values (u0 + i/N) */
    int* indices,               /* Output: found indices */
    int n)                      /* Number of particles */
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        indices[i] = binary_search_cumsum(cumsum, n, queries[i]);
    }
}

/**
 * Generate stratified uniform samples in parallel
 * u[i] = u0 + i * step, where u0 ~ Uniform(0, step)
 */
static void parallel_stratified_uniform(
    pf2d_real* out,
    pf2d_real u0,
    pf2d_real step,
    int n)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        out[i] = u0 + i * step;
    }
}

#endif /* PF2D_HIGH_N */

/*============================================================================
 * CREATE / DESTROY
 *============================================================================*/

PF2D* pf2d_create(int n_particles, int n_regimes) {
    PF2D* pf = (PF2D*)mkl_calloc(1, sizeof(PF2D), PF2D_ALIGN);
    if (!pf) return NULL;
    
    pf->n_particles = n_particles;
    pf->n_regimes = n_regimes < PF2D_MAX_REGIMES ? n_regimes : PF2D_MAX_REGIMES;
    pf->uniform_weight = (pf2d_real)1.0 / n_particles;
    
    pf->n_threads = omp_get_max_threads();
    if (pf->n_threads > PF2D_MAX_THREADS) pf->n_threads = PF2D_MAX_THREADS;
    
    /* Allocate state arrays */
    pf->prices       = aligned_alloc_real(n_particles);
    pf->prices_tmp   = aligned_alloc_real(n_particles);
    pf->log_vols     = aligned_alloc_real(n_particles);
    pf->log_vols_tmp = aligned_alloc_real(n_particles);
    pf->weights      = aligned_alloc_real(n_particles);
    pf->log_weights  = aligned_alloc_real(n_particles);
    pf->cumsum       = aligned_alloc_real(n_particles);
    pf->regimes      = aligned_alloc_int(n_particles);
    pf->regimes_tmp  = aligned_alloc_int(n_particles);
    pf->scratch1     = aligned_alloc_real(n_particles);
    pf->scratch2     = aligned_alloc_real(n_particles);
    pf->indices      = aligned_alloc_int(n_particles);  /* Proper int buffer */
    
    if (!pf->prices || !pf->prices_tmp || !pf->log_vols || !pf->log_vols_tmp ||
        !pf->weights || !pf->log_weights || !pf->cumsum ||
        !pf->regimes || !pf->regimes_tmp || !pf->scratch1 || !pf->scratch2 ||
        !pf->indices) {
        pf2d_destroy(pf);
        return NULL;
    }
    
    /* Initialize MKL RNG streams */
    for (int t = 0; t < pf->n_threads; t++) {
        vslNewStream(&pf->mkl_rng[t], VSL_BRNG_SFMT19937, 42 + t * 8192);
    }
    
    /* Initialize PCG streams */
    for (int t = 0; t < pf->n_threads; t++) {
        pf2d_pcg32_seed(&pf->pcg[t], 42 + t * 12345, t * 67890);
    }
    pf->use_pcg = (n_particles < PF2D_BLAS_THRESHOLD) ? 1 : 0;
    
    /* MKL fast math mode */
    vmlSetMode(VML_EP | VML_FTZDAZ_ON | VML_ERRMODE_IGNORE);
    
    /* Default regime parameters */
    /* Regime 0: Trend - low vol, slow MR */
    pf2d_set_regime_params(pf, 0, 0.001, 0.02, log(0.01), 0.05, 0.0);
    /* Regime 1: Mean-revert - stable, small vol */
    pf2d_set_regime_params(pf, 1, 0.0, 0.05, log(0.008), 0.03, 0.0);
    /* Regime 2: High-vol */
    pf2d_set_regime_params(pf, 2, 0.0, 0.10, log(0.03), 0.10, 0.0);
    /* Regime 3: Jump */
    pf2d_set_regime_params(pf, 3, 0.0, 0.20, log(0.05), 0.20, 0.0);
    
    /* Default observation variance */
    pf2d_set_observation_variance(pf, 0.0001);
    
    /* Adaptive resampling defaults */
    pf->resample_threshold = PF2D_RESAMPLE_THRESH_DEFAULT;
    pf->vol_ema = 0.01;
    pf->vol_baseline = 0.01;
    
    /* Uniform initial weights */
    for (int i = 0; i < n_particles; i++) {
        pf->weights[i] = pf->uniform_weight;
    }
    
    /* Initialize regime LUT */
    memset(pf->regime_lut, 0, PF2D_REGIME_LUT_SIZE);
    
    return pf;
}

void pf2d_destroy(PF2D* pf) {
    if (!pf) return;
    
    for (int t = 0; t < pf->n_threads; t++) {
        if (pf->mkl_rng[t]) vslDeleteStream(&pf->mkl_rng[t]);
    }
    
    mkl_free(pf->prices);
    mkl_free(pf->prices_tmp);
    mkl_free(pf->log_vols);
    mkl_free(pf->log_vols_tmp);
    mkl_free(pf->weights);
    mkl_free(pf->log_weights);
    mkl_free(pf->cumsum);
    mkl_free(pf->regimes);
    mkl_free(pf->regimes_tmp);
    mkl_free(pf->scratch1);
    mkl_free(pf->scratch2);
    mkl_free(pf->indices);
    mkl_free(pf);
}

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

void pf2d_set_observation_variance(PF2D* pf, pf2d_real var) {
    pf->obs_variance = var;
    pf->inv_obs_variance = (pf2d_real)1.0 / var;
    pf->neg_half_inv_var = (pf2d_real)-0.5 / var;
}

void pf2d_set_regime_params(PF2D* pf, int r,
                            pf2d_real drift,
                            pf2d_real theta_vol,
                            pf2d_real mu_vol,
                            pf2d_real sigma_vol,
                            pf2d_real rho) {
    if (r < 0 || r >= PF2D_MAX_REGIMES) return;
    
    PF2DRegimeParams* p = &pf->regimes_params[r];
    p->drift = drift;
    p->theta_vol = theta_vol;
    p->mu_vol = mu_vol;
    p->sigma_vol = sigma_vol;
    p->rho = rho;
    
    /* Precompute */
    p->one_minus_theta = (pf2d_real)1.0 - theta_vol;
    p->theta_mu = theta_vol * mu_vol;
    p->sqrt_one_minus_rho_sq = (pf2d_real)sqrt(1.0 - (double)(rho * rho));
}

void pf2d_precompute(PF2D* pf, const PF2DSSAFeatures* ssa) {
    /* Modulate drift by SSA trend */
    for (int r = 0; r < pf->n_regimes; r++) {
        PF2DRegimeParams* p = &pf->regimes_params[r];
        /* Could scale drift by trend, but keep base params intact */
        /* drift_effective = drift * (1 + ssa->trend) - applied in propagate */
        (void)p;
    }
    (void)ssa;
}

void pf2d_set_regime_probs(PF2DRegimeProbs* rp, const pf2d_real* probs, int n) {
    rp->n_regimes = n;
    rp->cumprobs[0] = probs[0];
    rp->probs[0] = probs[0];
    
    for (int i = 1; i < n; i++) {
        rp->probs[i] = probs[i];
        rp->cumprobs[i] = rp->cumprobs[i-1] + probs[i];
    }
    rp->cumprobs[n-1] = (pf2d_real)1.0;
}

void pf2d_build_regime_lut(PF2D* pf, const PF2DRegimeProbs* rp) {
    for (int i = 0; i < PF2D_REGIME_LUT_SIZE; i++) {
        pf2d_real u = (pf2d_real)i / (pf2d_real)PF2D_REGIME_LUT_SIZE;
        int regime = rp->n_regimes - 1;
        for (int r = 0; r < rp->n_regimes - 1; r++) {
            if (u < rp->cumprobs[r]) {
                regime = r;
                break;
            }
        }
        pf->regime_lut[i] = (uint8_t)regime;
    }
}

void pf2d_enable_pcg(PF2D* pf, int enable) {
    pf->use_pcg = enable;
}

void pf2d_set_resample_adaptive(PF2D* pf, pf2d_real baseline_vol) {
    pf->vol_baseline = baseline_vol;
    pf->vol_ema = baseline_vol;
    pf->resample_threshold = PF2D_RESAMPLE_THRESH_DEFAULT;
}

/*============================================================================
 * INITIALIZE
 *============================================================================*/

void pf2d_initialize(PF2D* pf, pf2d_real price0, pf2d_real price_spread,
                     pf2d_real log_vol0, pf2d_real log_vol_spread) {
    int n = pf->n_particles;
    
    if (pf->use_pcg) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            pf2d_pcg32_t* rng = &pf->pcg[tid];
            
            #pragma omp for
            for (int i = 0; i < n; i++) {
                pf->prices[i] = price0 + price_spread * pf2d_pcg32_gaussian(rng);
                pf->log_vols[i] = log_vol0 + log_vol_spread * pf2d_pcg32_gaussian(rng);
            }
        }
    } else {
        pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[0],
                         n, pf->prices, price0, price_spread);
        pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[0],
                         n, pf->log_vols, log_vol0, log_vol_spread);
    }
    
    /* Uniform weights */
    for (int i = 0; i < n; i++) {
        pf->weights[i] = pf->uniform_weight;
        pf->regimes[i] = i % pf->n_regimes;
    }
}

/*============================================================================
 * PROPAGATE
 *============================================================================*/

void pf2d_propagate(PF2D* pf, const PF2DRegimeProbs* rp) {
    int n = pf->n_particles;
    (void)rp;  /* Regime probs baked into LUT */
    
    const uint8_t* lut = pf->regime_lut;
    const PF2DRegimeParams* params = pf->regimes_params;
    
    pf2d_real* prices = pf->prices;
    pf2d_real* log_vols = pf->log_vols;
    int* regs = pf->regimes;
    
    if (pf->use_pcg) {
        /* PCG PATH: Inline RNG generation, per-particle exp()
         * 
         * Trade-off: exp() per particle is slower than vectorized vExp,
         * but inline RNG has better cache locality for small N.
         * 
         * For maximum throughput with N > 2000, disable PCG to use
         * the vectorized path which pre-computes exp(log_vol) via vExp.
         */
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            pf2d_pcg32_t* rng = &pf->pcg[tid];
            
            #pragma omp for
            for (int i = 0; i < n; i++) {
                /* Sample regime - use (SIZE-1) to avoid overflow when u=1.0 */
                pf2d_real u = pf2d_pcg32_uniform(rng);
                int lut_idx = (int)(u * (pf2d_real)(PF2D_REGIME_LUT_SIZE - 1));
                int r = lut[lut_idx];
                regs[i] = r;
                
                const PF2DRegimeParams* p = &params[r];
                
                /* Generate correlated noise */
                pf2d_real z1 = pf2d_pcg32_gaussian(rng);
                pf2d_real z2 = pf2d_pcg32_gaussian(rng);
                
                pf2d_real noise_price = z1;
                pf2d_real noise_vol = p->rho * z1 + p->sqrt_one_minus_rho_sq * z2;
                
                /* Current state */
                pf2d_real price = prices[i];
                pf2d_real lv = log_vols[i];
                
                /* Volatility for price dynamics */
                pf2d_real vol = (pf2d_real)exp((double)lv);
                
                /* Price dynamics: price += drift + vol * noise */
                price = price + p->drift + vol * noise_price;
                
                /* Log-vol dynamics: lv = (1-θ)*lv + θ*μ + σ_v*noise */
                lv = p->one_minus_theta * lv + p->theta_mu + p->sigma_vol * noise_vol;
                
                prices[i] = price;
                log_vols[i] = lv;
            }
        }
    } else {
        /* MKL: bulk RNG then propagate */
        pf2d_real* noise1 = pf->scratch1;
        pf2d_real* noise2 = pf->scratch2;
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            int chunk = (n + nt - 1) / nt;
            int start = tid * chunk;
            int end = start + chunk;
            if (end > n) end = n;
            int len = end - start;
            
            if (len > 0) {
                pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[tid],
                                 len, &noise1[start], 0.0, 1.0);
                pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[tid],
                                 len, &noise2[start], 0.0, 1.0);
            }
        }
        
        /* Need uniform for regime sampling */
        pf2d_real* uniform = pf->log_weights;  /* Reuse buffer temporarily */
        pf2d_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, pf->mkl_rng[0], n, uniform, 0.0, 1.0);
        
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            /* Sample regime - use (SIZE-1) to avoid overflow when u=1.0 */
            int lut_idx = (int)(uniform[i] * (pf2d_real)(PF2D_REGIME_LUT_SIZE - 1));
            int r = lut[lut_idx];
            regs[i] = r;
            
            const PF2DRegimeParams* p = &params[r];
            
            /* Correlated noise */
            pf2d_real z1 = noise1[i];
            pf2d_real z2 = noise2[i];
            pf2d_real noise_price = z1;
            pf2d_real noise_vol = p->rho * z1 + p->sqrt_one_minus_rho_sq * z2;
            
            /* Current state */
            pf2d_real price = prices[i];
            pf2d_real lv = log_vols[i];
            pf2d_real vol = (pf2d_real)exp((double)lv);
            
            /* Dynamics */
            price = price + p->drift + vol * noise_price;
            lv = p->one_minus_theta * lv + p->theta_mu + p->sigma_vol * noise_vol;
            
            prices[i] = price;
            log_vols[i] = lv;
        }
    }
}

/*============================================================================
 * VECTORIZED PATH (for N >= PF2D_BLAS_THRESHOLD)
 * 
 * Optimizations:
 *   - Pre-computed exp(log_vol) via MKL VML (removes transcendental from loop)
 *   - Thread-local RNG streams (NUMA-aware)
 *   - restrict hints for compiler optimization
 *   - Fused log-likelihood + max finding
 *============================================================================*/

/**
 * @brief Vectorized propagation using MKL VML for heavy math
 * 
 * Key optimization: exp(log_vol) computed once via vectorized vExp,
 * then physics loop becomes pure FMA operations.
 */
static void pf2d_propagate_vectorized(PF2D* pf) {
    int n = pf->n_particles;
    
    /* Use restrict to help compiler - no pointer aliasing */
    pf2d_real* restrict prices = pf->prices;
    pf2d_real* restrict log_vols = pf->log_vols;
    pf2d_real* restrict vols = pf->prices_tmp;      /* Reuse as vol buffer */
    pf2d_real* restrict z1 = pf->scratch1;
    pf2d_real* restrict z2 = pf->scratch2;
    pf2d_real* restrict u_reg = pf->log_weights;    /* Reuse temporarily */
    int* restrict regs = pf->regimes;
    
    const uint8_t* restrict lut = pf->regime_lut;
    const PF2DRegimeParams* restrict params = pf->regimes_params;
    
    /* 1. Pre-compute vol = exp(log_vol) using MKL VML
     *    This removes expensive transcendental from physics loop.
     *    MKL vdExp/vsExp uses AVX-512 internally. */
    pf2d_vExp(n, log_vols, vols);
    
    /* 2. Generate RNG in parallel using thread-local streams
     *    Better NUMA behavior than single-stream generation. */
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nt = omp_get_num_threads();
        int chunk = (n + nt - 1) / nt;
        int start = tid * chunk;
        int end = start + chunk;
        if (end > n) end = n;
        int len = end - start;
        
        if (len > 0) {
            pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[tid],
                             len, &z1[start], 0.0, 1.0);
            pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER2, pf->mkl_rng[tid],
                             len, &z2[start], 0.0, 1.0);
            pf2d_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, pf->mkl_rng[tid],
                            len, &u_reg[start], 0.0, 1.0);
        }
    }
    
    /* 3. Physics update - now just FMA, no exp() in loop
     *    Modern CPUs can execute 2 FMA per cycle per core. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        /* Regime lookup - O(1) via LUT, safe bounds */
        int lut_idx = (int)(u_reg[i] * (pf2d_real)(PF2D_REGIME_LUT_SIZE - 1));
        int r = lut[lut_idx];
        regs[i] = r;
        
        /* Load params - small struct fits in L1 cache */
        PF2DRegimeParams p = params[r];
        
        /* Correlated noise construction */
        pf2d_real eps_p = z1[i];
        pf2d_real eps_v = p.rho * z1[i] + p.sqrt_one_minus_rho_sq * z2[i];
        
        /* Price update: P_t = P_{t-1} + drift + vol * eps 
         * Note: vols[i] already computed via vExp above */
        prices[i] += p.drift + vols[i] * eps_p;
        
        /* Log-vol update: v_t = (1-θ)*v_{t-1} + θ*μ + σ*eps */
        log_vols[i] = p.one_minus_theta * log_vols[i] + 
                      p.theta_mu + 
                      p.sigma_vol * eps_v;
    }
}

/**
 * @brief Vectorized weight update with fused log-likelihood + max finding
 */
static void pf2d_update_weights_vectorized(PF2D* pf, pf2d_real observation) {
    int n = pf->n_particles;
    pf2d_real nhiv = pf->neg_half_inv_var;
    
    pf2d_real* restrict prices = pf->prices;
    pf2d_real* restrict lw = pf->log_weights;
    pf2d_real* restrict w = pf->weights;
    
    /* 1. Fused: compute log-likelihoods AND find max in single pass
     *    Reduces memory traffic - read prices once, write lw once. */
    pf2d_real max_lw = (pf2d_real)(-1e30);
    
    #pragma omp parallel for reduction(max: max_lw) schedule(static)
    for (int i = 0; i < n; i++) {
        pf2d_real diff = observation - prices[i];
        pf2d_real val = nhiv * diff * diff;
        lw[i] = val;
        if (val > max_lw) max_lw = val;
    }
    
    /* 2. Subtract max for numerical stability */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        lw[i] -= max_lw;
    }
    
    /* 3. Vectorized exp via MKL VML */
    pf2d_vExp(n, lw, w);
    
    /* 4. Sum via MKL BLAS (highly threaded) */
    pf2d_real sum = pf2d_cblas_asum(n, w, 1);
    
    /* 5. Degenerate weight detection */
    if (sum == (pf2d_real)0.0 || !isfinite(sum)) {
        /* All weights underflowed - reset to uniform */
        pf2d_real uw = pf->uniform_weight;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            w[i] = uw;
        }
        return;
    }
    
    /* 6. Normalize via MKL BLAS */
    pf2d_real inv_sum = (pf2d_real)1.0 / sum;
    pf2d_cblas_scal(n, inv_sum, w, 1);
}

/*============================================================================
 * UPDATE WEIGHTS
 *============================================================================*/

void pf2d_update_weights(PF2D* pf, pf2d_real observation) {
    int n = pf->n_particles;
    pf2d_real* prices = pf->prices;
    pf2d_real* lw = pf->log_weights;
    pf2d_real* w = pf->weights;
    pf2d_real nhiv = pf->neg_half_inv_var;
    
    if (n < PF2D_BLAS_THRESHOLD) {
        /* Manual loops for small N */
        pf2d_real max_lw = (pf2d_real)(-1e30);
        for (int i = 0; i < n; i++) {
            pf2d_real diff = observation - prices[i];
            pf2d_real logw = nhiv * diff * diff;
            lw[i] = logw;
            if (logw > max_lw) max_lw = logw;
        }
        
        for (int i = 0; i < n; i++) {
            lw[i] -= max_lw;
        }
        pf2d_vExp(n, lw, w);
        
        pf2d_real sum = (pf2d_real)0.0;
        for (int i = 0; i < n; i++) {
            sum += w[i];
        }
        
        /* Degenerate weight detection: all weights underflowed to 0 */
        if (sum == (pf2d_real)0.0 || !isfinite(sum)) {
            /* Fallback: reset to uniform weights */
            pf2d_real uw = pf->uniform_weight;
            for (int i = 0; i < n; i++) {
                w[i] = uw;
            }
            return;
        }
        
        pf2d_real inv_sum = (pf2d_real)1.0 / sum;
        for (int i = 0; i < n; i++) {
            w[i] *= inv_sum;
        }
    } else {
        /* BLAS/VML for large N */
        pf2d_real* scratch = pf->scratch1;
        
        for (int i = 0; i < n; i++) {
            scratch[i] = observation - prices[i];
        }
        
        pf2d_vSqr(n, scratch, lw);
        pf2d_cblas_scal(n, nhiv, lw, 1);
        
        /* Find max */
        pf2d_real max_lw = lw[0];
        for (int i = 1; i < n; i++) {
            if (lw[i] > max_lw) max_lw = lw[i];
        }
        
        for (int i = 0; i < n; i++) {
            lw[i] -= max_lw;
        }
        
        pf2d_vExp(n, lw, w);
        
        pf2d_real sum = pf2d_cblas_asum(n, w, 1);
        
        /* Degenerate weight detection */
        if (sum == (pf2d_real)0.0 || !isfinite(sum)) {
            pf2d_real uw = pf->uniform_weight;
            for (int i = 0; i < n; i++) {
                w[i] = uw;
            }
            return;
        }
        
        pf2d_real inv_sum = (pf2d_real)1.0 / sum;
        pf2d_cblas_scal(n, inv_sum, w, 1);
    }
}

/*============================================================================
 * EFFECTIVE SAMPLE SIZE
 *============================================================================*/

pf2d_real pf2d_effective_sample_size(const PF2D* pf) {
    pf2d_real sum_sq = pf2d_cblas_dot(pf->n_particles, pf->weights, 1, pf->weights, 1);
    return (pf2d_real)1.0 / sum_sq;
}

/*============================================================================
 * RESAMPLE
 *============================================================================*/

void pf2d_resample(PF2D* pf) {
    int n = pf->n_particles;
    pf2d_real* w = pf->weights;
    pf2d_real* cs = pf->cumsum;
    
#ifdef PF2D_HIGH_N
    /* HIGH-N PATH: Parallel cumsum */
    parallel_cumsum(cs, w, n);
#else
    /* Standard sequential cumsum */
    cs[0] = w[0];
    for (int i = 1; i < n; i++) {
        cs[i] = cs[i-1] + w[i];
    }
#endif
    
    /* Single random number for systematic resampling */
    pf2d_real u0;
    if (pf->use_pcg) {
        u0 = pf2d_pcg32_uniform(&pf->pcg[0]);
    } else {
        pf2d_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, pf->mkl_rng[0], 1, &u0, 0.0, 1.0);
    }
    u0 *= pf->uniform_weight;
    
    pf2d_real inv_n = pf->uniform_weight;
    
#ifdef PF2D_HIGH_N
    if (n >= PF2D_PARALLEL_SEARCH_THRESH) {
        /* HIGH-N PATH: Parallel search */
        
        /* Generate search sites in parallel */
        pf2d_real* search_sites = pf->scratch1;
        parallel_stratified_uniform(search_sites, u0, inv_n, n);
        
        /* Batch binary search in parallel - use proper int buffer */
        int* indices = pf->indices;
        parallel_batch_search(cs, search_sites, indices, n);
        
        /* Parallel gather */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            pf->prices_tmp[i] = pf->prices[idx];
            pf->log_vols_tmp[i] = pf->log_vols[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    } else {
        /* Medium N: sequential search with parallel gather */
        int* indices = pf->indices;
        
        /* Sequential systematic resampling (low overhead for N < 16k) */
        int idx = 0;
        for (int i = 0; i < n; i++) {
            pf2d_real u = u0 + i * inv_n;
            while (cs[idx] < u && idx < n - 1) idx++;
            indices[i] = idx;
        }
        
        /* Parallel gather */
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            int idx = indices[i];
            pf->prices_tmp[i] = pf->prices[idx];
            pf->log_vols_tmp[i] = pf->log_vols[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    }
#else
    /* STANDARD PATH: Sequential resampling */
    
    /* Check for skewed weights */
    pf2d_real ess = pf2d_effective_sample_size(pf);
    int use_binary = (ess < n * 0.1);
    
    if (use_binary) {
        for (int i = 0; i < n; i++) {
            pf2d_real u = u0 + i * inv_n;
            int idx = binary_search_cumsum(cs, n, u);
            pf->prices_tmp[i] = pf->prices[idx];
            pf->log_vols_tmp[i] = pf->log_vols[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    } else {
        int idx = 0;
        for (int i = 0; i < n; i++) {
            pf2d_real u = u0 + i * inv_n;
            while (cs[idx] < u && idx < n - 1) idx++;
            pf->prices_tmp[i] = pf->prices[idx];
            pf->log_vols_tmp[i] = pf->log_vols[idx];
            pf->regimes_tmp[i] = pf->regimes[idx];
        }
    }
#endif
    
    /* Swap pointers */
    pf2d_real* tmp;
    tmp = pf->prices; pf->prices = pf->prices_tmp; pf->prices_tmp = tmp;
    tmp = pf->log_vols; pf->log_vols = pf->log_vols_tmp; pf->log_vols_tmp = tmp;
    
    int* tmp_i = pf->regimes;
    pf->regimes = pf->regimes_tmp;
    pf->regimes_tmp = tmp_i;
    
    /* Reset weights */
    for (int i = 0; i < n; i++) {
        w[i] = inv_n;
    }
}

static inline void pf2d_update_resample_threshold(PF2D* pf, pf2d_real current_vol) {
    pf2d_real alpha = (pf2d_real)0.05;
    pf->vol_ema = alpha * current_vol + ((pf2d_real)1.0 - alpha) * pf->vol_ema;
    
    pf2d_real vol_ratio = pf->vol_ema / (pf->vol_baseline + (pf2d_real)1e-10);
    
    if (vol_ratio > (pf2d_real)2.0) {
        pf->resample_threshold = PF2D_RESAMPLE_THRESH_MIN;
    } else if (vol_ratio < (pf2d_real)0.5) {
        pf->resample_threshold = PF2D_RESAMPLE_THRESH_MAX;
    } else {
        pf2d_real t = (vol_ratio - (pf2d_real)0.5) / (pf2d_real)1.5;
        pf->resample_threshold = PF2D_RESAMPLE_THRESH_MAX - 
                                  t * (PF2D_RESAMPLE_THRESH_MAX - PF2D_RESAMPLE_THRESH_MIN);
    }
}

int pf2d_resample_if_needed(PF2D* pf) {
    pf2d_real ess = pf2d_effective_sample_size(pf);
    if (ess < pf->n_particles * pf->resample_threshold) {
        pf2d_resample(pf);
        return 1;
    }
    return 0;
}

/*============================================================================
 * ESTIMATES
 *============================================================================*/

pf2d_real pf2d_price_mean(const PF2D* pf) {
    return pf2d_cblas_dot(pf->n_particles, pf->weights, 1, pf->prices, 1);
}

pf2d_real pf2d_price_variance(const PF2D* pf) {
    int n = pf->n_particles;
    pf2d_real m = pf2d_price_mean(pf);
    pf2d_real sum = (pf2d_real)0.0;
    
    for (int i = 0; i < n; i++) {
        pf2d_real diff = pf->prices[i] - m;
        sum += pf->weights[i] * diff * diff;
    }
    return sum;
}

pf2d_real pf2d_log_vol_mean(const PF2D* pf) {
    return pf2d_cblas_dot(pf->n_particles, pf->weights, 1, pf->log_vols, 1);
}

pf2d_real pf2d_log_vol_variance(const PF2D* pf) {
    int n = pf->n_particles;
    pf2d_real m = pf2d_log_vol_mean(pf);
    pf2d_real sum = (pf2d_real)0.0;
    
    for (int i = 0; i < n; i++) {
        pf2d_real diff = pf->log_vols[i] - m;
        sum += pf->weights[i] * diff * diff;
    }
    return sum;
}

pf2d_real pf2d_vol_mean(const PF2D* pf) {
    /* E[exp(log_vol)] using log-normal formula:
     * If log_vol ~ N(μ, σ²), then E[exp(log_vol)] = exp(μ + σ²/2) */
    pf2d_real mu = pf2d_log_vol_mean(pf);
    pf2d_real var = pf2d_log_vol_variance(pf);
    return (pf2d_real)exp((double)(mu + var * (pf2d_real)0.5));
}

/*============================================================================
 * FULL UPDATE
 *============================================================================*/

PF2DOutput pf2d_update(PF2D* pf, pf2d_real observation, const PF2DRegimeProbs* rp) {
    PF2DOutput out;
    int n = pf->n_particles;
    
    /* Select implementation based on particle count and RNG mode
     * 
     * Path selection:
     *   - PCG + small N:  Inline RNG, per-particle exp (lowest latency)
     *   - MKL + small N:  Thread-local RNG, per-particle exp
     *   - Large N:        Vectorized path with pre-computed exp (highest throughput)
     */
    int use_vectorized = (n >= PF2D_BLAS_THRESHOLD) && (!pf->use_pcg);
    
    /* 1. Propagate */
    if (use_vectorized) {
        pf2d_propagate_vectorized(pf);
    } else {
        pf2d_propagate(pf, rp);
    }
    
    /* 2. Update weights */
    if (use_vectorized) {
        pf2d_update_weights_vectorized(pf, observation);
    } else {
        pf2d_update_weights(pf, observation);
    }
    
    /* 3. Compute estimates */
    out.price_mean = pf2d_price_mean(pf);
    out.price_variance = pf2d_price_variance(pf);
    out.log_vol_mean = pf2d_log_vol_mean(pf);
    out.log_vol_variance = pf2d_log_vol_variance(pf);
    out.vol_mean = pf2d_vol_mean(pf);
    out.ess = pf2d_effective_sample_size(pf);
    
    /* Regime distribution */
    for (int r = 0; r < PF2D_MAX_REGIMES; r++) out.regime_probs[r] = 0;
    pf2d_real max_prob = 0;
    out.dominant_regime = 0;
    for (int i = 0; i < n; i++) {
        out.regime_probs[pf->regimes[i]] += pf->weights[i];
    }
    for (int r = 0; r < pf->n_regimes; r++) {
        if (out.regime_probs[r] > max_prob) {
            max_prob = out.regime_probs[r];
            out.dominant_regime = r;
        }
    }
    
    /* 4. Update adaptive threshold */
    pf2d_update_resample_threshold(pf, out.vol_mean);
    
    /* 5. Resample if needed */
    out.resampled = pf2d_resample_if_needed(pf);
    
    return out;
}

/*============================================================================
 * DEBUG
 *============================================================================*/

void pf2d_print_config(const PF2D* pf) {
    printf("2D Particle Filter Configuration:\n");
    printf("  Precision:      %s (%d bytes)\n",
           PF2D_REAL_SIZE == 4 ? "float" : "double", PF2D_REAL_SIZE);
    printf("  Particles:      %d\n", pf->n_particles);
    printf("  Regimes:        %d\n", pf->n_regimes);
    printf("  Obs var:        %g\n", (double)pf->obs_variance);
    printf("  OMP threads:    %d\n", pf->n_threads);
    printf("  RNG:            %s\n", pf->use_pcg ? "PCG (fast)" : "MKL SFMT");
    
    /* Show which execution path will be used */
    int use_vec = (pf->n_particles >= PF2D_BLAS_THRESHOLD) && (!pf->use_pcg);
    if (pf->use_pcg) {
        printf("  Exec path:      PCG inline (lowest latency)\n");
    } else if (use_vec) {
        printf("  Exec path:      Vectorized (pre-computed vExp, fused loops)\n");
    } else {
        printf("  Exec path:      MKL thread-local RNG\n");
    }
    
#ifdef PF2D_HIGH_N
    printf("  HIGH_N mode:    ENABLED\n");
    printf("    Parallel cumsum thresh:  %d\n", PF2D_PARALLEL_CUMSUM_THRESH);
    printf("    Parallel search thresh:  %d\n", PF2D_PARALLEL_SEARCH_THRESH);
#else
    printf("  HIGH_N mode:    disabled (use -DPF2D_HIGH_N for N > 10k)\n");
#endif
    printf("  Resample thresh: %.2f (adaptive: %.2f-%.2f)\n",
           (double)pf->resample_threshold,
           (double)PF2D_RESAMPLE_THRESH_MIN, (double)PF2D_RESAMPLE_THRESH_MAX);
    
    printf("\n  Per-regime parameters:\n");
    printf("  %-8s %8s %8s %8s %8s %8s\n", 
           "Regime", "drift", "θ_vol", "μ_vol", "σ_vol", "ρ");
    for (int r = 0; r < pf->n_regimes; r++) {
        const PF2DRegimeParams* p = &pf->regimes_params[r];
        printf("  %-8d %8.4f %8.4f %8.4f %8.4f %8.4f\n",
               r, (double)p->drift, (double)p->theta_vol,
               (double)p->mu_vol, (double)p->sigma_vol, (double)p->rho);
    }
}
