/**
 * @file pmmh_mkl.h
 * @brief High-performance PMMH using Intel MKL
 *
 * Optimizations:
 *   - Arena allocation: Single contiguous block for particle arrays (better dTLB)
 *   - VSL SFMT19937 RNG with ICDF method (faster than BoxMuller)
 *   - VML vdExp for vectorized exp (standard precision)
 *   - OpenMP SIMD for particle loops
 *   - Reciprocal multiplication instead of division
 *   - Precomputed CDF for resampling with forced endpoint
 *   - Thread-local state reuse (avoids vslNewStream overhead)
 *   - Restrict pointers for aliasing safety
 *   - Single-pass mean/variance computation
 *
 * Threading model:
 *   - MKL internal: single-threaded (vectors too small to benefit)
 *   - Chain level: OpenMP parallel with thread-local states
 *
 * Memory layout:
 *   - All particle arrays in single 64-byte aligned arena
 *   - log_vol, log_vol_new, weights, weights_exp, noise, uniform, ancestors
 *
 * Compile: gcc -O3 -march=native -fopenmp -I${MKLROOT}/include \
 *          pmmh_mkl.c -L${MKLROOT}/lib/intel64 -lmkl_intel_lp64 \
 *          -lmkl_gnu_thread -lmkl_core -lgomp -lm
 */

#ifndef PMMH_MKL_H
#define PMMH_MKL_H

#include <mkl.h>
#include <mkl_vsl.h>
#include <mkl_vml.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define PMMH_N_PARTICLES 256
#define PMMH_N_ITERATIONS 400
#define PMMH_N_BURNIN 150
#define PMMH_CACHE_LINE 64

/* Align allocations for SIMD */
#define ALIGNED_ALLOC(size) mkl_malloc((size), PMMH_CACHE_LINE)
#define ALIGNED_FREE(ptr) mkl_free(ptr)

/*============================================================================
 * DATA STRUCTURES
 *============================================================================*/

typedef struct
{
    double drift;
    double mu_vol;
    double sigma_vol;
} PMMHParams;

typedef struct
{
    PMMHParams mean;
    PMMHParams std;
} PMMHPrior;

typedef struct
{
    PMMHParams posterior_mean;
    PMMHParams posterior_std;
    double acceptance_rate;
    int n_samples;
    double elapsed_ms;
} PMMHResult;

typedef struct
{
    /* Particle state - aligned for SIMD */
    double *log_vol;     /* [n_particles] current log-volatility */
    double *log_vol_new; /* [n_particles] proposed log-volatility */
    double *weights;     /* [n_particles] log-weights */
    double *weights_exp; /* [n_particles] exp(weights) for resampling */
    double *noise;       /* [n_particles] Gaussian noise buffer */
    double *uniform;     /* [n_particles] uniform noise for resampling */
    int *ancestors;      /* [n_particles] resampling indices */

    /* VSL streams - one per thread for parallel RNG */
    VSLStreamStatePtr *streams;
    int n_threads;
    int n_particles;

    /* Model parameters */
    double theta_vol; /* Mean reversion speed */

} PMMHState;

/*============================================================================
 * INITIALIZATION / CLEANUP
 *============================================================================*/

static PMMHState *pmmh_state_create(int n_particles, double theta_vol)
{
    PMMHState *s = (PMMHState *)ALIGNED_ALLOC(sizeof(PMMHState));
    memset(s, 0, sizeof(PMMHState));

    s->n_particles = n_particles;
    s->theta_vol = theta_vol;

    /* Arena allocation: single contiguous block for all particle arrays
     * Benefits: Better dTLB hit rates, prefetching, reduced heap fragmentation */
    size_t d_size = ((n_particles * sizeof(double) + PMMH_CACHE_LINE - 1) / PMMH_CACHE_LINE) * PMMH_CACHE_LINE; /* Align each array */
    size_t i_size = ((n_particles * sizeof(int) + PMMH_CACHE_LINE - 1) / PMMH_CACHE_LINE) * PMMH_CACHE_LINE;

    /* 6 double arrays + 1 int array */
    size_t total_size = 6 * d_size + i_size;

    char *arena = (char *)mkl_malloc(total_size, PMMH_CACHE_LINE);
    s->log_vol = (double *)(arena + 0 * d_size);
    s->log_vol_new = (double *)(arena + 1 * d_size);
    s->weights = (double *)(arena + 2 * d_size);
    s->weights_exp = (double *)(arena + 3 * d_size);
    s->noise = (double *)(arena + 4 * d_size);
    s->uniform = (double *)(arena + 5 * d_size);
    s->ancestors = (int *)(arena + 6 * d_size);

    /* Store arena base for cleanup */
    s->n_threads = omp_get_max_threads();

    /* Create VSL streams - one per thread */
    s->streams = (VSLStreamStatePtr *)malloc(s->n_threads * sizeof(VSLStreamStatePtr));

    for (int i = 0; i < s->n_threads; i++)
    {
        /* Use SFMT19937 - SIMD-optimized Mersenne Twister, fastest for bulk */
        vslNewStream(&s->streams[i], VSL_BRNG_SFMT19937, 12345 + i * 7919);
    }

    return s;
}

static void pmmh_state_destroy(PMMHState *s)
{
    if (!s)
        return;

    for (int i = 0; i < s->n_threads; i++)
    {
        vslDeleteStream(&s->streams[i]);
    }
    free(s->streams);

    /* Single free for arena (log_vol is base pointer) */
    mkl_free(s->log_vol);

    ALIGNED_FREE(s);
}

static void pmmh_state_seed(PMMHState *s, unsigned int seed)
{
    for (int i = 0; i < s->n_threads; i++)
    {
        vslDeleteStream(&s->streams[i]);
        vslNewStream(&s->streams[i], VSL_BRNG_SFMT19937, seed + i * 7919);
    }
}

/*============================================================================
 * VECTORIZED PARTICLE FILTER LOG-LIKELIHOOD
 *
 * Optimizations applied:
 *   1. SFMT19937 RNG with ICDF method
 *   2. vdExp for vectorized exp
 *   3. Reciprocal multiplication instead of division
 *   4. SIMD loops with #pragma omp simd
 *   5. Precomputed CDF for resampling (forced to 1.0 endpoint)
 *   6. Restrict pointers for aliasing safety
 *============================================================================*/

static double pmmh_log_likelihood_mkl(PMMHState *s,
                                      const double *returns, int n_obs,
                                      const PMMHParams *params)
{
    const int np = s->n_particles;
    const double theta = s->theta_vol;
    const double one_minus_theta = 1.0 - theta;
    const double theta_mu = theta * params->mu_vol;
    const double drift = params->drift;
    const double sigma_vol = params->sigma_vol;

    int tid = omp_get_thread_num();

    /* Initialize particles at mu_vol with some spread */
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->streams[tid],
                  np, s->log_vol, params->mu_vol, 0.3);

    double total_log_lik = 0.0;

    for (int t = 0; t < n_obs; t++)
    {
        const double ret = returns[t];
        const double ret_minus_drift = ret - drift;

        /* Generate noise for volatility evolution */
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->streams[tid],
                      np, s->noise, 0.0, sigma_vol);

/* Propagate volatility (SIMD) */
#pragma omp simd
        for (int i = 0; i < np; i++)
        {
            s->log_vol_new[i] = one_minus_theta * s->log_vol[i] + theta_mu + s->noise[i];
        }

        /* Vectorized exp for volatility */
        vdExp(np, s->log_vol_new, s->weights_exp);

        /* Compute log-weights using reciprocal (avoid division) */
        double max_log_w = -DBL_MAX;
#pragma omp simd reduction(max : max_log_w)
        for (int i = 0; i < np; i++)
        {
            double vol = s->weights_exp[i];
            double inv_vol = 1.0 / vol; /* Reciprocal - faster than division in loop */
            double z = ret_minus_drift * inv_vol;
            double log_w = -0.5 * z * z - s->log_vol_new[i];
            s->weights[i] = log_w;
            if (log_w > max_log_w)
                max_log_w = log_w;
        }

/* Normalize weights (subtract max for stability) */
#pragma omp simd
        for (int i = 0; i < np; i++)
        {
            s->weights[i] -= max_log_w;
        }

        /* Vectorized exp for normalized weights */
        vdExp(np, s->weights, s->weights_exp);

        /* Sum weights */
        double sum_w = 0.0;
#pragma omp simd reduction(+ : sum_w)
        for (int i = 0; i < np; i++)
        {
            sum_w += s->weights_exp[i];
        }

        total_log_lik += max_log_w + log(sum_w / np);

        /* Normalize weights */
        double inv_sum = 1.0 / sum_w;
#pragma omp simd
        for (int i = 0; i < np; i++)
        {
            s->weights_exp[i] *= inv_sum;
        }

        /* Always resample (adaptive resampling was causing accuracy issues) */
        {
            /* Build CDF in-place (reuse uniform buffer) */
            double *cdf = s->uniform;
            cdf[0] = s->weights_exp[0];
            for (int i = 1; i < np; i++)
            {
                cdf[i] = cdf[i - 1] + s->weights_exp[i];
            }
            /* Force CDF to end at exactly 1.0 (avoid float accumulation errors) */
            cdf[np - 1] = 1.0;

            /* Generate single uniform */
            double u0;
            vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s->streams[tid],
                         1, &u0, 0.0, 1.0);
            double inv_np = 1.0 / np;
            u0 *= inv_np;

            /* Systematic resampling with linear scan
             * Since targets are sorted and CDF is monotonic, j only increases */
            int j = 0;
            for (int i = 0; i < np; i++)
            {
                double target = u0 + i * inv_np;
                /* Advance j until CDF[j] >= target */
                while (j < np - 1 && cdf[j] < target)
                {
                    j++;
                }
                s->ancestors[i] = j;
            }

            /* Gather resampled particles (use restrict to prevent aliasing) */
            double *restrict tmp = s->noise;
            const double *restrict src = s->log_vol_new;
            const int *restrict idx = s->ancestors;

            for (int i = 0; i < np; i++)
            {
                tmp[i] = src[idx[i]];
            }

            double *restrict dst = s->log_vol;
#pragma omp simd
            for (int i = 0; i < np; i++)
            {
                dst[i] = tmp[i];
            }
        }
    }

    return total_log_lik;
}

/*============================================================================
 * LOG-PRIOR
 *============================================================================*/

static double pmmh_log_prior(const PMMHParams *p, const PMMHPrior *prior)
{
    double lp = 0.0;

    /* Gaussian priors */
    double z_drift = (p->drift - prior->mean.drift) / prior->std.drift;
    double z_mu = (p->mu_vol - prior->mean.mu_vol) / prior->std.mu_vol;
    double z_sigma = (log(p->sigma_vol) - log(prior->mean.sigma_vol)) / prior->std.sigma_vol;

    lp -= 0.5 * (z_drift * z_drift + z_mu * z_mu + z_sigma * z_sigma);
    lp -= log(prior->std.drift) + log(prior->std.mu_vol) + log(prior->std.sigma_vol);
    lp -= log(p->sigma_vol); /* Jacobian for log-normal */

    return lp;
}

/*============================================================================
 * ADAPTIVE PROPOSAL TUNING
 *
 * Target acceptance rate: 23-25% for 3 parameters (Roberts & Rosenthal)
 * Adapt during burnin only to maintain ergodicity
 *============================================================================*/

#define PMMH_ADAPT_WINDOW 50    /* Samples before adapting */
#define PMMH_TARGET_ACCEPT 0.30 /* Target 30% - works better with noisy likelihood */
#define PMMH_ACCEPT_TOL 0.10    /* Wider tolerance band */

/* Initial proposal scales - larger = lower acceptance rate */
#define PMMH_INIT_DRIFT_STD 0.001
#define PMMH_INIT_MU_STD 0.15
#define PMMH_INIT_SIGMA_LOG_STD 0.10

typedef struct
{
    double drift_std;
    double mu_std;
    double sigma_log_std;
    int window_accepts;
    int window_total;
} AdaptiveProposal;

static inline void adapt_proposal(AdaptiveProposal *ap)
{
    if (ap->window_total < PMMH_ADAPT_WINDOW)
        return;

    double rate = (double)ap->window_accepts / ap->window_total;

    /* Scale factor: decrease if accepting too little, increase if too much */
    double factor;
    if (rate < PMMH_TARGET_ACCEPT - PMMH_ACCEPT_TOL)
    {
        factor = 0.8; /* Accepting too little → smaller proposals */
    }
    else if (rate > PMMH_TARGET_ACCEPT + PMMH_ACCEPT_TOL)
    {
        factor = 1.25; /* Accepting too much → larger proposals */
    }
    else
    {
        factor = 1.0; /* In target range */
    }

    ap->drift_std *= factor;
    ap->mu_std *= factor;
    ap->sigma_log_std *= factor;

    /* Clamp to reasonable bounds */
    if (ap->drift_std < 1e-6)
        ap->drift_std = 1e-6;
    if (ap->drift_std > 0.01)
        ap->drift_std = 0.01;
    if (ap->mu_std < 0.01)
        ap->mu_std = 0.01;
    if (ap->mu_std > 1.0)
        ap->mu_std = 1.0;
    if (ap->sigma_log_std < 0.01)
        ap->sigma_log_std = 0.01;
    if (ap->sigma_log_std > 0.5)
        ap->sigma_log_std = 0.5;

    /* Reset window */
    ap->window_accepts = 0;
    ap->window_total = 0;
}

/*============================================================================
 * MAIN PMMH SAMPLER (with CPM + Adaptive Proposals)
 *============================================================================*/

static void pmmh_run_mkl(const double *returns, int n_obs,
                         const PMMHPrior *prior,
                         double theta_vol,
                         int n_iterations, int n_burnin,
                         int n_particles,
                         unsigned int seed,
                         PMMHResult *result)
{

    double t_start = omp_get_wtime();

    /* Create state with specific seed */
    PMMHState *state = pmmh_state_create(n_particles, theta_vol);
    pmmh_state_seed(state, seed);

    int tid = omp_get_thread_num();

    /* Adaptive proposal - initialized with reasonable defaults */
    AdaptiveProposal ap = {
        .drift_std = PMMH_INIT_DRIFT_STD,
        .mu_std = PMMH_INIT_MU_STD,
        .sigma_log_std = PMMH_INIT_SIGMA_LOG_STD,
        .window_accepts = 0,
        .window_total = 0};

    /* Sample storage */
    int max_samples = n_iterations - n_burnin;
    double *samples_drift = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_mu = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_sigma = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));

    int n_accepted = 0;
    int sample_idx = 0;

    /* Initialize chain at prior mean */
    PMMHParams current = prior->mean;
    double current_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &current);
    double current_lp = pmmh_log_prior(&current, prior);

    /* Proposal noise buffer */
    double prop_noise[3];

    for (int iter = 0; iter < n_iterations; iter++)
    {
        /* Generate proposal noise */
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->streams[tid],
                      3, prop_noise, 0.0, 1.0);

        /* Propose new parameters using adaptive scales */
        PMMHParams proposed;
        proposed.drift = current.drift + prop_noise[0] * ap.drift_std;
        proposed.mu_vol = current.mu_vol + prop_noise[1] * ap.mu_std;
        proposed.sigma_vol = current.sigma_vol * exp(prop_noise[2] * ap.sigma_log_std);

        /* Clamp to bounds */
        if (proposed.drift < -0.01)
            proposed.drift = -0.01;
        if (proposed.drift > 0.01)
            proposed.drift = 0.01;
        if (proposed.mu_vol < -8.0)
            proposed.mu_vol = -8.0;
        if (proposed.mu_vol > 0.0)
            proposed.mu_vol = 0.0;
        if (proposed.sigma_vol < 0.01)
            proposed.sigma_vol = 0.01;
        if (proposed.sigma_vol > 0.5)
            proposed.sigma_vol = 0.5;

        /* Compute acceptance ratio */
        double prop_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &proposed);
        double prop_lp = pmmh_log_prior(&proposed, prior);

        double log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);

        /* MH accept/reject */
        double u;
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->streams[tid], 1, &u, 0.0, 1.0);

        int accepted = (log(u) < log_alpha);
        if (accepted)
        {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }

        /* Track acceptance for adaptation */
        ap.window_accepts += accepted;
        ap.window_total++;

        /* Adapt proposal during burnin only */
        if (iter < n_burnin)
        {
            adapt_proposal(&ap);
        }

        /* Store sample after burnin */
        if (iter >= n_burnin)
        {
            samples_drift[sample_idx] = current.drift;
            samples_mu[sample_idx] = current.mu_vol;
            samples_sigma[sample_idx] = current.sigma_vol;
            sample_idx++;
        }
    }

    /* Compute posterior statistics using SIMD reductions */
    result->n_samples = sample_idx;

    /* Mean and sum of squares in single pass */
    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    double sum_sq_drift = 0, sum_sq_mu = 0, sum_sq_sigma = 0;

#pragma omp simd reduction(+ : sum_drift, sum_mu, sum_sigma, sum_sq_drift, sum_sq_mu, sum_sq_sigma)
    for (int i = 0; i < sample_idx; i++)
    {
        double d = samples_drift[i];
        double m = samples_mu[i];
        double s = samples_sigma[i];
        sum_drift += d;
        sum_mu += m;
        sum_sigma += s;
        sum_sq_drift += d * d;
        sum_sq_mu += m * m;
        sum_sq_sigma += s * s;
    }

    double inv_n = 1.0 / sample_idx;
    result->posterior_mean.drift = sum_drift * inv_n;
    result->posterior_mean.mu_vol = sum_mu * inv_n;
    result->posterior_mean.sigma_vol = sum_sigma * inv_n;

    /* Std = sqrt(E[x²] - E[x]²) */
    result->posterior_std.drift = sqrt(sum_sq_drift * inv_n -
                                       result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq_mu * inv_n -
                                        result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq_sigma * inv_n -
                                           result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol);

    result->acceptance_rate = (double)n_accepted / n_iterations;
    result->elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;

    /* Cleanup */
    ALIGNED_FREE(samples_drift);
    ALIGNED_FREE(samples_mu);
    ALIGNED_FREE(samples_sigma);
    pmmh_state_destroy(state);
}

/*============================================================================
 * PMMH WITH EXTERNALLY MANAGED STATE
 *
 * For parallel chains: reuse states to avoid vslNewStream overhead
 *============================================================================*/

static void pmmh_run_mkl_with_state(PMMHState *state,
                                    const double *returns, int n_obs,
                                    const PMMHPrior *prior,
                                    double theta_vol,
                                    int n_iterations, int n_burnin,
                                    PMMHResult *result)
{

    double t_start = omp_get_wtime();
    const int n_particles = state->n_particles;
    (void)n_particles; /* Silence unused warning */
    (void)theta_vol;

    int tid = omp_get_thread_num();

    /* Adaptive proposal - initialized with reasonable defaults */
    AdaptiveProposal ap = {
        .drift_std = PMMH_INIT_DRIFT_STD,
        .mu_std = PMMH_INIT_MU_STD,
        .sigma_log_std = PMMH_INIT_SIGMA_LOG_STD,
        .window_accepts = 0,
        .window_total = 0};

    /* Sample storage */
    int max_samples = n_iterations - n_burnin;
    double *samples_drift = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_mu = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_sigma = (double *)ALIGNED_ALLOC(max_samples * sizeof(double));

    int n_accepted = 0;
    int sample_idx = 0;

    /* Initialize chain at prior mean */
    PMMHParams current = prior->mean;
    double current_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &current);
    double current_lp = pmmh_log_prior(&current, prior);

    /* Proposal noise buffer */
    double prop_noise[3];

    for (int iter = 0; iter < n_iterations; iter++)
    {
        /* Generate proposal noise */
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->streams[tid],
                      3, prop_noise, 0.0, 1.0);

        /* Propose new parameters using adaptive scales */
        PMMHParams proposed;
        proposed.drift = current.drift + prop_noise[0] * ap.drift_std;
        proposed.mu_vol = current.mu_vol + prop_noise[1] * ap.mu_std;
        proposed.sigma_vol = current.sigma_vol * exp(prop_noise[2] * ap.sigma_log_std);

        /* Clamp to bounds */
        if (proposed.drift < -0.01)
            proposed.drift = -0.01;
        if (proposed.drift > 0.01)
            proposed.drift = 0.01;
        if (proposed.mu_vol < -8.0)
            proposed.mu_vol = -8.0;
        if (proposed.mu_vol > 0.0)
            proposed.mu_vol = 0.0;
        if (proposed.sigma_vol < 0.01)
            proposed.sigma_vol = 0.01;
        if (proposed.sigma_vol > 0.5)
            proposed.sigma_vol = 0.5;

        /* Compute acceptance ratio */
        double prop_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &proposed);
        double prop_lp = pmmh_log_prior(&proposed, prior);

        double log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);

        /* MH accept/reject */
        double u;
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->streams[tid], 1, &u, 0.0, 1.0);

        int accepted = (log(u) < log_alpha);
        if (accepted)
        {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }

        /* Track acceptance for adaptation */
        ap.window_accepts += accepted;
        ap.window_total++;

        /* Adapt proposal during burnin only */
        if (iter < n_burnin)
        {
            adapt_proposal(&ap);
        }

        /* Store sample after burnin */
        if (iter >= n_burnin)
        {
            samples_drift[sample_idx] = current.drift;
            samples_mu[sample_idx] = current.mu_vol;
            samples_sigma[sample_idx] = current.sigma_vol;
            sample_idx++;
        }
    }

    /* Compute posterior statistics using SIMD reductions */
    result->n_samples = sample_idx;

    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    double sum_sq_drift = 0, sum_sq_mu = 0, sum_sq_sigma = 0;

#pragma omp simd reduction(+ : sum_drift, sum_mu, sum_sigma, sum_sq_drift, sum_sq_mu, sum_sq_sigma)
    for (int i = 0; i < sample_idx; i++)
    {
        double d = samples_drift[i];
        double m = samples_mu[i];
        double s = samples_sigma[i];
        sum_drift += d;
        sum_mu += m;
        sum_sigma += s;
        sum_sq_drift += d * d;
        sum_sq_mu += m * m;
        sum_sq_sigma += s * s;
    }

    double inv_n = 1.0 / sample_idx;
    result->posterior_mean.drift = sum_drift * inv_n;
    result->posterior_mean.mu_vol = sum_mu * inv_n;
    result->posterior_mean.sigma_vol = sum_sigma * inv_n;

    result->posterior_std.drift = sqrt(sum_sq_drift * inv_n -
                                       result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq_mu * inv_n -
                                        result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq_sigma * inv_n -
                                           result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol);

    result->acceptance_rate = (double)n_accepted / n_iterations;
    result->elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;

    /* Cleanup sample storage only (state is external) */
    ALIGNED_FREE(samples_drift);
    ALIGNED_FREE(samples_mu);
    ALIGNED_FREE(samples_sigma);
}

/*============================================================================
 * PARALLEL MONTE CARLO - RUN MULTIPLE CHAINS
 *
 * Optimization: Pre-allocate states per-thread to avoid vslNewStream overhead
 *============================================================================*/

typedef struct
{
    PMMHResult *results; /* [n_chains] */
    int n_chains;
    double total_elapsed_ms;
} PMMHParallelResult;

static void pmmh_run_parallel(const double *returns, int n_obs,
                              const PMMHPrior *prior,
                              double theta_vol,
                              int n_iterations, int n_burnin,
                              int n_particles,
                              int n_chains,
                              PMMHParallelResult *result)
{

    double t_start = omp_get_wtime();

    result->n_chains = n_chains;
    result->results = (PMMHResult *)malloc(n_chains * sizeof(PMMHResult));

    /* Pre-allocate one state per thread to avoid vslNewStream overhead */
    int n_threads = omp_get_max_threads();
    PMMHState **thread_states = (PMMHState **)malloc(n_threads * sizeof(PMMHState *));

    for (int t = 0; t < n_threads; t++)
    {
        thread_states[t] = pmmh_state_create(n_particles, theta_vol);
    }

/* Run chains in parallel, reusing thread-local states */
#pragma omp parallel for schedule(dynamic)
    for (int chain = 0; chain < n_chains; chain++)
    {
        int tid = omp_get_thread_num();
        PMMHState *state = thread_states[tid];

        /* Seed state for this specific chain */
        pmmh_state_seed(state, 12345 + chain * 104729);

        /* Run PMMH with pre-allocated state */
        pmmh_run_mkl_with_state(state, returns, n_obs, prior, theta_vol,
                                n_iterations, n_burnin, &result->results[chain]);
    }

    /* Cleanup thread states */
    for (int t = 0; t < n_threads; t++)
    {
        pmmh_state_destroy(thread_states[t]);
    }
    free(thread_states);

    result->total_elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;
}

static void pmmh_parallel_aggregate(const PMMHParallelResult *pr, PMMHResult *agg)
{
    /* Aggregate results across chains */
    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    double sum_acc = 0;
    int total_samples = 0;

    for (int i = 0; i < pr->n_chains; i++)
    {
        sum_drift += pr->results[i].posterior_mean.drift * pr->results[i].n_samples;
        sum_mu += pr->results[i].posterior_mean.mu_vol * pr->results[i].n_samples;
        sum_sigma += pr->results[i].posterior_mean.sigma_vol * pr->results[i].n_samples;
        sum_acc += pr->results[i].acceptance_rate;
        total_samples += pr->results[i].n_samples;
    }

    agg->posterior_mean.drift = sum_drift / total_samples;
    agg->posterior_mean.mu_vol = sum_mu / total_samples;
    agg->posterior_mean.sigma_vol = sum_sigma / total_samples;
    agg->acceptance_rate = sum_acc / pr->n_chains;
    agg->n_samples = total_samples;
    agg->elapsed_ms = pr->total_elapsed_ms;

    /* Compute cross-chain std (simplified) */
    double var_drift = 0, var_mu = 0, var_sigma = 0;
    for (int i = 0; i < pr->n_chains; i++)
    {
        double d = pr->results[i].posterior_mean.drift - agg->posterior_mean.drift;
        double m = pr->results[i].posterior_mean.mu_vol - agg->posterior_mean.mu_vol;
        double s = pr->results[i].posterior_mean.sigma_vol - agg->posterior_mean.sigma_vol;
        var_drift += d * d;
        var_mu += m * m;
        var_sigma += s * s;
    }
    agg->posterior_std.drift = sqrt(var_drift / pr->n_chains);
    agg->posterior_std.mu_vol = sqrt(var_mu / pr->n_chains);
    agg->posterior_std.sigma_vol = sqrt(var_sigma / pr->n_chains);
}

static void pmmh_parallel_free(PMMHParallelResult *pr)
{
    free(pr->results);
    pr->results = NULL;
}

#endif /* PMMH_MKL_H */