/**
 * @file pmmh_mkl.h
 * @brief High-performance PMMH using Intel MKL
 *
 * Optimizations:
 *   - VSL for vectorized RNG (Gaussian, uniform)
 *   - VML for vectorized exp/log
 *   - OpenMP threading for particle operations
 *   - SIMD-friendly memory layout
 *
 * Compile: gcc -O3 -march=native -fopenmp -DMKL_ILP64 -I${MKLROOT}/include \
 *          pmmh_mkl.c -L${MKLROOT}/lib/intel64 -lmkl_intel_ilp64 \
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

#define PMMH_N_PARTICLES     256
#define PMMH_N_ITERATIONS    400
#define PMMH_N_BURNIN        150
#define PMMH_CACHE_LINE      64

/* Align allocations for SIMD */
#define ALIGNED_ALLOC(size) mkl_malloc((size), PMMH_CACHE_LINE)
#define ALIGNED_FREE(ptr)   mkl_free(ptr)

/*============================================================================
 * DATA STRUCTURES
 *============================================================================*/

typedef struct {
    double drift;
    double mu_vol;
    double sigma_vol;
} PMMHParams;

typedef struct {
    PMMHParams mean;
    PMMHParams std;
} PMMHPrior;

typedef struct {
    PMMHParams posterior_mean;
    PMMHParams posterior_std;
    double acceptance_rate;
    int n_samples;
    double elapsed_ms;
} PMMHResult;

typedef struct {
    /* Particle state - aligned for SIMD */
    double *log_vol;        /* [n_particles] current log-volatility */
    double *log_vol_new;    /* [n_particles] proposed log-volatility */
    double *weights;        /* [n_particles] log-weights */
    double *weights_exp;    /* [n_particles] exp(weights) for resampling */
    double *noise;          /* [n_particles] Gaussian noise buffer */
    double *uniform;        /* [n_particles] uniform noise for resampling */
    int *ancestors;         /* [n_particles] resampling indices */
    
    /* VSL streams - one per thread for parallel RNG */
    VSLStreamStatePtr *streams;
    int n_threads;
    int n_particles;
    
    /* Model parameters */
    double theta_vol;       /* Mean reversion speed */
    
} PMMHState;

/*============================================================================
 * INITIALIZATION / CLEANUP
 *============================================================================*/

static PMMHState* pmmh_state_create(int n_particles, double theta_vol) {
    PMMHState *s = (PMMHState*)ALIGNED_ALLOC(sizeof(PMMHState));
    memset(s, 0, sizeof(PMMHState));
    
    s->n_particles = n_particles;
    s->theta_vol = theta_vol;
    
    /* Allocate aligned particle arrays */
    s->log_vol = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->log_vol_new = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->weights = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->weights_exp = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->noise = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->uniform = (double*)ALIGNED_ALLOC(n_particles * sizeof(double));
    s->ancestors = (int*)ALIGNED_ALLOC(n_particles * sizeof(int));
    
    /* Create VSL streams - one per thread */
    s->n_threads = omp_get_max_threads();
    s->streams = (VSLStreamStatePtr*)malloc(s->n_threads * sizeof(VSLStreamStatePtr));
    
    for (int i = 0; i < s->n_threads; i++) {
        /* Use MT2203 - good for parallel streams with different seeds */
        vslNewStream(&s->streams[i], VSL_BRNG_MT2203 + i, 12345 + i * 7919);
    }
    
    return s;
}

static void pmmh_state_destroy(PMMHState *s) {
    if (!s) return;
    
    for (int i = 0; i < s->n_threads; i++) {
        vslDeleteStream(&s->streams[i]);
    }
    free(s->streams);
    
    ALIGNED_FREE(s->log_vol);
    ALIGNED_FREE(s->log_vol_new);
    ALIGNED_FREE(s->weights);
    ALIGNED_FREE(s->weights_exp);
    ALIGNED_FREE(s->noise);
    ALIGNED_FREE(s->uniform);
    ALIGNED_FREE(s->ancestors);
    ALIGNED_FREE(s);
}

static void pmmh_state_seed(PMMHState *s, unsigned int seed) {
    for (int i = 0; i < s->n_threads; i++) {
        vslDeleteStream(&s->streams[i]);
        vslNewStream(&s->streams[i], VSL_BRNG_MT2203 + i, seed + i * 7919);
    }
}

/*============================================================================
 * VECTORIZED PARTICLE FILTER LOG-LIKELIHOOD
 *============================================================================*/

static double pmmh_log_likelihood_mkl(PMMHState *s, 
                                       const double *returns, int n_obs,
                                       const PMMHParams *params) {
    const int np = s->n_particles;
    const double theta = s->theta_vol;
    const double one_minus_theta = 1.0 - theta;
    const double theta_mu = theta * params->mu_vol;
    const double drift = params->drift;
    const double sigma_vol = params->sigma_vol;
    
    /* Initialize particles at mu_vol with some spread */
    int tid = omp_get_thread_num();
    vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->streams[tid],
                  np, s->log_vol, params->mu_vol, 0.3);
    
    double total_log_lik = 0.0;
    
    for (int t = 0; t < n_obs; t++) {
        const double ret = returns[t];
        
        /* Generate noise for volatility evolution: N(0, sigma_vol^2) */
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, s->streams[tid],
                      np, s->noise, 0.0, sigma_vol);
        
        /* Propagate volatility: log_vol' = (1-θ)*log_vol + θ*μ + σ*ε */
        /* Using VML for vectorized operations */
        #pragma omp simd
        for (int i = 0; i < np; i++) {
            s->log_vol_new[i] = one_minus_theta * s->log_vol[i] + theta_mu + s->noise[i];
        }
        
        /* Compute log-weights: log p(y|x) = -0.5*(y-μ)²/σ² - log(σ) */
        /* First compute exp(log_vol) = volatility */
        vdExp(np, s->log_vol_new, s->weights_exp);  /* weights_exp = exp(log_vol) = vol */
        
        /* Compute log-weights */
        double max_log_w = -DBL_MAX;
        #pragma omp simd reduction(max:max_log_w)
        for (int i = 0; i < np; i++) {
            double vol = s->weights_exp[i];
            double z = (ret - drift) / vol;
            double log_w = -0.5 * z * z - s->log_vol_new[i];  /* -log(vol) = -log_vol */
            s->weights[i] = log_w;
            if (log_w > max_log_w) max_log_w = log_w;
        }
        
        /* Log-sum-exp for marginal likelihood */
        /* Subtract max for numerical stability, then exp */
        #pragma omp simd
        for (int i = 0; i < np; i++) {
            s->weights[i] -= max_log_w;
        }
        vdExp(np, s->weights, s->weights_exp);
        
        /* Sum weights */
        double sum_w = cblas_dasum(np, s->weights_exp, 1);
        total_log_lik += max_log_w + log(sum_w / np);
        
        /* Normalize weights for resampling */
        double inv_sum = 1.0 / sum_w;
        cblas_dscal(np, inv_sum, s->weights_exp, 1);
        
        /* Systematic resampling */
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, s->streams[tid],
                     1, s->uniform, 0.0, 1.0);
        double u = s->uniform[0] / np;
        double cumsum = 0.0;
        int j = 0;
        
        for (int i = 0; i < np; i++) {
            double target = u + (double)i / np;
            while (cumsum + s->weights_exp[j] < target && j < np - 1) {
                cumsum += s->weights_exp[j];
                j++;
            }
            s->ancestors[i] = j;
        }
        
        /* Gather resampled particles */
        #pragma omp simd
        for (int i = 0; i < np; i++) {
            s->log_vol[i] = s->log_vol_new[s->ancestors[i]];
        }
    }
    
    return total_log_lik;
}

/*============================================================================
 * LOG-PRIOR
 *============================================================================*/

static double pmmh_log_prior(const PMMHParams *p, const PMMHPrior *prior) {
    double lp = 0.0;
    
    /* Gaussian priors */
    double z_drift = (p->drift - prior->mean.drift) / prior->std.drift;
    double z_mu = (p->mu_vol - prior->mean.mu_vol) / prior->std.mu_vol;
    double z_sigma = (log(p->sigma_vol) - log(prior->mean.sigma_vol)) / prior->std.sigma_vol;
    
    lp -= 0.5 * (z_drift * z_drift + z_mu * z_mu + z_sigma * z_sigma);
    lp -= log(prior->std.drift) + log(prior->std.mu_vol) + log(prior->std.sigma_vol);
    lp -= log(p->sigma_vol);  /* Jacobian for log-normal */
    
    return lp;
}

/*============================================================================
 * MAIN PMMH SAMPLER
 *============================================================================*/

static void pmmh_run_mkl(const double *returns, int n_obs,
                         const PMMHPrior *prior,
                         double theta_vol,
                         int n_iterations, int n_burnin,
                         int n_particles,
                         PMMHResult *result) {
    
    double t_start = omp_get_wtime();
    
    /* Create state */
    PMMHState *state = pmmh_state_create(n_particles, theta_vol);
    
    /* Initialize chain at prior mean */
    PMMHParams current = prior->mean;
    double current_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &current);
    double current_lp = pmmh_log_prior(&current, prior);
    
    /* Proposal standard deviations */
    double prop_drift_std = 0.0004;
    double prop_mu_std = 0.08;
    double prop_sigma_log_std = 0.05;
    
    /* Sample storage */
    int max_samples = n_iterations - n_burnin;
    double *samples_drift = (double*)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_mu = (double*)ALIGNED_ALLOC(max_samples * sizeof(double));
    double *samples_sigma = (double*)ALIGNED_ALLOC(max_samples * sizeof(double));
    
    int n_accepted = 0;
    int sample_idx = 0;
    
    /* Proposal noise buffer */
    double prop_noise[3];
    int tid = omp_get_thread_num();
    
    for (int iter = 0; iter < n_iterations; iter++) {
        /* Generate proposal noise */
        vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, state->streams[tid],
                      3, prop_noise, 0.0, 1.0);
        
        /* Propose new parameters */
        PMMHParams proposed;
        proposed.drift = current.drift + prop_noise[0] * prop_drift_std;
        proposed.mu_vol = current.mu_vol + prop_noise[1] * prop_mu_std;
        proposed.sigma_vol = current.sigma_vol * exp(prop_noise[2] * prop_sigma_log_std);
        
        /* Clamp to bounds */
        if (proposed.drift < -0.01) proposed.drift = -0.01;
        if (proposed.drift > 0.01) proposed.drift = 0.01;
        if (proposed.mu_vol < -8.0) proposed.mu_vol = -8.0;
        if (proposed.mu_vol > 0.0) proposed.mu_vol = 0.0;
        if (proposed.sigma_vol < 0.01) proposed.sigma_vol = 0.01;
        if (proposed.sigma_vol > 0.5) proposed.sigma_vol = 0.5;
        
        /* Compute acceptance ratio */
        double prop_ll = pmmh_log_likelihood_mkl(state, returns, n_obs, &proposed);
        double prop_lp = pmmh_log_prior(&proposed, prior);
        
        double log_alpha = (prop_ll + prop_lp) - (current_ll + current_lp);
        
        /* MH accept/reject */
        double u;
        vdRngUniform(VSL_RNG_METHOD_UNIFORM_STD, state->streams[tid], 1, &u, 0.0, 1.0);
        
        if (log(u) < log_alpha) {
            current = proposed;
            current_ll = prop_ll;
            current_lp = prop_lp;
            n_accepted++;
        }
        
        /* Store sample after burnin */
        if (iter >= n_burnin) {
            samples_drift[sample_idx] = current.drift;
            samples_mu[sample_idx] = current.mu_vol;
            samples_sigma[sample_idx] = current.sigma_vol;
            sample_idx++;
        }
    }
    
    /* Compute posterior statistics using MKL */
    result->n_samples = sample_idx;
    
    /* Mean */
    result->posterior_mean.drift = cblas_dasum(sample_idx, samples_drift, 1) / sample_idx;
    result->posterior_mean.mu_vol = cblas_dasum(sample_idx, samples_mu, 1) / sample_idx;
    result->posterior_mean.sigma_vol = cblas_dasum(sample_idx, samples_sigma, 1) / sample_idx;
    
    /* Fix sign for dasum (it returns absolute sum) */
    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    for (int i = 0; i < sample_idx; i++) {
        sum_drift += samples_drift[i];
        sum_mu += samples_mu[i];
        sum_sigma += samples_sigma[i];
    }
    result->posterior_mean.drift = sum_drift / sample_idx;
    result->posterior_mean.mu_vol = sum_mu / sample_idx;
    result->posterior_mean.sigma_vol = sum_sigma / sample_idx;
    
    /* Std via sum of squares */
    double sum_sq_drift = cblas_ddot(sample_idx, samples_drift, 1, samples_drift, 1);
    double sum_sq_mu = cblas_ddot(sample_idx, samples_mu, 1, samples_mu, 1);
    double sum_sq_sigma = cblas_ddot(sample_idx, samples_sigma, 1, samples_sigma, 1);
    
    result->posterior_std.drift = sqrt(sum_sq_drift / sample_idx - 
                                        result->posterior_mean.drift * result->posterior_mean.drift);
    result->posterior_std.mu_vol = sqrt(sum_sq_mu / sample_idx - 
                                         result->posterior_mean.mu_vol * result->posterior_mean.mu_vol);
    result->posterior_std.sigma_vol = sqrt(sum_sq_sigma / sample_idx - 
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
 * PARALLEL MONTE CARLO - RUN MULTIPLE CHAINS
 *============================================================================*/

typedef struct {
    PMMHResult *results;    /* [n_chains] */
    int n_chains;
    double total_elapsed_ms;
} PMMHParallelResult;

static void pmmh_run_parallel(const double *returns, int n_obs,
                               const PMMHPrior *prior,
                               double theta_vol,
                               int n_iterations, int n_burnin,
                               int n_particles,
                               int n_chains,
                               PMMHParallelResult *result) {
    
    double t_start = omp_get_wtime();
    
    result->n_chains = n_chains;
    result->results = (PMMHResult*)malloc(n_chains * sizeof(PMMHResult));
    
    /* Run chains in parallel */
    #pragma omp parallel for schedule(dynamic)
    for (int chain = 0; chain < n_chains; chain++) {
        /* Each chain gets different seed based on chain ID */
        PMMHState *state = pmmh_state_create(n_particles, theta_vol);
        pmmh_state_seed(state, 12345 + chain * 104729);
        
        pmmh_run_mkl(returns, n_obs, prior, theta_vol,
                     n_iterations, n_burnin, n_particles,
                     &result->results[chain]);
        
        pmmh_state_destroy(state);
    }
    
    result->total_elapsed_ms = (omp_get_wtime() - t_start) * 1000.0;
}

static void pmmh_parallel_aggregate(const PMMHParallelResult *pr, PMMHResult *agg) {
    /* Aggregate results across chains */
    double sum_drift = 0, sum_mu = 0, sum_sigma = 0;
    double sum_acc = 0;
    int total_samples = 0;
    
    for (int i = 0; i < pr->n_chains; i++) {
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
    for (int i = 0; i < pr->n_chains; i++) {
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

static void pmmh_parallel_free(PMMHParallelResult *pr) {
    free(pr->results);
    pr->results = NULL;
}

#endif /* PMMH_MKL_H */
