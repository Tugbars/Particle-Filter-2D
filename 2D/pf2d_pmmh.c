/**
 * @file pf2d_pmmh.c
 * @brief Particle Marginal Metropolis-Hastings Implementation
 *
 * PMMH for online parameter estimation triggered by BOCPD.
 * Runs asynchronously to avoid blocking the main PF loop.
 *
 * Algorithm:
 *   1. Propose θ' = θ + ε (random walk)
 *   2. Run PF to estimate log p(y_{1:T} | θ')
 *   3. Accept/reject via Metropolis-Hastings
 *   4. Adapt proposal std based on acceptance rate
 *   5. Return posterior mean after burn-in
 */

#include "pf2d_pmmh.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>

#ifdef _WIN32
#include <windows.h>
#define pmmh_get_time_ms() ((double)GetTickCount64())
#else
#include <sys/time.h>
static inline double pmmh_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

/*============================================================================
 * INTERNAL HELPERS
 *============================================================================*/

static inline pf2d_real pmmh_randn(VSLStreamStatePtr rng) {
    pf2d_real x;
    pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng, 1, &x, 0.0, 1.0);
    return x;
}

static inline pf2d_real pmmh_randu(VSLStreamStatePtr rng) {
    pf2d_real x;
    pf2d_RngUniform(VSL_RNG_METHOD_UNIFORM_STD, rng, 1, &x, 0.0, 1.0);
    return x;
}

static inline pf2d_real pmmh_clamp(pf2d_real x, pf2d_real lo, pf2d_real hi) {
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

/*============================================================================
 * OBSERVATION WINDOW
 *============================================================================*/

PMMHObsWindow* pmmh_obs_window_create(int capacity) {
    PMMHObsWindow *win = (PMMHObsWindow*)malloc(sizeof(PMMHObsWindow));
    if (!win) return NULL;
    
    win->buffer = (pf2d_real*)mkl_malloc(capacity * sizeof(pf2d_real), 64);
    if (!win->buffer) {
        free(win);
        return NULL;
    }
    
    win->capacity = capacity;
    win->head = 0;
    win->count = 0;
    
    return win;
}

void pmmh_obs_window_destroy(PMMHObsWindow *win) {
    if (!win) return;
    mkl_free(win->buffer);
    free(win);
}

void pmmh_obs_window_push(PMMHObsWindow *win, pf2d_real obs) {
    win->buffer[win->head] = obs;
    win->head = (win->head + 1) % win->capacity;
    if (win->count < win->capacity) {
        win->count++;
    }
}

int pmmh_obs_window_copy(const PMMHObsWindow *win, pf2d_real *out) {
    if (win->count == 0) return 0;
    
    if (win->count < win->capacity) {
        /* Buffer not full yet - simple copy from start */
        memcpy(out, win->buffer, win->count * sizeof(pf2d_real));
    } else {
        /* Buffer full - copy in order (oldest first) */
        int oldest = win->head;  /* head points to oldest when full */
        int first_chunk = win->capacity - oldest;
        memcpy(out, &win->buffer[oldest], first_chunk * sizeof(pf2d_real));
        memcpy(&out[first_chunk], win->buffer, oldest * sizeof(pf2d_real));
    }
    
    return win->count;
}

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

void pmmh_config_defaults(PMMHConfig *cfg) {
    cfg->n_iterations = PMMH_DEFAULT_ITERATIONS;
    cfg->n_burnin = PMMH_DEFAULT_BURNIN;
    cfg->n_particles = PMMH_DEFAULT_PARTICLES;
    cfg->posterior_window = PMMH_DEFAULT_POSTERIOR_WINDOW;
    cfg->target_regime = 0;
    
    /* Proposal std - tuned for typical financial data */
    cfg->proposal_std.drift_std = (pf2d_real)0.0005;
    cfg->proposal_std.mu_vol_std = (pf2d_real)0.08;
    cfg->proposal_std.sigma_vol_log_std = (pf2d_real)0.02;
    cfg->adaptive_proposal = 1;
    
    /* Prior - will be set from PF */
    memset(&cfg->prior, 0, sizeof(cfg->prior));
    
    cfg->window_size = 500;
    
    /* Parameter bounds */
    cfg->drift_min = (pf2d_real)PMMH_DRIFT_MIN;
    cfg->drift_max = (pf2d_real)PMMH_DRIFT_MAX;
}

void pmmh_config_set_prior_from_pf(PMMHConfig *cfg, const PF2D *pf, 
                                    int regime, pf2d_real prior_scale) {
    const PF2DRegimeParams *p = &pf->regimes_params[regime];
    
    /* Prior mean = current params */
    cfg->prior.mean.drift = p->drift;
    cfg->prior.mean.mu_vol = p->mu_vol;
    cfg->prior.mean.sigma_vol = p->sigma_vol;
    
    /* Prior std - scaled by prior_scale
     * Larger scale = looser prior = more exploration */
    cfg->prior.std.drift = (pf2d_real)0.002 * prior_scale;
    cfg->prior.std.mu_vol = (pf2d_real)0.2 * prior_scale;
    cfg->prior.std.sigma_vol = (pf2d_real)0.05 * prior_scale;
    
    cfg->target_regime = regime;
}

/*============================================================================
 * LOG-LIKELIHOOD COMPUTATION VIA PARTICLE FILTER
 *
 * This is the core of PMMH: run a particle filter and accumulate
 * log(sum of unnormalized weights) at each step.
 *
 * log p(y_{1:T} | θ) = Σ_{t=0}^{T-1} log(Σ_i w_i^t)
 *
 * We use a simplified PF here (single regime, no regime switching)
 * since we're estimating parameters for one regime at a time.
 *============================================================================*/

pf2d_real pmmh_compute_log_likelihood(const pf2d_real *observations, int n_obs,
                                       const PMMHParams *params,
                                       pf2d_real theta_vol, pf2d_real rho,
                                       int n_particles, VSLStreamStatePtr rng) {
    if (n_obs <= 0 || n_particles <= 0) return -INFINITY;
    
    /* Allocate particle state + resampling buffers (FIX #2: preallocate) */
    pf2d_real *prices = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);
    pf2d_real *log_vols = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);
    pf2d_real *weights = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);
    pf2d_real *noise1 = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);
    pf2d_real *noise2 = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);
    pf2d_real *new_log_vols = (pf2d_real*)mkl_malloc(n_particles * sizeof(pf2d_real), 64);  /* FIX #2 */
    
    if (!prices || !log_vols || !weights || !noise1 || !noise2 || !new_log_vols) {
        mkl_free(prices); mkl_free(log_vols); mkl_free(weights);
        mkl_free(noise1); mkl_free(noise2); mkl_free(new_log_vols);
        return -INFINITY;
    }
    
    /* Precompute regime parameters */
    pf2d_real drift = params->drift;
    pf2d_real mu_vol = params->mu_vol;
    pf2d_real sigma_vol = params->sigma_vol;
    pf2d_real one_minus_theta = (pf2d_real)1.0 - theta_vol;
    pf2d_real theta_mu = theta_vol * mu_vol;
    pf2d_real sqrt_one_minus_rho_sq = (pf2d_real)sqrt(1.0 - (double)(rho * rho));
    
    /* Observation model */
    pf2d_real obs_var = (pf2d_real)0.0001;  /* Same as main PF */
    pf2d_real neg_half_inv_var = (pf2d_real)-0.5 / obs_var;
    
    /* FIX #1: Initialize particles from PRIOR, not centered on y_0 
     * Use broader spread so first observation contributes to likelihood */
    pf2d_real init_price = observations[0];
    pf2d_real init_log_vol = mu_vol;
    pf2d_real init_spread_p = (pf2d_real)0.01;   /* Wider than before */
    pf2d_real init_spread_v = (pf2d_real)0.2;    /* Wider than before */
    
    pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng,
                     n_particles, prices, init_price, init_spread_p);
    pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng,
                     n_particles, log_vols, init_log_vol, init_spread_v);
    
    pf2d_real uniform_weight = (pf2d_real)1.0 / n_particles;
    
    /* FIX #1: Compute t=0 likelihood contribution */
    pf2d_real log_lik = (pf2d_real)0.0;
    {
        pf2d_real obs = observations[0];
        pf2d_real max_log_w = (pf2d_real)(-1e30);
        
        for (int i = 0; i < n_particles; i++) {
            pf2d_real diff = obs - prices[i];
            pf2d_real log_w = neg_half_inv_var * diff * diff;
            weights[i] = log_w;
            if (log_w > max_log_w) max_log_w = log_w;
        }
        
        pf2d_real sum_w = (pf2d_real)0.0;
        for (int i = 0; i < n_particles; i++) {
            weights[i] = (pf2d_real)exp((double)(weights[i] - max_log_w));
            sum_w += weights[i];
        }
        
        if (sum_w > 0) {
            log_lik = max_log_w + (pf2d_real)log((double)sum_w);
            pf2d_real inv_sum = (pf2d_real)1.0 / sum_w;
            for (int i = 0; i < n_particles; i++) {
                weights[i] *= inv_sum;
            }
        } else {
            mkl_free(prices); mkl_free(log_vols); mkl_free(weights);
            mkl_free(noise1); mkl_free(noise2); mkl_free(new_log_vols);
            return -INFINITY;
        }
    }
    
    /* Process observations t=1 to n_obs-1 */
    for (int t = 1; t < n_obs; t++) {
        pf2d_real obs = observations[t];
        
        /* Generate noise */
        pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng,
                         n_particles, noise1, 0.0, 1.0);
        pf2d_RngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, rng,
                         n_particles, noise2, 0.0, 1.0);
        
        /* Propagate and weight */
        pf2d_real max_log_w = (pf2d_real)(-1e30);
        
        for (int i = 0; i < n_particles; i++) {
            /* Correlated noise */
            pf2d_real eps_p = noise1[i];
            pf2d_real eps_v = rho * noise1[i] + sqrt_one_minus_rho_sq * noise2[i];
            
            /* Volatility */
            pf2d_real vol = (pf2d_real)exp((double)log_vols[i]);
            
            /* Price dynamics */
            prices[i] = prices[i] + drift + vol * eps_p;
            
            /* Log-vol dynamics */
            log_vols[i] = one_minus_theta * log_vols[i] + theta_mu + sigma_vol * eps_v;
            
            /* FIX #3: Clamp log_vols to prevent vol explosion */
            log_vols[i] = pmmh_clamp(log_vols[i], 
                                      (pf2d_real)PMMH_LOG_VOL_MIN, 
                                      (pf2d_real)PMMH_LOG_VOL_MAX);
            
            /* Log-likelihood (Gaussian observation) */
            pf2d_real diff = obs - prices[i];
            pf2d_real log_w = neg_half_inv_var * diff * diff;
            weights[i] = log_w;  /* Store log-weight temporarily */
            
            if (log_w > max_log_w) max_log_w = log_w;
        }
        
        /* Log-sum-exp for numerical stability */
        pf2d_real sum_w = (pf2d_real)0.0;
        for (int i = 0; i < n_particles; i++) {
            weights[i] = (pf2d_real)exp((double)(weights[i] - max_log_w));
            sum_w += weights[i];
        }
        
        /* Accumulate log-likelihood */
        if (sum_w > 0) {
            log_lik += max_log_w + (pf2d_real)log((double)sum_w);
        } else {
            log_lik = -INFINITY;
            break;
        }
        
        /* Normalize weights */
        pf2d_real inv_sum = (pf2d_real)1.0 / sum_w;
        for (int i = 0; i < n_particles; i++) {
            weights[i] *= inv_sum;
        }
        
        /* Simple resampling every 10 steps or when ESS low */
        pf2d_real sum_sq = (pf2d_real)0.0;
        for (int i = 0; i < n_particles; i++) {
            sum_sq += weights[i] * weights[i];
        }
        pf2d_real ess = (pf2d_real)1.0 / sum_sq;
        
        if (ess < n_particles * 0.5 || t % 10 == 0) {
            /* Systematic resampling */
            pf2d_real *cumsum = noise1;  /* Reuse buffer */
            cumsum[0] = weights[0];
            for (int i = 1; i < n_particles; i++) {
                cumsum[i] = cumsum[i-1] + weights[i];
            }
            
            pf2d_real u0 = pmmh_randu(rng) * uniform_weight;
            
            /* FIX #2: Use preallocated buffer, no malloc inside loop */
            pf2d_real *new_prices = noise2;  /* Reuse as temp */
            
            int idx = 0;
            for (int i = 0; i < n_particles; i++) {
                pf2d_real u = u0 + i * uniform_weight;
                while (cumsum[idx] < u && idx < n_particles - 1) idx++;
                new_prices[i] = prices[idx];
                new_log_vols[i] = log_vols[idx];
            }
            
            memcpy(prices, new_prices, n_particles * sizeof(pf2d_real));
            memcpy(log_vols, new_log_vols, n_particles * sizeof(pf2d_real));
            
            /* Reset weights */
            for (int i = 0; i < n_particles; i++) {
                weights[i] = uniform_weight;
            }
        }
    }
    
    /* Cleanup */
    mkl_free(prices);
    mkl_free(log_vols);
    mkl_free(weights);
    mkl_free(noise1);
    mkl_free(noise2);
    mkl_free(new_log_vols);  /* FIX #2 */
    
    return log_lik;
}

/*============================================================================
 * PRIOR AND PROPOSAL
 *============================================================================*/

static pf2d_real pmmh_log_prior(const PMMHParams *params, const PMMHPrior *prior) {
    pf2d_real log_p = (pf2d_real)0.0;
    
    /* Gaussian log-prior for each parameter */
    pf2d_real d;
    
    /* drift ~ N(prior_mean, prior_std) */
    d = (params->drift - prior->mean.drift) / prior->std.drift;
    log_p -= (pf2d_real)0.5 * d * d;
    
    /* mu_vol ~ N(prior_mean, prior_std) */
    d = (params->mu_vol - prior->mean.mu_vol) / prior->std.mu_vol;
    log_p -= (pf2d_real)0.5 * d * d;
    
    /* sigma_vol ~ LogNormal (enforce positivity) */
    if (params->sigma_vol <= 0) return -INFINITY;
    pf2d_real log_sv = (pf2d_real)log((double)params->sigma_vol);
    pf2d_real log_sv_mean = (pf2d_real)log((double)prior->mean.sigma_vol);
    d = (log_sv - log_sv_mean) / prior->std.sigma_vol;
    log_p -= (pf2d_real)0.5 * d * d;
    log_p -= log_sv;  /* Jacobian for log-normal */
    
    return log_p;
}

static void pmmh_propose(PMMHParams *proposed, const PMMHParams *current,
                         const PMMHProposalStd *std, 
                         pf2d_real drift_min, pf2d_real drift_max,
                         VSLStreamStatePtr rng) {
    /* Random walk proposals */
    proposed->drift = current->drift + std->drift_std * pmmh_randn(rng);
    proposed->mu_vol = current->mu_vol + std->mu_vol_std * pmmh_randn(rng);
    
    /* FIX #7: Clamp drift to prevent dynamics breakdown */
    proposed->drift = pmmh_clamp(proposed->drift, drift_min, drift_max);
    
    /* Log-normal for sigma_vol (ensures positivity) */
    proposed->sigma_vol = current->sigma_vol * 
                          (pf2d_real)exp((double)(std->sigma_vol_log_std * pmmh_randn(rng)));
    
    /* Clamp sigma_vol to reasonable range */
    proposed->sigma_vol = pmmh_clamp(proposed->sigma_vol, (pf2d_real)0.001, (pf2d_real)1.0);
}

/*============================================================================
 * ADAPTIVE PROPOSAL
 *============================================================================*/

static void pmmh_adapt_proposal(PMMHProposalStd *std, pf2d_real acceptance_rate) {
    /* Scale factor based on acceptance rate */
    pf2d_real scale;
    
    if (acceptance_rate < PMMH_TARGET_ACCEPTANCE_LOW) {
        /* Too few accepts - shrink proposal */
        scale = (pf2d_real)0.9;
    } else if (acceptance_rate > PMMH_TARGET_ACCEPTANCE_HIGH) {
        /* Too many accepts - expand proposal */
        scale = (pf2d_real)1.1;
    } else {
        /* In target range - no change */
        return;
    }
    
    std->drift_std *= scale;
    std->mu_vol_std *= scale;
    std->sigma_vol_log_std *= scale;
    
    /* Clamp to reasonable bounds */
    std->drift_std = pmmh_clamp(std->drift_std, (pf2d_real)0.00001, (pf2d_real)0.01);
    std->mu_vol_std = pmmh_clamp(std->mu_vol_std, (pf2d_real)0.01, (pf2d_real)0.5);
    std->sigma_vol_log_std = pmmh_clamp(std->sigma_vol_log_std, (pf2d_real)0.005, (pf2d_real)0.2);
}

/*============================================================================
 * CORE PMMH ALGORITHM
 *============================================================================*/

static void pmmh_run_internal(const pf2d_real *observations, int n_obs,
                              const PMMHConfig *cfg,
                              pf2d_real theta_vol, pf2d_real rho,
                              VSLStreamStatePtr rng,
                              atomic_bool *cancel_flag,
                              PMMHResult *result) {
    int n_iter = cfg->n_iterations;
    int n_burnin = cfg->n_burnin;
    int n_particles = cfg->n_particles;
    
    /* Initialize from prior mean */
    PMMHParams current = cfg->prior.mean;
    PMMHProposalStd proposal_std = cfg->proposal_std;
    
    /* Compute initial log-likelihood */
    pf2d_real current_log_lik = pmmh_compute_log_likelihood(
        observations, n_obs, &current, theta_vol, rho, n_particles, rng);
    pf2d_real current_log_prior = pmmh_log_prior(&current, &cfg->prior);
    pf2d_real current_log_post = current_log_lik + current_log_prior;
    
    /* Storage for posterior samples (post burn-in) */
    int max_samples = n_iter - n_burnin;
    if (max_samples <= 0) max_samples = 1;
    
    PMMHParams *samples = (PMMHParams*)malloc(max_samples * sizeof(PMMHParams));
    int n_samples = 0;
    
    /* MCMC statistics */
    int n_accepted = 0;
    int n_since_adapt = 0;
    int n_accepted_since_adapt = 0;
    
    /* Main MCMC loop */
    for (int iter = 0; iter < n_iter; iter++) {
        /* Check for cancellation */
        if (cancel_flag && atomic_load(cancel_flag)) {
            break;
        }
        
        /* Propose new parameters (FIX #7: with drift clamping) */
        PMMHParams proposed;
        pmmh_propose(&proposed, &current, &proposal_std,
                     cfg->drift_min, cfg->drift_max, rng);
        
        /* Compute log-posterior for proposal */
        pf2d_real proposed_log_prior = pmmh_log_prior(&proposed, &cfg->prior);
        
        pf2d_real proposed_log_lik;
        pf2d_real proposed_log_post;
        
        if (proposed_log_prior > -INFINITY) {
            proposed_log_lik = pmmh_compute_log_likelihood(
                observations, n_obs, &proposed, theta_vol, rho, n_particles, rng);
            proposed_log_post = proposed_log_lik + proposed_log_prior;
        } else {
            proposed_log_lik = -INFINITY;
            proposed_log_post = -INFINITY;
        }
        
        /* Metropolis-Hastings accept/reject
         * FIX #5: Use log(u) < log_alpha to avoid exp overflow */
        pf2d_real log_alpha = proposed_log_post - current_log_post;
        pf2d_real log_u = (pf2d_real)log((double)pmmh_randu(rng));
        
        int accept = (log_alpha >= 0) || (log_u < log_alpha);
        
        if (accept) {
            current = proposed;
            current_log_lik = proposed_log_lik;
            current_log_prior = proposed_log_prior;
            current_log_post = proposed_log_post;
            n_accepted++;
            n_accepted_since_adapt++;
        }
        
        /* Store sample after burn-in */
        if (iter >= n_burnin && samples) {
            samples[n_samples++] = current;
        }
        
        /* Adaptive proposal */
        n_since_adapt++;
        if (cfg->adaptive_proposal && n_since_adapt >= PMMH_ADAPTATION_INTERVAL) {
            pf2d_real recent_accept_rate = (pf2d_real)n_accepted_since_adapt / 
                                           (pf2d_real)n_since_adapt;
            pmmh_adapt_proposal(&proposal_std, recent_accept_rate);
            n_since_adapt = 0;
            n_accepted_since_adapt = 0;
        }
    }
    
    /* Compute posterior mean from samples */
    if (n_samples > 0 && samples) {
        /* FIX #8: Use configurable posterior window */
        int posterior_win = cfg->posterior_window > 0 ? cfg->posterior_window : 150;
        int use_last = (n_samples > posterior_win) ? posterior_win : n_samples;
        int start = n_samples - use_last;
        
        PMMHParams sum = {0, 0, 0};
        PMMHParams sum_sq = {0, 0, 0};
        
        for (int i = start; i < n_samples; i++) {
            sum.drift += samples[i].drift;
            sum.mu_vol += samples[i].mu_vol;
            sum.sigma_vol += samples[i].sigma_vol;
            
            sum_sq.drift += samples[i].drift * samples[i].drift;
            sum_sq.mu_vol += samples[i].mu_vol * samples[i].mu_vol;
            sum_sq.sigma_vol += samples[i].sigma_vol * samples[i].sigma_vol;
        }
        
        pf2d_real inv_n = (pf2d_real)1.0 / use_last;
        
        result->posterior_mean.drift = sum.drift * inv_n;
        result->posterior_mean.mu_vol = sum.mu_vol * inv_n;
        result->posterior_mean.sigma_vol = sum.sigma_vol * inv_n;
        
        /* Posterior std */
        result->posterior_std.drift = (pf2d_real)sqrt(
            (double)(sum_sq.drift * inv_n - result->posterior_mean.drift * result->posterior_mean.drift));
        result->posterior_std.mu_vol = (pf2d_real)sqrt(
            (double)(sum_sq.mu_vol * inv_n - result->posterior_mean.mu_vol * result->posterior_mean.mu_vol));
        result->posterior_std.sigma_vol = (pf2d_real)sqrt(
            (double)(sum_sq.sigma_vol * inv_n - result->posterior_mean.sigma_vol * result->posterior_mean.sigma_vol));
        
        result->n_samples_used = use_last;
    } else {
        /* Fallback to current */
        result->posterior_mean = current;
        result->posterior_std = (PMMHParams){0, 0, 0};
        result->n_samples_used = 0;
    }
    
    /* Diagnostics */
    result->acceptance_rate = (pf2d_real)n_accepted / (pf2d_real)n_iter;
    result->final_log_lik = current_log_lik;
    result->converged = (result->acceptance_rate >= PMMH_TARGET_ACCEPTANCE_LOW &&
                         result->acceptance_rate <= PMMH_TARGET_ACCEPTANCE_HIGH);
    
    free(samples);
}

/*============================================================================
 * SYNCHRONOUS API
 *============================================================================*/

void pmmh_run_sync(const pf2d_real *observations, int n_obs,
                   const PMMHConfig *cfg, const PF2D *pf,
                   PMMHResult *result) {
    /* Create RNG */
    VSLStreamStatePtr rng;
    vslNewStream(&rng, VSL_BRNG_SFMT19937, 12345);
    
    /* Get fixed params from PF */
    const PF2DRegimeParams *regime = &pf->regimes_params[cfg->target_regime];
    pf2d_real theta_vol = regime->theta_vol;
    pf2d_real rho = regime->rho;
    
    double t0 = pmmh_get_time_ms();
    
    /* Run PMMH */
    pmmh_run_internal(observations, n_obs, cfg, theta_vol, rho, rng, NULL, result);
    
    result->elapsed_ms = pmmh_get_time_ms() - t0;
    
    vslDeleteStream(&rng);
}

/*============================================================================
 * ASYNC API
 *============================================================================*/

static void* pmmh_thread_func(void *arg) {
    PMMHJob *job = (PMMHJob*)arg;
    
    double t0 = pmmh_get_time_ms();
    
    pmmh_run_internal(job->observations, job->n_obs, 
                      &job->config, job->theta_vol, job->rho,
                      job->rng, &job->cancel_flag, &job->result);
    
    job->result.elapsed_ms = pmmh_get_time_ms() - t0;
    
    atomic_store(&job->done_flag, 1);
    
    return NULL;
}

PMMHJob* pmmh_start_async(const PMMHObsWindow *win, 
                           const PMMHConfig *cfg,
                           const PF2D *pf) {
    if (!win || win->count == 0) return NULL;
    
    PMMHJob *job = (PMMHJob*)calloc(1, sizeof(PMMHJob));
    if (!job) return NULL;
    
    /* Determine window size to use */
    int n_obs = cfg->window_size;
    if (n_obs > win->count) n_obs = win->count;
    if (n_obs < PMMH_DEFAULT_WINDOW_MIN) n_obs = win->count;
    
    /* Allocate and copy observations */
    job->observations = (pf2d_real*)mkl_malloc(n_obs * sizeof(pf2d_real), 64);
    if (!job->observations) {
        free(job);
        return NULL;
    }
    
    /* Copy from ring buffer (may need two-part copy) */
    if (n_obs <= win->count) {
        /* Copy last n_obs from window */
        int start;
        if (win->count < win->capacity) {
            start = win->count - n_obs;
        } else {
            start = (win->head - n_obs + win->capacity) % win->capacity;
        }
        
        if (start + n_obs <= win->capacity) {
            memcpy(job->observations, &win->buffer[start], n_obs * sizeof(pf2d_real));
        } else {
            int first_chunk = win->capacity - start;
            memcpy(job->observations, &win->buffer[start], first_chunk * sizeof(pf2d_real));
            memcpy(&job->observations[first_chunk], win->buffer, (n_obs - first_chunk) * sizeof(pf2d_real));
        }
    }
    job->n_obs = n_obs;
    
    /* Copy config */
    job->config = *cfg;
    
    /* Get fixed params from PF */
    const PF2DRegimeParams *regime = &pf->regimes_params[cfg->target_regime];
    job->theta_vol = regime->theta_vol;
    job->rho = regime->rho;
    
    /* Initialize flags */
    atomic_store(&job->cancel_flag, 0);
    atomic_store(&job->done_flag, 0);
    
    /* Create RNG with unique seed */
    static int seed_counter = 0;
    vslNewStream(&job->rng, VSL_BRNG_SFMT19937, 77777 + (seed_counter++ * 1000));
    
    /* Spawn thread */
    if (pthread_create(&job->thread, NULL, pmmh_thread_func, job) != 0) {
        mkl_free(job->observations);
        vslDeleteStream(&job->rng);
        free(job);
        return NULL;
    }
    
    return job;
}

int pmmh_job_is_done(const PMMHJob *job) {
    if (!job) return 1;
    return atomic_load(&job->done_flag);
}

void pmmh_job_cancel(PMMHJob *job) {
    if (!job) return;
    atomic_store(&job->cancel_flag, 1);
}

void pmmh_job_finish(PMMHJob *job, PMMHResult *result) {
    if (!job) return;
    
    /* Wait for thread to complete */
    pthread_join(job->thread, NULL);
    
    /* Copy result */
    if (result) {
        *result = job->result;
    }
    
    /* Cleanup */
    mkl_free(job->observations);
    vslDeleteStream(&job->rng);
    free(job);
}

/*============================================================================
 * ATOMIC PARAMETER STORAGE
 *============================================================================*/

void pmmh_params_atomic_init(PMMHParamsAtomic *pa, const PF2D *pf) {
    memcpy(pa->params, pf->regimes_params, sizeof(pa->params));
    pthread_mutex_init(&pa->mutex, NULL);
    pa->version = 0;
}

void pmmh_params_atomic_destroy(PMMHParamsAtomic *pa) {
    pthread_mutex_destroy(&pa->mutex);
}

void pmmh_params_atomic_update(PMMHParamsAtomic *pa, int regime, 
                                const PMMHResult *result) {
    pthread_mutex_lock(&pa->mutex);
    
    pa->params[regime].drift = result->posterior_mean.drift;
    pa->params[regime].mu_vol = result->posterior_mean.mu_vol;
    pa->params[regime].sigma_vol = result->posterior_mean.sigma_vol;
    
    /* Recompute derived values */
    pa->params[regime].theta_mu = pa->params[regime].theta_vol * pa->params[regime].mu_vol;
    
    pa->version++;
    
    pthread_mutex_unlock(&pa->mutex);
}

int pmmh_params_atomic_apply(PMMHParamsAtomic *pa, PF2D *pf) {
    static int last_applied_version = -1;
    
    pthread_mutex_lock(&pa->mutex);
    
    if (pa->version == last_applied_version) {
        pthread_mutex_unlock(&pa->mutex);
        return 0;  /* No update */
    }
    
    /* Apply all regime parameters */
    for (int r = 0; r < pf->n_regimes; r++) {
        pf2d_set_regime_params(pf, r,
                               pa->params[r].drift,
                               pa->params[r].theta_vol,
                               pa->params[r].mu_vol,
                               pa->params[r].sigma_vol,
                               pa->params[r].rho);
    }
    
    last_applied_version = pa->version;
    
    pthread_mutex_unlock(&pa->mutex);
    
    /* CRITICAL: Reset adaptive scaling after PMCMC update
     * PMCMC has already calibrated σ_vol - don't double-adapt */
    pf2d_adaptive_reset_after_pmcmc(pf);
    
    return 1;  /* Updated */
}
