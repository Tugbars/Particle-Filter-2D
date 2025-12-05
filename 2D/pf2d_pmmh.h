/**
 * @file pf2d_pmmh.h
 * @brief Particle Marginal Metropolis-Hastings for PF2D Parameter Estimation
 *
 * Triggered by BOCPD changepoint detection to recalibrate regime parameters.
 * Runs asynchronously without blocking the main PF loop.
 *
 * Estimates per-regime: {drift, mu_vol, sigma_vol}
 * Fixes: {theta_vol, rho} (slow-moving, calibrated offline)
 */

#ifndef PF2D_PMMH_H
#define PF2D_PMMH_H

#include "particle_filter_2d.h"
#include <stdatomic.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

/*============================================================================
 * CONFIGURATION DEFAULTS
 *============================================================================*/

#define PMMH_DEFAULT_ITERATIONS     500
#define PMMH_DEFAULT_BURNIN         150
#define PMMH_DEFAULT_PARTICLES      256
#define PMMH_DEFAULT_WINDOW_MIN     300
#define PMMH_DEFAULT_WINDOW_MAX     1500
#define PMMH_DEFAULT_POSTERIOR_WINDOW 150

/* Adaptive proposal targets */
#define PMMH_TARGET_ACCEPTANCE_LOW  0.25
#define PMMH_TARGET_ACCEPTANCE_HIGH 0.40
#define PMMH_ADAPTATION_INTERVAL    50

/* Parameter bounds */
#define PMMH_DRIFT_MIN             (-0.005)
#define PMMH_DRIFT_MAX             (0.005)
#define PMMH_LOG_VOL_MIN           (-12.0)   /* exp(-12) ≈ 6e-6 */
#define PMMH_LOG_VOL_MAX           (6.0)     /* exp(6) ≈ 400 */

/*============================================================================
 * TYPES
 *============================================================================*/

/**
 * @brief Subset of regime parameters estimated by PMMH
 * 
 * Only the parameters that change on short timescales:
 *   - drift: can shift on regime change
 *   - mu_vol: long-term vol level can jump
 *   - sigma_vol: vol-of-vol can spike during turbulence
 */
typedef struct {
    pf2d_real drift;
    pf2d_real mu_vol;
    pf2d_real sigma_vol;
} PMMHParams;

/**
 * @brief Proposal distribution standard deviations
 * 
 * Random walk proposals:
 *   drift' = drift + N(0, drift_std²)
 *   mu_vol' = mu_vol + N(0, mu_vol_std²)
 *   sigma_vol' = sigma_vol * exp(N(0, sigma_vol_log_std²))  [log-normal]
 */
typedef struct {
    pf2d_real drift_std;          /* Linear RW std (e.g., 0.0005) */
    pf2d_real mu_vol_std;         /* Linear RW std (e.g., 0.08) */
    pf2d_real sigma_vol_log_std;  /* Log-RW std (e.g., 0.02) */
} PMMHProposalStd;

/**
 * @brief Prior distribution parameters
 * 
 * Gaussian priors centered on current regime parameters.
 * Keeps PMCMC from wandering too far from calibrated values.
 *
 * NOTE: For sigma_vol, std is in LOG-SPACE (not raw space).
 *       e.g., std.sigma_vol = 0.3 means 1-sigma range is exp(±0.3) ≈ [0.74x, 1.35x]
 */
typedef struct {
    PMMHParams mean;   /* Prior mean (typically current params) */
    PMMHParams std;    /* Prior std (sigma_vol std is LOG-SPACE) */
} PMMHPrior;

/**
 * @brief PMMH configuration
 */
typedef struct {
    /* MCMC settings */
    int n_iterations;        /* Total iterations (default 500) */
    int n_burnin;            /* Discard first N (default 150) */
    int n_particles;         /* Particles for likelihood PF (default 256) */
    int posterior_window;    /* Samples for posterior mean (default 150) */
    
    /* Target regime */
    int target_regime;       /* Which regime to recalibrate (dominant) */
    
    /* Proposal distribution */
    PMMHProposalStd proposal_std;
    int adaptive_proposal;   /* 1 = adapt std based on acceptance rate */
    
    /* Prior */
    PMMHPrior prior;
    
    /* Observation window */
    int window_size;         /* Number of observations to use */
    
    /* Parameter bounds */
    pf2d_real drift_min;     /* Min drift (default -0.005) */
    pf2d_real drift_max;     /* Max drift (default +0.005) */
    
} PMMHConfig;

/**
 * @brief PMMH results
 */
typedef struct {
    /* Posterior estimates */
    PMMHParams posterior_mean;   /* Mean of post-burnin samples */
    PMMHParams posterior_std;    /* Std of post-burnin samples */
    
    /* Diagnostics */
    pf2d_real acceptance_rate;   /* Fraction of accepted proposals */
    pf2d_real final_log_lik;     /* Log-likelihood at final params */
    int n_samples_used;          /* Samples used for posterior mean */
    int converged;               /* 1 if acceptance rate in target range */
    
    /* Timing */
    double elapsed_ms;           /* Wall clock time */
    
} PMMHResult;

/**
 * @brief Ring buffer for observation history
 */
typedef struct {
    pf2d_real *buffer;    /* Circular buffer storage */
    int capacity;         /* Max observations (e.g., 2000) */
    int head;             /* Next write position */
    int count;            /* Current count (≤ capacity) */
} PMMHObsWindow;

/**
 * @brief PMMH job handle for async execution
 */
typedef struct {
    /* Thread management */
    pthread_t thread;
    atomic_bool cancel_flag;     /* Set to request early termination */
    atomic_bool done_flag;       /* Set when PMCMC completes */
    
    /* Input (owned by job, freed on finish) */
    pf2d_real *observations;     /* Frozen copy of observation window */
    int n_obs;
    PMMHConfig config;
    
    /* Fixed parameters (not estimated) */
    pf2d_real theta_vol;         /* Mean-reversion speed */
    pf2d_real rho;               /* Price-vol correlation */
    
    /* Output */
    PMMHResult result;
    
    /* RNG state (separate from main PF) */
    VSLStreamStatePtr rng;
    
} PMMHJob;

/**
 * @brief Thread-safe parameter storage with mutex
 */
typedef struct {
    PF2DRegimeParams params[PF2D_MAX_REGIMES];
    pthread_mutex_t mutex;
    int version;                 /* Incremented on each update */
} PMMHParamsAtomic;

/*============================================================================
 * OBSERVATION WINDOW API
 *============================================================================*/

/**
 * @brief Create observation window
 * @param capacity Maximum observations to store (e.g., 2000)
 */
PMMHObsWindow* pmmh_obs_window_create(int capacity);

/**
 * @brief Destroy observation window
 */
void pmmh_obs_window_destroy(PMMHObsWindow *win);

/**
 * @brief Add observation to window
 */
void pmmh_obs_window_push(PMMHObsWindow *win, pf2d_real obs);

/**
 * @brief Copy window contents to contiguous array
 * @param out Output array (must have space for win->count elements)
 * @return Number of observations copied
 */
int pmmh_obs_window_copy(const PMMHObsWindow *win, pf2d_real *out);

/**
 * @brief Get window count
 */
static inline int pmmh_obs_window_count(const PMMHObsWindow *win) {
    return win->count;
}

/*============================================================================
 * PMMH CONFIGURATION API
 *============================================================================*/

/**
 * @brief Initialize config with defaults
 */
void pmmh_config_defaults(PMMHConfig *cfg);

/**
 * @brief Set prior from current PF regime parameters
 * @param cfg Config to update
 * @param pf Current particle filter (reads regime params)
 * @param regime Which regime's params to use as prior mean
 * @param prior_scale How tight the prior is (0.5 = moderate, 1.0 = loose)
 */
void pmmh_config_set_prior_from_pf(PMMHConfig *cfg, const PF2D *pf, 
                                    int regime, pf2d_real prior_scale);

/*============================================================================
 * ASYNC PMMH API
 *============================================================================*/

/**
 * @brief Start async PMMH job
 * 
 * Copies observation window and spawns background thread.
 * Returns immediately - use pmmh_job_is_done() to poll completion.
 *
 * @param win Observation window (copied, not consumed)
 * @param cfg PMMH configuration
 * @param pf Current particle filter (for fixed params theta_vol, rho)
 * @return Job handle, or NULL on failure
 */
PMMHJob* pmmh_start_async(const PMMHObsWindow *win, 
                           const PMMHConfig *cfg,
                           const PF2D *pf);

/**
 * @brief Check if PMMH job is complete (non-blocking)
 * @return 1 if done, 0 if still running
 */
int pmmh_job_is_done(const PMMHJob *job);

/**
 * @brief Request cancellation of running job
 * 
 * Sets cancel flag - job will terminate at next iteration.
 * Non-blocking: call pmmh_job_finish() to wait and cleanup.
 */
void pmmh_job_cancel(PMMHJob *job);

/**
 * @brief Wait for job completion and get results
 * 
 * Blocks until job finishes (or is cancelled).
 * Frees job resources - handle is invalid after this call.
 *
 * @param job Job handle
 * @param result Output: PMMH results (can be NULL to discard)
 */
void pmmh_job_finish(PMMHJob *job, PMMHResult *result);

/*============================================================================
 * PARAMETER UPDATE API
 *============================================================================*/

/**
 * @brief Initialize atomic parameter storage
 */
void pmmh_params_atomic_init(PMMHParamsAtomic *pa, const PF2D *pf);

/**
 * @brief Destroy atomic parameter storage
 */
void pmmh_params_atomic_destroy(PMMHParamsAtomic *pa);

/**
 * @brief Update regime parameters from PMMH result
 * 
 * Thread-safe: can be called from PMMH thread.
 *
 * @param pa Atomic parameter storage
 * @param regime Which regime to update
 * @param result PMMH result with new parameters
 */
void pmmh_params_atomic_update(PMMHParamsAtomic *pa, int regime, 
                                const PMMHResult *result);

/**
 * @brief Apply atomic parameters to PF
 * 
 * Call from main thread to sync PF with latest PMMH estimates.
 * Non-blocking if no update pending.
 *
 * @param pa Atomic parameter storage
 * @param pf Particle filter to update
 * @return 1 if params were updated, 0 if no change
 */
int pmmh_params_atomic_apply(PMMHParamsAtomic *pa, PF2D *pf);

/*============================================================================
 * SYNCHRONOUS API (for testing/debugging)
 *============================================================================*/

/**
 * @brief Run PMMH synchronously (blocking)
 * 
 * For testing or when async isn't needed.
 *
 * @param observations Observation array
 * @param n_obs Number of observations
 * @param cfg PMMH configuration
 * @param pf Current particle filter
 * @param result Output: PMMH results
 */
void pmmh_run_sync(const pf2d_real *observations, int n_obs,
                   const PMMHConfig *cfg, const PF2D *pf,
                   PMMHResult *result);

/**
 * @brief Compute log-likelihood for given parameters
 * 
 * Runs a particle filter over observations and returns marginal log-likelihood.
 * Used internally by PMMH, exposed for testing.
 *
 * @param observations Observation array
 * @param n_obs Number of observations
 * @param params Parameters to evaluate
 * @param theta_vol Fixed mean-reversion speed
 * @param rho Fixed price-vol correlation
 * @param n_particles Number of particles
 * @param rng MKL RNG stream
 * @return Log marginal likelihood
 */
pf2d_real pmmh_compute_log_likelihood(const pf2d_real *observations, int n_obs,
                                       const PMMHParams *params,
                                       pf2d_real theta_vol, pf2d_real rho,
                                       int n_particles, VSLStreamStatePtr rng);

#ifdef __cplusplus
}
#endif

#endif /* PF2D_PMMH_H */
