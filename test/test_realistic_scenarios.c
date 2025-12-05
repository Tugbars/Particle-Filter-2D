/**
 * @file test_realistic_scenarios.c
 * @brief Realistic scenario tests for BOCPD + PMMH integration
 *
 * Tests against real-world-like scenarios:
 *   1. Flash crash (sudden vol spike, quick recovery)
 *   2. Fed announcement (scheduled event, vol crush after)
 *   3. Earnings gap (overnight gap, elevated vol)
 *   4. Liquidity crisis (persistent high vol, negative drift)
 *   5. Gradual regime shift (slow transition)
 *
 * Each scenario runs Monte Carlo simulations to validate:
 *   - Detection delay distribution
 *   - False positive rate
 *   - Parameter recovery accuracy
 *
 * Compile: gcc -O2 test_realistic_scenarios.c -o test_scenarios -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

/*============================================================================
 * CONFIGURATION
 *============================================================================*/

#define N_MONTE_CARLO       50      /* Simulations per scenario */
#define N_OBSERVATIONS      800     /* Ticks per simulation */
#define PMMH_ITERATIONS     300
#define PMMH_BURNIN         100
#define PMMH_WINDOW         150

/*============================================================================
 * RNG (xorshift128+)
 *============================================================================*/

static uint64_t rng_s[2];

static void rng_seed(uint64_t seed) {
    rng_s[0] = seed;
    rng_s[1] = seed ^ 0xDEADBEEFCAFEBABEULL;
    /* Warmup */
    for (int i = 0; i < 20; i++) {
        rng_s[0] ^= rng_s[0] << 13;
        rng_s[0] ^= rng_s[0] >> 7;
        rng_s[0] ^= rng_s[0] << 17;
    }
}

static inline uint64_t rng_next(void) {
    uint64_t s1 = rng_s[0];
    const uint64_t s0 = rng_s[1];
    rng_s[0] = s0;
    s1 ^= s1 << 23;
    rng_s[1] = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    return rng_s[1] + s0;
}

static inline double randu(void) {
    return (rng_next() >> 11) * (1.0 / 9007199254740992.0);
}

static inline double randn(void) {
    double u1 = randu(), u2 = randu();
    if (u1 < 1e-10) u1 = 1e-10;
    return sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2);
}

/* Poisson random variable (for jump arrivals) */
static int rand_poisson(double lambda) {
    double L = exp(-lambda);
    double p = 1.0;
    int k = 0;
    do {
        k++;
        p *= randu();
    } while (p > L);
    return k - 1;
}

/* Student-t random variable (for fat tails) */
static double rand_student_t(double df) {
    /* Using ratio of normals method */
    double z = randn();
    double chi2 = 0.0;
    for (int i = 0; i < (int)df; i++) {
        double g = randn();
        chi2 += g * g;
    }
    return z / sqrt(chi2 / df);
}

/*============================================================================
 * STOCHASTIC VOLATILITY MODEL WITH EXTENSIONS
 *
 * Extended Heston-like model:
 *   dS/S = μ dt + σ dW_1
 *   d(log σ) = θ(μ_v - log σ) dt + ξ dW_2 + J dN
 *
 * Where:
 *   - Corr(dW_1, dW_2) = ρ (leverage effect)
 *   - J ~ N(μ_j, σ_j) is jump size
 *   - N is Poisson process with intensity λ
 *============================================================================*/

typedef struct {
    /* Base parameters */
    double drift;           /* μ: expected return */
    double mu_vol;          /* μ_v: long-run log-volatility */
    double theta_vol;       /* θ: mean-reversion speed */
    double sigma_vol;       /* ξ: vol-of-vol */
    double rho;             /* ρ: leverage (typically -0.5 to -0.8) */
    
    /* Jump parameters */
    double jump_intensity;  /* λ: expected jumps per tick */
    double jump_mean;       /* μ_j: mean jump size in log-vol */
    double jump_std;        /* σ_j: jump size std */
    
    /* Innovation distribution */
    double student_df;      /* If > 0, use Student-t innovations */
    
} SVParams;

/* Default "normal market" parameters */
static SVParams sv_normal_market(void) {
    return (SVParams){
        .drift = 0.0,
        .mu_vol = -4.0,         /* exp(-4) ≈ 1.8% daily vol */
        .theta_vol = 0.02,
        .sigma_vol = 0.05,
        .rho = -0.5,            /* Leverage effect */
        .jump_intensity = 0.0,
        .jump_mean = 0.0,
        .jump_std = 0.0,
        .student_df = 0         /* Gaussian */
    };
}

/* High volatility crisis parameters */
static SVParams sv_crisis(void) {
    return (SVParams){
        .drift = -0.001,        /* Negative drift during crisis */
        .mu_vol = -2.5,         /* exp(-2.5) ≈ 8% daily vol */
        .theta_vol = 0.01,      /* Slower mean reversion */
        .sigma_vol = 0.15,      /* Higher vol-of-vol */
        .rho = -0.7,            /* Stronger leverage */
        .jump_intensity = 0.02, /* ~2% chance of jump per tick */
        .jump_mean = 0.3,       /* Jumps increase vol */
        .jump_std = 0.2,
        .student_df = 5         /* Fat tails */
    };
}

/* Generate one tick of SV process */
static void sv_step(double *log_vol, double *price, 
                    const SVParams *p, double *last_return) {
    /* Generate correlated innovations */
    double z1 = randn();
    double z2 = p->rho * z1 + sqrt(1.0 - p->rho * p->rho) * randn();
    
    /* Optional Student-t for fat tails */
    if (p->student_df > 0) {
        z1 = rand_student_t(p->student_df) / sqrt(p->student_df / (p->student_df - 2));
    }
    
    /* Vol dynamics */
    double new_log_vol = (1.0 - p->theta_vol) * (*log_vol) 
                        + p->theta_vol * p->mu_vol
                        + p->sigma_vol * z2;
    
    /* Jump in volatility */
    if (p->jump_intensity > 0 && randu() < p->jump_intensity) {
        new_log_vol += p->jump_mean + p->jump_std * randn();
    }
    
    *log_vol = new_log_vol;
    
    /* Price dynamics */
    double vol = exp(*log_vol);
    double ret = p->drift + vol * z1;
    *price = (*price) * (1.0 + ret);
    *last_return = ret;
}

/*============================================================================
 * SCENARIO DEFINITIONS
 *============================================================================*/

typedef struct {
    const char *name;
    const char *description;
    int changepoint;            /* When regime changes */
    int transition_ticks;       /* 0 = instant, >0 = gradual */
    SVParams before;
    SVParams after;
    
    /* For complex scenarios */
    int has_second_change;
    int second_changepoint;
    SVParams final;
} Scenario;

/* Scenario 1: Flash Crash */
static Scenario scenario_flash_crash(void) {
    SVParams normal = sv_normal_market();
    SVParams crash = sv_crisis();
    crash.drift = -0.003;       /* Sharp negative drift */
    crash.mu_vol = -2.0;        /* Very high vol */
    crash.jump_intensity = 0.05;/* Frequent jumps */
    
    SVParams recovery = sv_normal_market();
    recovery.mu_vol = -3.5;     /* Slightly elevated post-crash */
    
    return (Scenario){
        .name = "Flash Crash",
        .description = "Sudden vol spike with quick recovery",
        .changepoint = 300,
        .transition_ticks = 0,
        .before = normal,
        .after = crash,
        .has_second_change = 1,
        .second_changepoint = 350,  /* Recovery after 50 ticks */
        .final = recovery
    };
}

/* Scenario 2: Fed Announcement */
static Scenario scenario_fed_announcement(void) {
    SVParams pre_fed = sv_normal_market();
    pre_fed.sigma_vol = 0.08;   /* Vol-of-vol rises before announcement */
    
    SVParams post_fed = sv_normal_market();
    post_fed.mu_vol = -3.0;     /* Vol spike on announcement */
    post_fed.drift = 0.0005;    /* Slight positive drift (dovish Fed) */
    
    SVParams crush = sv_normal_market();
    crush.mu_vol = -4.5;        /* Vol crush as uncertainty resolves */
    crush.sigma_vol = 0.03;
    
    return (Scenario){
        .name = "Fed Announcement",
        .description = "Scheduled event with vol spike then crush",
        .changepoint = 400,
        .transition_ticks = 0,
        .before = pre_fed,
        .after = post_fed,
        .has_second_change = 1,
        .second_changepoint = 450,
        .final = crush
    };
}

/* Scenario 3: Earnings Surprise */
static Scenario scenario_earnings_gap(void) {
    SVParams normal = sv_normal_market();
    
    SVParams gap = sv_normal_market();
    gap.drift = 0.002;          /* Positive surprise */
    gap.mu_vol = -3.0;          /* Elevated vol */
    gap.sigma_vol = 0.10;
    gap.jump_intensity = 0.01;
    
    return (Scenario){
        .name = "Earnings Surprise",
        .description = "Gap up with elevated post-earnings vol",
        .changepoint = 350,
        .transition_ticks = 5,  /* Quick but not instant */
        .before = normal,
        .after = gap,
        .has_second_change = 0
    };
}

/* Scenario 4: Liquidity Crisis */
static Scenario scenario_liquidity_crisis(void) {
    SVParams normal = sv_normal_market();
    
    SVParams crisis = sv_crisis();
    crisis.theta_vol = 0.005;   /* Very slow mean reversion */
    crisis.rho = -0.8;          /* Strong leverage */
    
    return (Scenario){
        .name = "Liquidity Crisis",
        .description = "Persistent high vol, negative drift, slow recovery",
        .changepoint = 300,
        .transition_ticks = 50, /* Gradual onset */
        .before = normal,
        .after = crisis,
        .has_second_change = 0
    };
}

/* Scenario 5: Gradual Trend Change */
static Scenario scenario_gradual_shift(void) {
    SVParams bull = sv_normal_market();
    bull.drift = 0.0005;
    bull.mu_vol = -4.2;
    
    SVParams bear = sv_normal_market();
    bear.drift = -0.0003;
    bear.mu_vol = -3.5;
    bear.sigma_vol = 0.08;
    
    return (Scenario){
        .name = "Gradual Regime Shift",
        .description = "Slow transition from bull to bear market",
        .changepoint = 300,
        .transition_ticks = 100, /* Very gradual */
        .before = bull,
        .after = bear,
        .has_second_change = 0
    };
}

/*============================================================================
 * DATA GENERATION
 *============================================================================*/

/* Interpolate between two SVParams (for gradual transitions) */
static SVParams sv_interpolate(const SVParams *a, const SVParams *b, double t) {
    SVParams p;
    p.drift = a->drift + t * (b->drift - a->drift);
    p.mu_vol = a->mu_vol + t * (b->mu_vol - a->mu_vol);
    p.theta_vol = a->theta_vol + t * (b->theta_vol - a->theta_vol);
    p.sigma_vol = a->sigma_vol + t * (b->sigma_vol - a->sigma_vol);
    p.rho = a->rho + t * (b->rho - a->rho);
    p.jump_intensity = a->jump_intensity + t * (b->jump_intensity - a->jump_intensity);
    p.jump_mean = a->jump_mean + t * (b->jump_mean - a->jump_mean);
    p.jump_std = a->jump_std + t * (b->jump_std - a->jump_std);
    p.student_df = a->student_df + t * (b->student_df - a->student_df);
    return p;
}

static void generate_scenario_data(const Scenario *s, double *prices, 
                                    double *returns, int n) {
    prices[0] = 100.0;
    double log_vol = s->before.mu_vol;
    
    for (int t = 1; t < n; t++) {
        SVParams params;
        
        /* Determine current regime */
        if (t < s->changepoint) {
            params = s->before;
        } else if (s->transition_ticks > 0 && t < s->changepoint + s->transition_ticks) {
            /* Gradual transition */
            double frac = (double)(t - s->changepoint) / s->transition_ticks;
            params = sv_interpolate(&s->before, &s->after, frac);
        } else if (s->has_second_change && t >= s->second_changepoint) {
            params = s->final;
        } else {
            params = s->after;
        }
        
        sv_step(&log_vol, &prices[t], &params, &returns[t-1]);
        if (t == 1) prices[1] = prices[0] * (1.0 + returns[0]);
    }
    returns[n-1] = 0.0;
}

/*============================================================================
 * CHANGEPOINT DETECTOR
 *============================================================================*/

typedef struct {
    double *buffer;
    int window;
    int head;
    int count;
    double baseline_var;
    int baseline_set;
    int warmup_end;
} Detector;

static Detector* detector_create(int window, int warmup) {
    Detector *d = (Detector*)calloc(1, sizeof(Detector));
    d->window = window;
    d->warmup_end = warmup;
    d->buffer = (double*)calloc(window, sizeof(double));
    return d;
}

static void detector_destroy(Detector *d) {
    if (d) { free(d->buffer); free(d); }
}

static void detector_reset(Detector *d) {
    d->head = 0;
    d->count = 0;
    d->baseline_set = 0;
    d->baseline_var = 0;
}

static double compute_var(const double *buf, int n) {
    if (n < 2) return 0;
    double sum = 0, sum2 = 0;
    for (int i = 0; i < n; i++) {
        sum += buf[i];
        sum2 += buf[i] * buf[i];
    }
    return (sum2 - sum * sum / n) / (n - 1);
}

/* Returns variance ratio, or 0 if not ready */
static double detector_update(Detector *d, double x, int t) {
    d->buffer[d->head] = x;
    d->head = (d->head + 1) % d->window;
    if (d->count < d->window) d->count++;
    
    if (d->count < d->window) return 0.0;
    
    if (!d->baseline_set && t >= d->warmup_end) {
        d->baseline_var = compute_var(d->buffer, d->count);
        d->baseline_set = 1;
        if (d->baseline_var < 1e-12) d->baseline_var = 1e-12;
    }
    
    if (!d->baseline_set) return 0.0;
    
    double current_var = compute_var(d->buffer, d->count);
    return current_var / d->baseline_var;
}

/*============================================================================
 * PMMH (simplified for testing)
 *============================================================================*/

typedef struct {
    double drift, mu_vol, sigma_vol;
} PMMHParams;

typedef struct {
    PMMHParams mean, std;
} PMMHPrior;

typedef struct {
    PMMHParams posterior;
    double acceptance_rate;
} PMMHResult;

static double pmmh_log_lik(const double *ret, int n, const PMMHParams *p, double theta) {
    int np = 64;
    double *lv = (double*)malloc(np * sizeof(double));
    double *lv2 = (double*)malloc(np * sizeof(double));
    double *w = (double*)malloc(np * sizeof(double));
    
    for (int i = 0; i < np; i++) lv[i] = p->mu_vol + randn() * 0.2;
    
    double ll = 0;
    double one_m_theta = 1.0 - theta;
    double theta_mu = theta * p->mu_vol;
    
    for (int t = 0; t < n; t++) {
        double max_lw = -1e30;
        for (int i = 0; i < np; i++) {
            lv2[i] = one_m_theta * lv[i] + theta_mu + p->sigma_vol * randn();
            double vol = exp(lv2[i]);
            double z = (ret[t] - p->drift) / vol;
            w[i] = -0.5 * z * z - lv2[i];
            if (w[i] > max_lw) max_lw = w[i];
        }
        
        double sum_w = 0;
        for (int i = 0; i < np; i++) {
            w[i] = exp(w[i] - max_lw);
            sum_w += w[i];
        }
        ll += max_lw + log(sum_w / np);
        
        /* Resample */
        double *tmp = (double*)malloc(np * sizeof(double));
        for (int i = 0; i < np; i++) {
            double u = randu() * sum_w;
            double cum = 0;
            int j = 0;
            while (j < np - 1 && cum + w[j] < u) { cum += w[j]; j++; }
            tmp[i] = lv2[j];
        }
        memcpy(lv, tmp, np * sizeof(double));
        free(tmp);
    }
    
    free(lv); free(lv2); free(w);
    return ll;
}

static void pmmh_run(const double *ret, int n, const PMMHPrior *prior,
                     double theta, int n_iter, int n_burn, PMMHResult *res) {
    PMMHParams cur = prior->mean;
    double cur_ll = pmmh_log_lik(ret, n, &cur, theta);
    
    int n_acc = 0;
    double sum_d = 0, sum_m = 0, sum_s = 0;
    int n_samples = 0;
    
    for (int it = 0; it < n_iter; it++) {
        PMMHParams prop;
        prop.drift = cur.drift + randn() * 0.0003;
        prop.mu_vol = cur.mu_vol + randn() * 0.05;
        prop.sigma_vol = cur.sigma_vol * exp(randn() * 0.03);
        
        prop.drift = fmax(-0.01, fmin(0.01, prop.drift));
        prop.mu_vol = fmax(-8, fmin(0, prop.mu_vol));
        prop.sigma_vol = fmax(0.01, fmin(0.5, prop.sigma_vol));
        
        double prop_ll = pmmh_log_lik(ret, n, &prop, theta);
        
        if (log(randu()) < prop_ll - cur_ll) {
            cur = prop;
            cur_ll = prop_ll;
            n_acc++;
        }
        
        if (it >= n_burn) {
            sum_d += cur.drift;
            sum_m += cur.mu_vol;
            sum_s += cur.sigma_vol;
            n_samples++;
        }
    }
    
    res->posterior.drift = sum_d / n_samples;
    res->posterior.mu_vol = sum_m / n_samples;
    res->posterior.sigma_vol = sum_s / n_samples;
    res->acceptance_rate = (double)n_acc / n_iter;
}

/*============================================================================
 * MONTE CARLO EVALUATION
 *============================================================================*/

typedef struct {
    double mean_detection_delay;
    double std_detection_delay;
    double false_positive_rate;
    double false_negative_rate;
    double mu_vol_rmse;
    double mu_vol_bias;
    double drift_rmse;
    double sigma_vol_rmse;
    int n_runs;
} MCStats;

static void run_scenario_monte_carlo(const Scenario *s, int n_runs, MCStats *stats) {
    double *prices = (double*)malloc(N_OBSERVATIONS * sizeof(double));
    double *returns = (double*)malloc(N_OBSERVATIONS * sizeof(double));
    
    double sum_delay = 0, sum_delay2 = 0;
    int n_detected = 0, n_false_pos = 0, n_false_neg = 0;
    double sum_mu_err = 0, sum_mu_err2 = 0;
    double sum_d_err2 = 0, sum_s_err2 = 0;
    
    int detect_window = 40;
    int warmup = 200;
    double ratio_thresh = 2.5;
    
    for (int run = 0; run < n_runs; run++) {
        rng_seed(12345 + run * 7919);
        
        generate_scenario_data(s, prices, returns, N_OBSERVATIONS);
        
        /* Run detector */
        Detector *det = detector_create(detect_window, warmup);
        int detection_time = -1;
        
        for (int t = 0; t < N_OBSERVATIONS - 1; t++) {
            double ratio = detector_update(det, returns[t], t);
            if (detection_time < 0 && ratio > ratio_thresh && t > warmup + 50) {
                detection_time = t;
            }
        }
        detector_destroy(det);
        
        /* Evaluate detection */
        int true_cp = s->changepoint;
        int tolerance = s->transition_ticks > 0 ? s->transition_ticks + 30 : 50;
        
        if (detection_time >= 0) {
            int delay = detection_time - true_cp;
            if (delay >= -20 && delay <= tolerance + 50) {
                /* True positive */
                sum_delay += delay;
                sum_delay2 += delay * delay;
                n_detected++;
            } else if (delay < -20) {
                /* False positive (detected before change) */
                n_false_pos++;
            }
        } else {
            n_false_neg++;
        }
        
        /* Run PMMH if detected */
        if (detection_time >= 0 && detection_time + PMMH_WINDOW < N_OBSERVATIONS) {
            PMMHPrior prior;
            prior.mean = (PMMHParams){s->before.drift, s->before.mu_vol, s->before.sigma_vol};
            prior.std = (PMMHParams){0.002, 0.5, 0.5};
            
            PMMHResult res;
            pmmh_run(&returns[detection_time], PMMH_WINDOW, &prior, 
                     s->after.theta_vol, PMMH_ITERATIONS, PMMH_BURNIN, &res);
            
            /* Compare to true post-change params */
            double true_mu = s->after.mu_vol;
            double true_d = s->after.drift;
            double true_s = s->after.sigma_vol;
            
            double mu_err = res.posterior.mu_vol - true_mu;
            sum_mu_err += mu_err;
            sum_mu_err2 += mu_err * mu_err;
            
            sum_d_err2 += (res.posterior.drift - true_d) * (res.posterior.drift - true_d);
            sum_s_err2 += (res.posterior.sigma_vol - true_s) * (res.posterior.sigma_vol - true_s);
        }
    }
    
    /* Compute statistics */
    stats->n_runs = n_runs;
    
    if (n_detected > 0) {
        stats->mean_detection_delay = sum_delay / n_detected;
        stats->std_detection_delay = sqrt(sum_delay2 / n_detected - 
                                          stats->mean_detection_delay * stats->mean_detection_delay);
        stats->mu_vol_bias = sum_mu_err / n_detected;
        stats->mu_vol_rmse = sqrt(sum_mu_err2 / n_detected);
        stats->drift_rmse = sqrt(sum_d_err2 / n_detected);
        stats->sigma_vol_rmse = sqrt(sum_s_err2 / n_detected);
    }
    
    stats->false_positive_rate = (double)n_false_pos / n_runs;
    stats->false_negative_rate = (double)n_false_neg / n_runs;
    
    free(prices);
    free(returns);
}

/*============================================================================
 * MAIN
 *============================================================================*/

static void print_scenario_results(const Scenario *s, const MCStats *stats) {
    printf("\n┌─────────────────────────────────────────────────────────────────┐\n");
    printf("│ %-63s │\n", s->name);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ %-63s │\n", s->description);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ True changepoint: t=%d", s->changepoint);
    if (s->transition_ticks > 0) printf(" (gradual, %d ticks)", s->transition_ticks);
    printf("%*s│\n", s->transition_ticks > 0 ? 24 : 41, "");
    printf("│ Before: μ_v=%.2f, σ_v=%.3f, drift=%.5f                       │\n",
           s->before.mu_vol, s->before.sigma_vol, s->before.drift);
    printf("│ After:  μ_v=%.2f, σ_v=%.3f, drift=%.5f                       │\n",
           s->after.mu_vol, s->after.sigma_vol, s->after.drift);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ DETECTION PERFORMANCE (n=%d)                                    │\n", stats->n_runs);
    printf("│   Mean delay:      %+6.1f ticks (std=%.1f)                       │\n",
           stats->mean_detection_delay, stats->std_detection_delay);
    printf("│   False positive:  %5.1f%%                                       │\n",
           100.0 * stats->false_positive_rate);
    printf("│   False negative:  %5.1f%%                                       │\n",
           100.0 * stats->false_negative_rate);
    printf("├─────────────────────────────────────────────────────────────────┤\n");
    printf("│ PMMH PARAMETER RECOVERY                                         │\n");
    printf("│   μ_vol:     RMSE=%.3f, bias=%+.3f                             │\n",
           stats->mu_vol_rmse, stats->mu_vol_bias);
    printf("│   drift:     RMSE=%.6f                                        │\n",
           stats->drift_rmse);
    printf("│   σ_vol:     RMSE=%.4f                                          │\n",
           stats->sigma_vol_rmse);
    printf("└─────────────────────────────────────────────────────────────────┘\n");
}

int main(void) {
    printf("╔═════════════════════════════════════════════════════════════════╗\n");
    printf("║     REALISTIC SCENARIO TESTS: BOCPD + PMMH INTEGRATION          ║\n");
    printf("║                                                                 ║\n");
    printf("║  Running %d Monte Carlo simulations per scenario...            ║\n", N_MONTE_CARLO);
    printf("╚═════════════════════════════════════════════════════════════════╝\n");
    
    Scenario scenarios[] = {
        scenario_flash_crash(),
        scenario_fed_announcement(),
        scenario_earnings_gap(),
        scenario_liquidity_crisis(),
        scenario_gradual_shift()
    };
    int n_scenarios = sizeof(scenarios) / sizeof(scenarios[0]);
    
    MCStats all_stats[5];
    
    for (int i = 0; i < n_scenarios; i++) {
        printf("\nRunning: %s", scenarios[i].name);
        fflush(stdout);
        
        run_scenario_monte_carlo(&scenarios[i], N_MONTE_CARLO, &all_stats[i]);
        
        printf(" ✓\n");
        print_scenario_results(&scenarios[i], &all_stats[i]);
    }
    
    /* Summary */
    printf("\n╔═════════════════════════════════════════════════════════════════╗\n");
    printf("║                          SUMMARY                                ║\n");
    printf("╠═════════════════════════════════════════════════════════════════╣\n");
    printf("║ Scenario             │ Delay │ FP%%  │ FN%%  │ μ_v RMSE          ║\n");
    printf("╠══════════════════════╪═══════╪══════╪══════╪═══════════════════╣\n");
    
    for (int i = 0; i < n_scenarios; i++) {
        printf("║ %-20s │ %+5.0f │ %4.1f │ %4.1f │ %.3f             ║\n",
               scenarios[i].name,
               all_stats[i].mean_detection_delay,
               100.0 * all_stats[i].false_positive_rate,
               100.0 * all_stats[i].false_negative_rate,
               all_stats[i].mu_vol_rmse);
    }
    
    printf("╚══════════════════════╧═══════╧══════╧══════╧═══════════════════╝\n");
    
    return 0;
}
