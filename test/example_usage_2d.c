/**
 * @file example_usage_2d.c
 * @brief Example usage of 2D Particle Filter with stochastic volatility
 *
 * Demonstrates: SSA → BOCPD → PF2D → Kelly pipeline
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include "particle_filter_2d.h"
#include "pf2d_kelly_interface.h"

/*============================================================================
 * SIMULATED MARKET DATA
 *============================================================================*/

typedef struct {
    pf2d_real price;
    pf2d_real true_vol;
    int true_regime;
} MarketState;

/* Simulate regime-switching stochastic volatility process */
static void simulate_tick(MarketState* m, pf2d_pcg32_t* rng) {
    /* Regime transition (simplified) */
    pf2d_real u = pf2d_pcg32_uniform(rng);
    if (u < 0.01) {
        /* 1% chance of regime change */
        m->true_regime = (m->true_regime + 1) % 4;
    }
    
    /* Regime-dependent dynamics */
    pf2d_real drift, theta_v, mu_v, sigma_v;
    switch (m->true_regime) {
        case 0:  /* Trend */
            drift = 0.001; theta_v = 0.02; mu_v = log(0.01); sigma_v = 0.05;
            break;
        case 1:  /* Mean-revert */
            drift = 0.0; theta_v = 0.05; mu_v = log(0.008); sigma_v = 0.03;
            break;
        case 2:  /* High-vol */
            drift = 0.0; theta_v = 0.10; mu_v = log(0.03); sigma_v = 0.10;
            break;
        case 3:  /* Jump */
            drift = 0.0; theta_v = 0.20; mu_v = log(0.05); sigma_v = 0.20;
            break;
        default:
            drift = 0.0; theta_v = 0.05; mu_v = log(0.01); sigma_v = 0.05;
    }
    
    /* Update true volatility */
    pf2d_real lv = log(m->true_vol);
    lv = (1.0 - theta_v) * lv + theta_v * mu_v + sigma_v * pf2d_pcg32_gaussian(rng);
    m->true_vol = exp(lv);
    
    /* Update price */
    m->price = m->price + drift + m->true_vol * pf2d_pcg32_gaussian(rng);
}

/*============================================================================
 * MAIN
 *============================================================================*/

int main(int argc, char* argv[]) {
    int n_particles = 4000;
    int n_ticks = 10000;
    
    if (argc > 1) n_particles = atoi(argv[1]);
    if (argc > 2) n_ticks = atoi(argv[2]);
    
    printf("=== 2D Particle Filter with Stochastic Volatility ===\n");
    printf("Particles: %d, Ticks: %d\n\n", n_particles, n_ticks);
    
    /* Create PF2D */
    PF2D* pf = pf2d_create(n_particles, 4);
    if (!pf) {
        fprintf(stderr, "Failed to create particle filter\n");
        return 1;
    }
    
    /* Configure regimes (matching simulation) */
    pf2d_set_regime_params(pf, 0, 0.001, 0.02, log(0.01), 0.05, 0.0);  /* Trend */
    pf2d_set_regime_params(pf, 1, 0.0,   0.05, log(0.008), 0.03, 0.0); /* Mean-revert */
    pf2d_set_regime_params(pf, 2, 0.0,   0.10, log(0.03), 0.10, 0.0);  /* High-vol */
    pf2d_set_regime_params(pf, 3, 0.0,   0.20, log(0.05), 0.20, 0.0);  /* Jump */
    
    /* Initialize */
    pf2d_real initial_price = 100.0;
    pf2d_real initial_log_vol = log(0.01);
    pf2d_initialize(pf, initial_price, 1.0, initial_log_vol, 0.1);
    
    /* Enable PCG and adaptive resampling */
    pf2d_enable_pcg(pf, 1);
    pf2d_set_resample_adaptive(pf, 0.01);
    
    pf2d_print_config(pf);
    printf("\n");
    
    /* Setup regime probabilities (from BOCPD in real system) */
    PF2DRegimeProbs rp;
    pf2d_real probs[4] = {0.4, 0.3, 0.2, 0.1};
    pf2d_set_regime_probs(&rp, probs, 4);
    pf2d_build_regime_lut(pf, &rp);
    
    /* Kelly tracker */
    PF2DKellyTracker kelly_tracker;
    pf2d_kelly_tracker_init(&kelly_tracker);
    
    /* Simulated market state */
    MarketState market = {
        .price = initial_price,
        .true_vol = 0.01,
        .true_regime = 0
    };
    pf2d_pcg32_t sim_rng;
    pf2d_pcg32_seed(&sim_rng, 12345, 0);
    
    /* Tracking stats */
    double total_time = 0.0;
    double price_rmse = 0.0;
    double vol_rmse = 0.0;
    int resample_count = 0;
    double total_position = 0.0;
    
    /* Warmup */
    for (int t = 0; t < 100; t++) {
        simulate_tick(&market, &sim_rng);
        pf2d_update(pf, market.price, &rp);
    }
    
    printf("Running %d ticks...\n\n", n_ticks);
    
    /* Main loop */
    clock_t start = clock();
    
    for (int t = 0; t < n_ticks; t++) {
        /* Simulate market */
        simulate_tick(&market, &sim_rng);
        
        /* Update PF */
        PF2DOutput out = pf2d_update(pf, market.price, &rp);
        
        /* Kelly sizing */
        double position = pf2d_to_kelly(&kelly_tracker, &out, n_particles, INFINITY, 0.5);
        total_position += fabs(position);
        
        /* Track errors */
        double price_err = out.price_mean - market.price;
        double vol_err = out.vol_mean - market.true_vol;
        price_rmse += price_err * price_err;
        vol_rmse += vol_err * vol_err;
        
        if (out.resampled) resample_count++;
        
        /* Print progress */
        if ((t + 1) % 2000 == 0 || t == 0) {
            printf("Tick %5d: price=%.4f (est=%.4f, err=%.4f) "
                   "vol=%.5f (est=%.5f) ESS=%.0f regime=%d kelly=%.3f\n",
                   t + 1, (double)market.price, (double)out.price_mean, 
                   price_err, (double)market.true_vol, (double)out.vol_mean,
                   (double)out.ess, out.dominant_regime, position);
        }
    }
    
    clock_t end = clock();
    total_time = (double)(end - start) / CLOCKS_PER_SEC;
    
    /* Results */
    price_rmse = sqrt(price_rmse / n_ticks);
    vol_rmse = sqrt(vol_rmse / n_ticks);
    
    printf("\n=== Results ===\n");
    printf("Total time:       %.3f sec\n", total_time);
    printf("Time per tick:    %.2f μs\n", total_time / n_ticks * 1e6);
    printf("Price RMSE:       %.6f\n", price_rmse);
    printf("Vol RMSE:         %.6f\n", vol_rmse);
    printf("Resample rate:    %.1f%%\n", 100.0 * resample_count / n_ticks);
    printf("Avg |position|:   %.4f\n", total_position / n_ticks);
    
    /* Final state */
    printf("\n=== Final State ===\n");
    printf("True price:       %.4f\n", (double)market.price);
    printf("True vol:         %.6f\n", (double)market.true_vol);
    printf("True regime:      %d\n", market.true_regime);
    
    PF2DOutput final_out = pf2d_update(pf, market.price, &rp);
    printf("Est price:        %.4f (± %.4f)\n", 
           (double)final_out.price_mean, sqrt((double)final_out.price_variance));
    printf("Est vol:          %.6f (log_vol: %.4f ± %.4f)\n",
           (double)final_out.vol_mean, (double)final_out.log_vol_mean,
           sqrt((double)final_out.log_vol_variance));
    printf("ESS:              %.0f / %d (%.1f%%)\n",
           (double)final_out.ess, n_particles, 
           100.0 * final_out.ess / n_particles);
    
    /* Kelly bridge details */
    printf("\n=== Kelly Bridge ===\n");
    PF2DKellyBridge bridge;
    pf2d_kelly_bridge_update(&bridge, &kelly_tracker, &final_out, n_particles);
    pf2d_kelly_bridge_print(&bridge);
    
    KellyResult kelly;
    pf2d_kelly_compute_full(&bridge, INFINITY, 0.5, &kelly);
    kelly_print_result(&kelly);
    
    /* Cleanup */
    pf2d_destroy(pf);
    
    printf("\nDone.\n");
    return 0;
}
