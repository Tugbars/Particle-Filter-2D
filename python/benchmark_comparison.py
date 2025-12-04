"""
Benchmark: pf2d vs particles (Nicolas Chopin's SMC library)

Compare our MKL-optimized implementation against the academic reference.

Install particles: pip install particles
"""

# CRITICAL: Set thread config BEFORE any imports that load MKL
import os
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'

import numpy as np
import time
import sys

# =============================================================================
# Generate test data (stochastic volatility model)
# =============================================================================

def generate_sv_data(T=1000, seed=42):
    """
    Generate stochastic volatility data matching pf2d.py test.
    
    Model matches default regime 0:
        log_vol_t = 0.95 * log_vol_{t-1} + 0.05 * log(0.01) + 0.05 * eps
        price_t = price_{t-1} + vol_t * eta
    """
    np.random.seed(seed)
    
    price = 100.0
    true_vol = 0.01
    
    prices = [price]
    vols = [true_vol]
    
    for _ in range(T - 1):
        # OU process for log-volatility (matches regime 0: theta=0.02, mu=-4.6, sigma=0.05)
        log_vol = 0.95 * np.log(true_vol) + 0.05 * np.log(0.01) + 0.05 * np.random.randn()
        true_vol = np.exp(log_vol)
        
        # Price update
        price += true_vol * np.random.randn()
        
        prices.append(price)
        vols.append(true_vol)
    
    return np.array(prices), np.array(vols)


# =============================================================================
# Benchmark: particles library
# =============================================================================

def benchmark_particles(y, N=4000, n_runs=5):
    """Benchmark Nicolas Chopin's particles library."""
    try:
        import particles
        from particles import state_space_models as ssm
    except ImportError:
        print("particles not installed. Run: pip install particles")
        return None, None
    
    # Define stochastic volatility model
    class StochVol(ssm.StateSpaceModel):
        default_params = {'mu': -4.6, 'phi': 0.98, 'sigma': 0.15}
        
        def PX0(self):
            return particles.distributions.Normal(
                loc=self.mu,
                scale=self.sigma / np.sqrt(1 - self.phi**2)
            )
        
        def PX(self, t, xp):
            return particles.distributions.Normal(
                loc=self.mu + self.phi * (xp - self.mu),
                scale=self.sigma
            )
        
        def PY(self, t, xp, x):
            return particles.distributions.Normal(
                loc=0.0,
                scale=np.exp(x / 2)
            )
    
    model = StochVol()
    
    # Warmup
    fk = ssm.Bootstrap(ssm=model, data=y[:100])
    alg = particles.SMC(fk=fk, N=N, resampling='multinomial')
    alg.run()
    
    # Benchmark
    latencies = []
    for _ in range(n_runs):
        fk = ssm.Bootstrap(ssm=model, data=y)
        alg = particles.SMC(fk=fk, N=N, resampling='multinomial', verbose=False)
        
        start = time.perf_counter()
        alg.run()
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed)
    
    mean_time = np.mean(latencies)
    per_tick = mean_time / len(y) * 1e6  # μs
    
    return per_tick, alg


# =============================================================================
# Benchmark: pf2d (our implementation)
# =============================================================================

def benchmark_pf2d(y, N=4000, n_runs=5):
    """Benchmark our MKL-optimized implementation."""
    try:
        from pf2d import create_default_filter
    except ImportError:
        print("pf2d not found. Make sure particle_filter_2d.dll is in this directory.")
        return None, None
    
    # Use same config as pf2d.py test (4 regimes, default params)
    pf = create_default_filter(n_particles=N)
    pf.initialize(price0=100.0, log_vol0=np.log(0.01), price_spread=0.01, log_vol_spread=0.5)
    pf.warmup()
    
    # Warmup run (using batch)
    pf.run(y[:100])
    
    # Benchmark using BATCH function (single C call)
    latencies = []
    for _ in range(n_runs):
        pf.initialize(price0=100.0, log_vol0=np.log(0.01), price_spread=0.01, log_vol_spread=0.5)
        
        start = time.perf_counter()
        price_est, vol_est, ess = pf.run(y)  # Uses batch C function
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed)
    
    mean_time = np.mean(latencies)
    per_tick = mean_time / len(y) * 1e6  # μs
    
    return per_tick, pf


def benchmark_pf2d_loop(y, N=4000, n_runs=5):
    """Benchmark pf2d with Python loop (for comparison)."""
    try:
        from pf2d import create_default_filter
    except ImportError:
        return None, None
    
    pf = create_default_filter(n_particles=N)
    pf.initialize(price0=100.0, log_vol0=np.log(0.01), price_spread=0.01, log_vol_spread=0.5)
    pf.warmup()
    
    # Warmup
    for obs in y[:100]:
        pf.update(float(obs))
    
    # Benchmark with Python loop
    latencies = []
    for _ in range(n_runs):
        pf.initialize(price0=100.0, log_vol0=np.log(0.01), price_spread=0.01, log_vol_spread=0.5)
        
        start = time.perf_counter()
        for obs in y:
            pf.update(float(obs))
        elapsed = time.perf_counter() - start
        
        latencies.append(elapsed)
    
    mean_time = np.mean(latencies)
    per_tick = mean_time / len(y) * 1e6
    
    return per_tick, pf


# =============================================================================
# Benchmark: FilterPy (if available)
# =============================================================================

def benchmark_filterpy(y, N=4000, n_runs=5):
    """Benchmark FilterPy's particle filter (basic implementation)."""
    try:
        from filterpy.monte_carlo import systematic_resample
    except ImportError:
        return None, None
    
    mu = -4.6
    phi = 0.98
    sigma = 0.15
    
    def run_filter():
        # Initialize particles
        particles_x = np.random.normal(mu, sigma / np.sqrt(1 - phi**2), N)
        weights = np.ones(N) / N
        
        for obs in y:
            # Predict
            particles_x = mu + phi * (particles_x - mu) + sigma * np.random.randn(N)
            
            # Update weights
            vol = np.exp(particles_x / 2)
            log_weights = -0.5 * (obs / vol)**2 - np.log(vol)
            log_weights -= np.max(log_weights)
            weights = np.exp(log_weights)
            weights /= np.sum(weights)
            
            # Resample
            ess = 1 / np.sum(weights**2)
            if ess < N / 2:
                indices = systematic_resample(weights)
                particles_x = particles_x[indices]
                weights = np.ones(N) / N
        
        return particles_x
    
    # Warmup
    run_filter()
    
    # Benchmark
    latencies = []
    for _ in range(n_runs):
        start = time.perf_counter()
        run_filter()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
    
    mean_time = np.mean(latencies)
    per_tick = mean_time / len(y) * 1e6
    
    return per_tick, None


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("Particle Filter Benchmark: pf2d vs Academic Implementations")
    print("=" * 70)
    
    # Parameters
    T = 1000      # Time steps
    N = 4000      # Particles
    n_runs = 5    # Benchmark runs
    
    print(f"\nParameters: T={T} ticks, N={N} particles, {n_runs} runs each")
    
    # Generate data
    print("\nGenerating stochastic volatility data...")
    prices, true_vols = generate_sv_data(T=T)
    print(f"  Price range: [{prices.min():.2f}, {prices.max():.2f}]")
    print(f"  Vol range:   [{true_vols.min():.6f}, {true_vols.max():.6f}]")
    
    y = prices  # Use prices as observations
    results = {}
    
    # Benchmark pf2d (batch)
    print("\n" + "-" * 70)
    print("Benchmarking: pf2d BATCH (single C call)")
    print("-" * 70)
    pf2d_time, _ = benchmark_pf2d(y, N=N, n_runs=n_runs)
    if pf2d_time:
        results['pf2d (batch)'] = pf2d_time
        print(f"  Latency: {pf2d_time:.1f} μs/tick")
        print(f"  Throughput: {1e6/pf2d_time:.0f} ticks/sec")
    
    # Benchmark pf2d (Python loop)
    print("\n" + "-" * 70)
    print("Benchmarking: pf2d LOOP (Python ctypes overhead)")
    print("-" * 70)
    pf2d_loop_time, _ = benchmark_pf2d_loop(y, N=N, n_runs=n_runs)
    if pf2d_loop_time:
        results['pf2d (loop)'] = pf2d_loop_time
        print(f"  Latency: {pf2d_loop_time:.1f} μs/tick")
        print(f"  Throughput: {1e6/pf2d_loop_time:.0f} ticks/sec")
    
    # Benchmark particles
    print("\n" + "-" * 70)
    print("Benchmarking: particles (Nicolas Chopin)")
    print("-" * 70)
    particles_time, _ = benchmark_particles(y, N=N, n_runs=n_runs)
    if particles_time:
        results['particles'] = particles_time
        print(f"  Latency: {particles_time:.1f} μs/tick")
        print(f"  Throughput: {1e6/particles_time:.0f} ticks/sec")
    
    # Benchmark FilterPy
    print("\n" + "-" * 70)
    print("Benchmarking: FilterPy (basic numpy)")
    print("-" * 70)
    filterpy_time, _ = benchmark_filterpy(y, N=N, n_runs=n_runs)
    if filterpy_time:
        results['filterpy'] = filterpy_time
        print(f"  Latency: {filterpy_time:.1f} μs/tick")
        print(f"  Throughput: {1e6/filterpy_time:.0f} ticks/sec")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Library':<20} {'Latency (μs)':<15} {'Throughput':<15} {'vs pf2d':<10}")
    print("-" * 60)
    
    baseline = results.get('pf2d (batch)', 1)
    for name, latency in sorted(results.items(), key=lambda x: x[1]):
        throughput = f"{1e6/latency:.0f}/sec"
        speedup = latency / baseline if baseline else 1
        if name == 'pf2d (batch)':
            speedup_str = "baseline"
        elif speedup > 1:
            speedup_str = f"{speedup:.1f}x slower"
        else:
            speedup_str = f"{1/speedup:.1f}x faster"
        print(f"{name:<20} {latency:<15.1f} {throughput:<15} {speedup_str:<10}")
    
    if 'pf2d (batch)' in results and 'particles' in results:
        speedup = results['particles'] / results['pf2d (batch)']
        print(f"\npf2d is {speedup:.0f}x faster than particles (academic reference)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()