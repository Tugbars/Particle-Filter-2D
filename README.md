# 2D Particle Filter with Stochastic Volatility

High-performance particle filter implementation in C with Intel MKL acceleration. Designed for quantitative trading applications where latency and throughput matter. Tracks a 2D latent state `[price, log_volatility]` using Sequential Monte Carlo methods.

## Performance

**27 μs/tick @ 37,000 ticks/sec** on Intel i9-14900KF (8 P-cores, 16 threads)

| Metric | Value |
|--------|-------|
| Mean latency | 27 μs/tick |
| P50 latency | 17 μs/tick |
| Throughput | 37,000 ticks/sec |
| Headroom | 185× over ES futures peak (200/sec) |

### Benchmark Comparison

| Implementation | Latency | Throughput | vs pf2d |
|----------------|---------|------------|---------|
| **pf2d (C/MKL)** | 27 μs | 37,000/sec | baseline |
| FilterPy (NumPy) | 744 μs | 1,343/sec | **27.6× slower** |
| particles (academic) | ~1-2 ms | ~500-1000/sec | **~40-70× slower** |

### Optimization History

| Version | Mean | P50 | Throughput | Change |
|---------|------|-----|------------|--------|
| Baseline (separate regions) | 73.7 μs | 92.1 μs | 13,574/sec | — |
| MKL vectorized path | 37.8 μs | 23.1 μs | 26,468/sec | +95% |
| Fused parallel regions | 35.7 μs | 20.7 μs | 27,995/sec | +6% |
| ICDF RNG method | 33.4 μs | 19.1 μs | 29,988/sec | +7% |
| P-core affinity (Python) | 27.0 μs | 16.7 μs | 37,000/sec | +24% |

**Total improvement: 2.7×** from baseline to final optimized version.

## Features

### Core Algorithm
- **2D State Space**: Joint tracking of price and latent log-volatility
- **Stochastic Volatility Model**: Log-vol follows Ornstein-Uhlenbeck process
- **Multi-Regime Dynamics**: Up to 8 regimes with per-regime parameters
- **Adaptive Resampling**: ESS-based threshold with volatility adaptation

### Optimizations
- **Fused Parallel Regions**: RNG + physics + weights in single OpenMP region (eliminates 3 barrier syncs)
- **ICDF Gaussian RNG**: Faster than Box-Muller on modern CPUs (~7% speedup)
- **Pre-computed Vectorized exp()**: MKL VML for batch transcendentals
- **LUT Regime Sampling**: O(1) lookup instead of O(R) search
- **P-core Affinity**: Avoids slow E-cores on Intel hybrid CPUs
- **Cache-Optimized**: Arrays sized to fit in L3 cache

### Execution Paths
| Path | Condition | Description |
|------|-----------|-------------|
| FUSED vectorized | `N ≥ 4000, !PCG` | Fastest: fused RNG+physics+weights |
| MKL thread-local | `N < 4000, !PCG` | Per-thread MKL streams |
| PCG inline | `use_pcg=1` | Lowest latency for small N |

### Python Bindings
- **ctypes-based**: Zero-copy numpy integration
- **Batch processing**: Single C call for entire time series
- **Auto-configuration**: Detects Intel hybrid CPUs, sets P-core affinity

## Requirements

- Intel MKL (Math Kernel Library) 2024+
- OpenMP
- C11 compiler (MSVC 2019+, GCC 7+, or Intel ICX)
- CMake 3.16+

```bash
# Intel oneAPI (provides MKL + compiler)
# Windows: Install from Intel website
# Linux:
source /opt/intel/oneapi/setvars.sh
```

## Build

### Windows (Visual Studio)

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release --target particle_filter_2d_shared

# Copy DLL to python directory
copy Release\particle_filter_2d.dll ..\python\
```

### Linux

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make particle_filter_2d_shared -j

# Copy library to python directory
cp libparticle_filter_2d.so ../python/
```

### Build with HIGH_N mode (50k+ particles)

```bash
cmake .. -DPF2D_HIGH_N=ON
```

## Quick Start

### C API

```c
#include "particle_filter_2d.h"

int main() {
    // Configure MKL for Intel hybrid CPU (16 P-core threads)
    pf2d_mkl_config_14900kf(0);  // 0 = not verbose
    
    // Create filter: 4000 particles, 4 regimes
    PF2D* pf = pf2d_create(4000, 4);
    
    // Configure regime parameters
    //              regime, drift,   θ_vol,  μ_vol,      σ_vol, ρ
    pf2d_set_regime_params(pf, 0, 0.0010, 0.02, -4.6052, 0.05, 0.0);  // Trend
    pf2d_set_regime_params(pf, 1, 0.0000, 0.05, -4.8283, 0.03, 0.0);  // Mean-revert
    pf2d_set_regime_params(pf, 2, 0.0000, 0.10, -3.5066, 0.10, 0.0);  // High-vol
    pf2d_set_regime_params(pf, 3, 0.0000, 0.20, -2.9957, 0.20, 0.0);  // Crisis
    
    // Initialize at price=100, log_vol=log(0.01)
    pf2d_initialize(pf, 100.0, 0.01, -4.6, 0.5);
    
    // Set regime probabilities
    PF2DRegimeProbs rp;
    double probs[4] = {0.4, 0.3, 0.2, 0.1};
    pf2d_set_regime_probs(&rp, probs, 4);
    pf2d_build_regime_lut(pf, &rp);
    
    // Warmup (eliminates first-call latency)
    pf2d_warmup(pf);
    
    // Process observations
    for (int i = 0; i < n_ticks; i++) {
        PF2DOutput out = pf2d_update(pf, prices[i], &rp);
        
        printf("t=%d: price=%.2f±%.3f, vol=%.4f, ESS=%.0f\n",
               i, out.price_mean, sqrt(out.price_variance),
               out.vol_mean, out.ess);
    }
    
    pf2d_destroy(pf);
    return 0;
}
```

### Python API

```python
from pf2d import ParticleFilter2D, create_default_filter
import numpy as np

# Create filter with default 4-regime configuration
pf = create_default_filter(n_particles=4000)
pf.initialize(price0=100.0, log_vol0=np.log(0.01))
pf.warmup()

# Option 1: Streaming updates (for real-time)
for price in price_stream:
    result = pf.update(price)
    print(f"Price: {result.price_mean:.2f}, Vol: {result.vol_mean:.4f}, ESS: {result.ess:.0f}")

# Option 2: Batch processing (for backtesting) - much faster
price_est, vol_est, ess = pf.run(prices_array)
```

## State Space Model

### State Vector

```
x_t = [price_t, log_vol_t]
```

### Dynamics (per regime r)

```
price_t    = price_{t-1} + drift_r + exp(log_vol_{t-1}) × ε₁
log_vol_t  = (1 - θ_r) × log_vol_{t-1} + θ_r × μ_r + σ_r × ε₂

where:
  ε₁, ε₂ ~ N(0, 1)  (generated via MKL ICDF method)
  ε₂ = ρ × ε₁ + √(1-ρ²) × ε₂'  (optional leverage correlation)
```

### Observation Model

```
y_t ~ N(price_t, σ²_obs)
```

### Regime Parameters

| Parameter | Symbol | Description | Typical Values |
|-----------|--------|-------------|----------------|
| `drift` | — | Price drift per tick | 0.0 to 0.001 |
| `theta_vol` | θ | Mean-reversion speed | 0.02 to 0.20 |
| `mu_vol` | μ | Long-run log-volatility | -4.6 (≈1% vol) |
| `sigma_vol` | σ | Vol-of-vol | 0.03 to 0.20 |
| `rho` | ρ | Price-vol correlation | -0.5 to 0.0 |

**Stability constraint:** `sigma_vol < 2 × theta_vol` (Feller condition)

## Intel Hybrid CPU Configuration

Intel 12th-14th gen CPUs have Performance (P) and Efficiency (E) cores:

| Core Type | i9-14900KF | Clock | AVX2 | Notes |
|-----------|------------|-------|------|-------|
| P-cores | 8 × 2 threads = 16 | 5.8 GHz | Full | Use these |
| E-cores | 16 × 1 thread = 16 | 4.3 GHz | Slow | Avoid |

**Problem:** Default MKL uses all 32 threads, causing E-core bottleneck.

**Solution:** `pf2d_mkl_config_14900kf()` sets:
- 16 P-core threads only
- `KMP_AFFINITY=granularity=fine,compact,1,0`
- `KMP_HW_SUBSET=8c,2t`
- AVX2 instructions (AVX-512 disabled on consumer chips)

**Impact:** +30-50% performance vs default settings.

## Key Optimizations Explained

### 1. Fused Parallel Regions

**Before (5 parallel regions):**
```
propagate:  vdExp → RNG (barrier) → physics (barrier)
weights:    log-likelihood (barrier) → subtract max (barrier) → vdExp → normalize
```

**After (2 parallel regions):**
```
fused:      vdExp → single region [RNG + physics + log-likelihood + max] → normalize
```

**Savings:** 3 barrier eliminations × ~2 μs = **~6 μs/tick**

### 2. ICDF vs Box-Muller RNG

| Method | Operations | Performance |
|--------|------------|-------------|
| Box-Muller | `log() + sqrt() + sin() + cos()` | Baseline |
| ICDF | `erfinv()` polynomial | **+7% faster** |

MKL's ICDF implementation uses optimized polynomial approximation.

### 3. Pre-computed Vectorized exp()

**Slow (per-particle):**
```c
for (i = 0; i < n; i++)
    vol = exp(log_vols[i]);  // Scalar transcendental
```

**Fast (batch VML):**
```c
vdExp(n, log_vols, vols);  // AVX2-vectorized, single call
```

### 4. LUT Regime Sampling

**Slow (O(R) per particle):**
```c
for (r = 0; r < n_regimes; r++)
    if (u < cumsum[r]) return r;
```

**Fast (O(1) lookup):**
```c
int lut_idx = (int)(u * 1023);  // 1024-entry LUT
int regime = regime_lut[lut_idx];
```

## API Reference

### Lifecycle

```c
PF2D* pf2d_create(int n_particles, int n_regimes);
void pf2d_destroy(PF2D* pf);
void pf2d_initialize(PF2D* pf, pf2d_real price0, pf2d_real price_std,
                     pf2d_real log_vol0, pf2d_real log_vol_std);
void pf2d_warmup(PF2D* pf);
```

### Configuration

```c
void pf2d_mkl_config_14900kf(int verbose);  // Intel hybrid CPU setup
void pf2d_set_regime_params(PF2D* pf, int regime, pf2d_real drift,
                            pf2d_real theta_vol, pf2d_real mu_vol,
                            pf2d_real sigma_vol, pf2d_real rho);
void pf2d_set_observation_variance(PF2D* pf, pf2d_real var);
void pf2d_set_regime_probs(PF2DRegimeProbs* rp, const double* probs, int n);
void pf2d_build_regime_lut(PF2D* pf, const PF2DRegimeProbs* rp);
```

### Update

```c
PF2DOutput pf2d_update(PF2D* pf, pf2d_real observation, const PF2DRegimeProbs* rp);

// Batch update (for Python/FFI efficiency)
void pf2d_update_batch_minimal(PF2D* pf, const pf2d_real* observations, int n,
                               const PF2DRegimeProbs* rp,
                               pf2d_real* price_means, pf2d_real* vol_means,
                               pf2d_real* ess_values);
```

### Output Structure

```c
typedef struct {
    pf2d_real price_mean;        // E[price]
    pf2d_real price_variance;    // Var[price]
    pf2d_real log_vol_mean;      // E[log_vol]
    pf2d_real log_vol_variance;  // Var[log_vol]
    pf2d_real vol_mean;          // E[exp(log_vol)]
    pf2d_real ess;               // Effective sample size
    pf2d_real regime_probs[8];   // Posterior regime distribution
    int dominant_regime;         // argmax regime
    int resampled;               // Whether resampling occurred
} PF2DOutput;
```

## Python Bindings

### Installation

```bash
# 1. Build the DLL/SO (see Build section)
# 2. Copy to python/ directory
# 3. Import
from pf2d import ParticleFilter2D, create_default_filter
```

### Features

- **Auto MKL config**: Calls `pf2d_mkl_config_14900kf()` on import
- **Batch processing**: `pf.run(observations)` - single C call
- **Context manager**: `with ParticleFilter2D() as pf:`
- **NumPy integration**: Zero-copy array passing

### Performance: Python vs C

| Method | Latency | Notes |
|--------|---------|-------|
| C direct call | 27 μs | Native performance |
| Python batch (`pf.run()`) | 27 μs | Single C call |
| Python loop (`pf.update()`) | ~100 μs | ctypes overhead per call |

**Recommendation:** Use `pf.run()` for backtesting, `pf.update()` only for streaming.

## HIGH_N Mode (50k+ particles)

For offline analysis with large particle counts:

```bash
cmake .. -DPF2D_HIGH_N=ON
```

Enables:
- Parallel prefix sum (cumulative sum)
- Parallel batch binary search
- Parallel gather in resampling

| Particles | Standard | HIGH_N |
|-----------|----------|--------|
| 4,000 | 27 μs | 35 μs (overhead) |
| 50,000 | timeout | 120 μs |
| 100,000 | timeout | 230 μs |

## Files

```
Particle-Filter-2D/
├── CMakeLists.txt
├── build_and_deploy.bat      # Windows build script
├── 2D/
│   ├── particle_filter_2d.h  # Header
│   ├── particle_filter_2d.c  # Implementation
│   ├── particle_filter_2d.def # DLL exports (Windows)
│   └── example_usage_2d.c    # Benchmark
├── python/
│   ├── pf2d.py               # Python bindings
│   ├── benchmark_comparison.py
│   └── particle_filter_2d.dll # (copied after build)
└── mkl_config.h              # Intel hybrid CPU config
```

## References

- Gordon, Salmond, Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation.
- Doucet, Johansen (2009). A tutorial on particle filtering and smoothing.
- Chopin, Papaspiliopoulos (2020). An Introduction to Sequential Monte Carlo.
- O'Neill (2014). PCG: A family of simple fast space-efficient statistically good algorithms for random number generation.

## License

MIT License
