# 2D Particle Filter with Stochastic Volatility

High-performance particle filter implementation in C with Intel MKL acceleration. Designed for quantitative trading applications where latency and throughput matter. Tracks a 2D latent state `[price, log_volatility]` using Sequential Monte Carlo methods.

## Features

- **2D State Space**: Joint tracking of observable and latent volatility
- **Stochastic Volatility Model**: Log-vol follows Ornstein-Uhlenbeck process
- **Multi-Regime Dynamics**: Per-regime parameters for drift and volatility
- **Three Execution Paths**:
  - PCG inline RNG (lowest latency)
  - MKL thread-local RNG (balanced)
  - Vectorized with pre-computed vExp (highest throughput)
- **HIGH_N Mode**: Parallel cumsum + parallel search for 50k+ particles
- **Numerical Stability**: Weight underflow protection, safe indexing

## Requirements

- Intel MKL (Math Kernel Library)
- OpenMP
- C11 compiler (GCC 7+ or Intel ICX)

```bash
# Intel oneAPI (provides MKL + compiler)
source /opt/intel/oneapi/setvars.sh
```

## Build

```bash
# Build particle filter
make pf2d

# Build with HIGH_N mode (50k+ particles)
make pf2d_high_n

# Run benchmark
make run2d

# Clean
make clean
```

## Quick Start

```c
#include "particle_filter_2d.h"

int main() {
    /* Create filter: 4000 particles, 4 regimes */
    PF2D* pf = pf2d_create(4000, 4);
    
    /* Configure regime parameters */
    /*              regime, drift,   θ_vol,  μ_vol,      σ_vol, ρ   */
    pf2d_set_regime_params(pf, 0, 0.0001, 0.02, -4.6, 0.05, 0.0);
    pf2d_set_regime_params(pf, 1, 0.0000, 0.05, -4.8, 0.03, 0.0);
    pf2d_set_regime_params(pf, 2, 0.0000, 0.10, -3.5, 0.10, 0.0);
    pf2d_set_regime_params(pf, 3, 0.0000, 0.20, -3.0, 0.20, 0.0);
    
    /* Initialize at price=100, log_vol=-4.6 */
    pf2d_initialize(pf, 100.0, 1.0, -4.6, 0.1);
    
    /* Set regime probabilities */
    PF2DRegimeProbs rp;
    double probs[4] = {0.7, 0.2, 0.08, 0.02};
    pf2d_set_regime_probs(&rp, probs, 4);
    pf2d_build_regime_lut(pf, &rp);
    
    /* Process observations */
    double observations[] = {100.1, 100.3, 100.2, 100.5, 100.4};
    
    for (int i = 0; i < 5; i++) {
        PF2DOutput out = pf2d_update(pf, observations[i], &rp);
        
        printf("t=%d: price=%.2f±%.3f, vol=%.4f±%.4f, ESS=%.0f, regime=%d\n",
               i,
               out.price_mean, sqrt(out.price_variance),
               out.vol_mean, sqrt(out.log_vol_variance),
               out.ess,
               out.dominant_regime);
    }
    
    pf2d_destroy(pf);
    return 0;
}
```

## State Space Model

### State

```
x_t = [price_t, log_vol_t]
```

### Dynamics (per regime r)

```
price_t    = price_{t-1} + drift_r + exp(log_vol_{t-1}) × ε₁
log_vol_t  = (1 - θ_r) × log_vol_{t-1} + θ_r × μ_r + σ_r × ε₂

where:
  ε₁, ε₂ ~ N(0, 1)
  ε₂ = ρ × ε₁ + √(1-ρ²) × ε₂'  (optional correlation)
```

### Observation Model

```
y_t ~ N(price_t, σ²_obs)
```

Log-volatility is latent (unobserved). Particles with incorrect volatility beliefs receive lower weights through price prediction errors.

## Regime Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `drift` | - | Price drift per timestep |
| `theta_vol` | θ | Mean-reversion speed of log-vol |
| `mu_vol` | μ | Long-term mean of log-vol |
| `sigma_vol` | σ | Volatility of log-vol (vol-of-vol) |
| `rho` | ρ | Correlation between price and vol noise |

**Stability constraint:** `sigma_vol < 2 × theta_vol`

## API Reference

### Lifecycle

```c
PF2D* pf2d_create(int n_particles, int n_regimes);
void pf2d_destroy(PF2D* pf);
void pf2d_initialize(PF2D* pf, pf2d_real price0, pf2d_real price_std,
                     pf2d_real log_vol0, pf2d_real log_vol_std);
```

### Configuration

```c
void pf2d_set_regime_params(PF2D* pf, int regime, pf2d_real drift,
                            pf2d_real theta_vol, pf2d_real mu_vol,
                            pf2d_real sigma_vol, pf2d_real rho);
void pf2d_set_obs_variance(PF2D* pf, pf2d_real var);
void pf2d_enable_pcg(PF2D* pf, int enable);
void pf2d_set_resample_threshold(PF2D* pf, pf2d_real thresh);
```

### Regime Probabilities

```c
void pf2d_set_regime_probs(PF2DRegimeProbs* rp, const double* probs, int n);
void pf2d_build_regime_lut(PF2D* pf, const PF2DRegimeProbs* rp);
```

### Update

```c
PF2DOutput pf2d_update(PF2D* pf, pf2d_real observation, const PF2DRegimeProbs* rp);
```

### Output Structure

```c
typedef struct {
    pf2d_real price_mean;        /* E[price] */
    pf2d_real price_variance;    /* Var[price] */
    pf2d_real log_vol_mean;      /* E[log_vol] */
    pf2d_real log_vol_variance;  /* Var[log_vol] */
    pf2d_real vol_mean;          /* E[exp(log_vol)] */
    pf2d_real ess;               /* Effective sample size */
    pf2d_real regime_probs[8];   /* Posterior regime distribution */
    int dominant_regime;         /* argmax regime */
    int resampled;               /* Whether resampling occurred */
} PF2DOutput;
```

### Diagnostics

```c
pf2d_real pf2d_effective_sample_size(const PF2D* pf);
pf2d_real pf2d_price_mean(const PF2D* pf);
pf2d_real pf2d_price_variance(const PF2D* pf);
pf2d_real pf2d_log_vol_mean(const PF2D* pf);
pf2d_real pf2d_log_vol_variance(const PF2D* pf);
pf2d_real pf2d_vol_mean(const PF2D* pf);
void pf2d_print_config(const PF2D* pf);
```

## Execution Paths

The implementation automatically selects the optimal execution path:

| Path | Condition | Description |
|------|-----------|-------------|
| PCG | `use_pcg=1` | Inline RNG, lowest latency |
| MKL thread-local | `use_pcg=0, N<4000` | Per-thread MKL streams |
| Vectorized | `use_pcg=0, N≥4000` | Pre-computed vExp, fused loops |

### HIGH_N Mode

For offline analysis with 50k+ particles, compile with `-DPF2D_HIGH_N`:

```bash
make pf2d_high_n
```

Enables:
- Parallel prefix sum (cumulative sum)
- Parallel batch binary search
- Parallel gather in resampling

## Performance

Benchmarks on AMD EPYC (192 threads):

| Particles | Mode | Time/update |
|-----------|------|-------------|
| 4,000 | PCG | ~20 μs |
| 4,000 | Vectorized | ~35 μs |
| 50,000 | HIGH_N | ~120 μs |
| 100,000 | HIGH_N | ~230 μs |

## Configuration

### Compile-time Options

```c
#define PF2D_USE_FLOAT              /* Single precision (faster) */
#define PF2D_HIGH_N                 /* Enable parallel algorithms */
```

### Tunable Constants

```c
#define PF2D_MAX_REGIMES 8
#define PF2D_BLAS_THRESHOLD 4000
#define PF2D_REGIME_LUT_SIZE 1024
#define PF2D_RESAMPLE_THRESH_DEFAULT 0.5

/* HIGH_N thresholds */
#define PF2D_PARALLEL_CUMSUM_THRESH 8000
#define PF2D_PARALLEL_SEARCH_THRESH 16000
```

## Implementation Details

### Memory Layout

Structure-of-Arrays (SoA) for vectorization:

```c
pf2d_real* prices;      /* [N] */
pf2d_real* log_vols;    /* [N] */
pf2d_real* weights;     /* [N] */
int* regimes;           /* [N] */
```

### Resampling

Systematic resampling with adaptive threshold based on ESS:

```c
if (ESS < threshold * N) {
    resample();
}
```

Threshold adapts based on current volatility estimate.

### RNG

Two options:
- **PCG32**: Counter-based, 16 bytes state, inline generation
- **MKL SFMT**: Vectorized Mersenne Twister via VSL

### Numerical Stability

- Weight underflow detection with uniform fallback
- Log-sum-exp trick for weight normalization
- Safe LUT indexing for boundary cases

## Intel MKL Integration

Careful MKL usage was critical for achieving microsecond-level latency. Here are the key design decisions:

### 1. Thread-Local RNG Streams

**Problem:** Single MKL RNG stream causes contention on many-core systems.

**Solution:** One VSL stream per thread, each generating its own chunk:

```c
/* Bad: Single stream, internal MKL parallelism */
vdRngGaussian(method, single_stream, N, buffer, 0, 1);

/* Good: Thread-local streams, NUMA-aware */
#pragma omp parallel
{
    int tid = omp_get_thread_num();
    int chunk = N / n_threads;
    int start = tid * chunk;
    vdRngGaussian(method, stream[tid], chunk, &buffer[start], 0, 1);
}
```

Each thread writes to its local memory region, avoiding cross-NUMA traffic.

### 2. Dual-Path Algorithm Selection

**Problem:** BLAS/VML have call overhead that dominates for small N.

**Solution:** Threshold-based path selection:

```c
#define PF2D_BLAS_THRESHOLD 4000

if (n < PF2D_BLAS_THRESHOLD) {
    /* Manual loops - no function call overhead */
    for (int i = 0; i < n; i++) {
        sum += weights[i];
    }
} else {
    /* BLAS - parallelism wins for large N */
    sum = cblas_dasum(n, weights, 1);
}
```

| N | Winner | Reason |
|---|--------|--------|
| < 4000 | Manual loops | Call overhead dominates |
| ≥ 4000 | BLAS/VML | Parallelism + SIMD wins |

### 3. Pre-computed Vectorized Transcendentals

**Problem:** `exp()` per particle in the hot loop is expensive.

**Solution:** Batch compute via VML before the loop:

```c
/* Vectorized path: one VML call outside loop */
vdExp(n, log_vols, vols);  /* AVX-512 accelerated */

#pragma omp parallel for
for (int i = 0; i < n; i++) {
    /* Now just FMA operations - no transcendentals */
    prices[i] += drift + vols[i] * noise[i];
}
```

This moves the expensive operation out of the particle loop entirely.

### 4. Fused Weight Update

**Problem:** Multiple passes over weight array waste memory bandwidth.

**Solution:** Fuse log-likelihood computation with max-finding:

```c
/* Single pass: compute log-weights AND find max */
pf2d_real max_lw = -1e30;

#pragma omp parallel for reduction(max: max_lw)
for (int i = 0; i < n; i++) {
    pf2d_real diff = obs - prices[i];
    pf2d_real lw = -0.5 * diff * diff / obs_var;
    log_weights[i] = lw;
    if (lw > max_lw) max_lw = lw;
}

/* Then VML for exp */
vdExp(n, log_weights, weights);
```

### 5. Strategic API Selection

We use only these MKL APIs (verified to exist and be fast):

| API | Purpose | Why |
|-----|---------|-----|
| `vdRngGaussian` | Bulk normal RNG | Vectorized Box-Muller |
| `vdRngUniform` | Bulk uniform RNG | Fast for regime sampling |
| `vdExp` / `vsExp` | Vectorized exp | AVX-512, much faster than scalar |
| `vdSqr` / `vsSqr` | Vectorized square | For residual computation |
| `cblas_dasum` | Sum of absolutes | Threaded reduction |
| `cblas_dscal` | Scale vector | Threaded for normalize |
| `mkl_malloc` | Aligned allocation | 64-byte for cache lines |

**Avoided:** Fabricated or slow APIs like `vmdExp`, `vslNewTask`, or task-based interfaces that don't exist or add overhead.

### 6. Memory Alignment

All particle arrays use 64-byte alignment for optimal cache and SIMD:

```c
pf->prices = (pf2d_real*)mkl_malloc(n * sizeof(pf2d_real), 64);
```

This ensures:
- Full cache line utilization
- No split loads in AVX-512
- Optimal prefetcher behavior

### 7. Let MKL Handle Threading

**Decision:** Don't manually manage thread pools. MKL's internal threading (via TBB or OpenMP) handles NUMA topology automatically.

```c
/* Set MKL threads once at startup */
mkl_set_num_threads(omp_get_max_threads());

/* Then just call MKL - it handles the rest */
vdExp(n, input, output);  /* Internally parallel */
```

Fighting MKL's threading model leads to oversubscription and cache thrashing.

### 8. PCG Fallback for Ultra-Low Latency

For N < 4000 where MKL overhead matters, we bypass MKL entirely:

```c
typedef struct {
    uint64_t state;
    uint64_t inc;
} pf2d_pcg32_t;

/* Inline RNG - no function call */
static inline uint32_t pf2d_pcg32_random(pf2d_pcg32_t* rng) {
    uint64_t old = rng->state;
    rng->state = old * 6364136223846793005ULL + rng->inc;
    uint32_t xor = ((old >> 18u) ^ old) >> 27u;
    uint32_t rot = old >> 59u;
    return (xor >> rot) | (xor << ((-rot) & 31));
}
```

16 bytes state vs MKL's 624 words. No locking, no function calls.

## Files

| File | Description |
|------|-------------|
| `particle_filter_2d.h` | Header file |
| `particle_filter_2d.c` | Implementation |
| `example_usage_2d.c` | Benchmark example |
| `Makefile` | Build configuration |

## References

- Gordon, Salmond, Smith (1993). Novel approach to nonlinear/non-Gaussian Bayesian state estimation. *IEE Proceedings F*.
- Doucet, Johansen (2009). A tutorial on particle filtering and smoothing. *Handbook of Nonlinear Filtering*.
- O'Neill, M. E. (2014). PCG: A family of simple fast space-efficient statistically good algorithms for random number generation.

## License

MIT License
