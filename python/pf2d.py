"""
Python bindings for the 2D Particle Filter (pf2d)

High-performance stochastic volatility particle filter with regime switching.
Optimized C implementation with MKL acceleration.

Usage:
    from pf2d import ParticleFilter2D
    
    pf = ParticleFilter2D(n_particles=4000, n_regimes=4)
    pf.set_regime_params(0, drift=0.001, theta=0.02, mu_vol=-4.6, sigma_vol=0.05)
    pf.initialize(price0=100.0, log_vol0=-4.6)
    
    for price in price_stream:
        result = pf.update(price)
        print(f"Est: {result.price_mean:.4f}, Vol: {result.vol_mean:.4f}")
"""

import ctypes
import numpy as np
import os
import sys
from ctypes import (
    c_int, c_double, c_void_p, c_uint8,
    POINTER, Structure, byref
)
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import platform


# =============================================================================
# Windows MKL DLL paths (must be set BEFORE loading the library)
# =============================================================================

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.0\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\2025.3\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            try:
                os.add_dll_directory(p)
            except (OSError, AttributeError):
                pass  # add_dll_directory not available on older Python


# =============================================================================
# CONSTANTS (must match particle_filter_2d.h)
# =============================================================================

PF2D_MAX_REGIMES = 8
PF2D_REAL_SIZE = 8  # sizeof(double)


# =============================================================================
# C STRUCTURES
# =============================================================================

class PF2DRegimeProbs(Structure):
    """Regime transition probabilities"""
    _fields_ = [("probs", c_double * PF2D_MAX_REGIMES)]


class PF2DOutput(Structure):
    """Output from pf2d_update()"""
    _fields_ = [
        ("price_mean", c_double),
        ("price_variance", c_double),
        ("log_vol_mean", c_double),
        ("log_vol_variance", c_double),
        ("vol_mean", c_double),
        ("ess", c_double),
        ("regime_probs", c_double * PF2D_MAX_REGIMES),
        ("dominant_regime", c_int),
        ("resampled", c_int),
    ]


# =============================================================================
# PYTHON RESULT DATACLASS
# =============================================================================

@dataclass
class PF2DResult:
    """Python-friendly result from particle filter update"""
    price_mean: float
    price_variance: float
    price_std: float
    log_vol_mean: float
    log_vol_variance: float
    log_vol_std: float
    vol_mean: float
    ess: float
    ess_ratio: float
    regime_probs: np.ndarray
    dominant_regime: int
    resampled: bool
    
    def __repr__(self):
        return (
            f"PF2DResult(price={self.price_mean:.4f}±{self.price_std:.4f}, "
            f"vol={self.vol_mean:.6f}, ESS={self.ess:.0f} ({self.ess_ratio:.1%}), "
            f"regime={self.dominant_regime})"
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

c_double_p = POINTER(c_double)

def _to_c_array(arr: np.ndarray, dtype=np.float64) -> c_double_p:
    """Convert numpy array to ctypes pointer."""
    arr = np.ascontiguousarray(arr, dtype=dtype)
    return arr.ctypes.data_as(c_double_p)


def _from_c_array(ptr: c_double_p, shape: tuple) -> np.ndarray:
    """Create numpy array from ctypes pointer (copy)."""
    size = int(np.prod(shape))
    arr = np.ctypeslib.as_array(ptr, shape=(size,)).copy()
    return arr.reshape(shape) if len(shape) > 1 else arr


# =============================================================================
# LIBRARY LOADER
# =============================================================================

def _load_library(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Load the pf2d shared library"""
    
    if lib_path:
        return ctypes.CDLL(lib_path)
    
    # Auto-detect library location
    system = platform.system()
    
    # Get directory of this file
    this_dir = Path(__file__).parent
    
    search_paths = [
        # Same directory as this module
        this_dir,
        # Build directories (relative to module)
        this_dir / "build" / "Release",
        this_dir / "build" / "Debug",
        this_dir / "build",
        # Current working directory
        Path("."),
        Path("build/Release"),
        Path("build/Debug"),
        Path("build"),
        # Installed locations
        Path("/usr/local/lib"),
        Path("/usr/lib"),
    ]
    
    if system == "Windows":
        lib_names = ["particle_filter_2d.dll", "pf2d.dll", "libpf2d.dll"]
    elif system == "Darwin":
        lib_names = ["particle_filter_2d.so", "libparticle_filter_2d.dylib", "libpf2d.dylib"]
    else:
        lib_names = ["particle_filter_2d.so", "libparticle_filter_2d.so", "libpf2d.so"]
    
    for path in search_paths:
        for name in lib_names:
            full_path = path / name
            if full_path.exists():
                try:
                    return ctypes.CDLL(str(full_path))
                except OSError:
                    continue
    
    # Try system path
    for name in lib_names:
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    
    raise RuntimeError(
        f"Could not find pf2d library. Searched: {lib_names}\n"
        f"Build with: cmake --build build --target particle_filter_2d_shared --config Release\n"
        f"Then copy DLL to this directory or specify path: ParticleFilter2D(lib_path='path/to/lib')"
    )


def _setup_signatures(lib: ctypes.CDLL):
    """Set up C function signatures for type safety."""
    
    # pf2d_create
    lib.pf2d_create.argtypes = [c_int, c_int]
    lib.pf2d_create.restype = c_void_p
    
    # pf2d_destroy
    lib.pf2d_destroy.argtypes = [c_void_p]
    lib.pf2d_destroy.restype = None
    
    # pf2d_set_observation_variance
    lib.pf2d_set_observation_variance.argtypes = [c_void_p, c_double]
    lib.pf2d_set_observation_variance.restype = None
    
    # pf2d_set_regime_params
    lib.pf2d_set_regime_params.argtypes = [
        c_void_p, c_int, c_double, c_double, c_double, c_double, c_double
    ]
    lib.pf2d_set_regime_params.restype = None
    
    # pf2d_set_regime_probs - operates on PF2DRegimeProbs struct
    lib.pf2d_set_regime_probs.argtypes = [POINTER(PF2DRegimeProbs), POINTER(c_double), c_int]
    lib.pf2d_set_regime_probs.restype = None
    
    # pf2d_build_regime_lut - builds LUT from probs
    lib.pf2d_build_regime_lut.argtypes = [c_void_p, POINTER(PF2DRegimeProbs)]
    lib.pf2d_build_regime_lut.restype = None
    
    # pf2d_initialize
    lib.pf2d_initialize.argtypes = [
        c_void_p, c_double, c_double, c_double, c_double
    ]
    lib.pf2d_initialize.restype = None
    
    # pf2d_update
    lib.pf2d_update.argtypes = [c_void_p, c_double, POINTER(PF2DRegimeProbs)]
    lib.pf2d_update.restype = PF2DOutput
    
    # pf2d_warmup
    lib.pf2d_warmup.argtypes = [c_void_p]
    lib.pf2d_warmup.restype = None
    
    # pf2d_enable_pcg
    lib.pf2d_enable_pcg.argtypes = [c_void_p, c_int]
    lib.pf2d_enable_pcg.restype = None
    
    # pf2d_print_config
    lib.pf2d_print_config.argtypes = [c_void_p]
    lib.pf2d_print_config.restype = None
    
    # pf2d_effective_sample_size
    lib.pf2d_effective_sample_size.argtypes = [c_void_p]
    lib.pf2d_effective_sample_size.restype = c_double
    
    lib.pf2d_mkl_config_14900kf.argtypes = [c_int]
    lib.pf2d_mkl_config_14900kf.restype = None


# =============================================================================
# MODULE-LEVEL LIBRARY (loaded at import)
# =============================================================================

_lib: Optional[ctypes.CDLL] = None


def _get_lib(lib_path: Optional[str] = None) -> ctypes.CDLL:
    """Get library, loading if necessary."""
    global _lib
    if _lib is None or lib_path is not None:
        _lib = _load_library(lib_path)
        _setup_signatures(_lib)
        _lib.pf2d_mkl_config_14900kf(0)  # Configure 16 P-core threads
    return _lib


# =============================================================================
# PARTICLE FILTER CLASS
# =============================================================================

class ParticleFilter2D:
    """
    2D Particle Filter for stochastic volatility with regime switching.
    
    State: [price, log_volatility]
    Dynamics: Per-regime OU process for log-volatility
    
    Parameters
    ----------
    n_particles : int
        Number of particles (default 4000)
    n_regimes : int
        Number of market regimes (default 4)
    lib_path : str, optional
        Path to the shared library
    
    Example
    -------
    >>> pf = ParticleFilter2D(n_particles=4000)
    >>> pf.set_regime_params(0, drift=0.001, theta=0.02, mu_vol=-4.6, sigma_vol=0.05)
    >>> pf.initialize(price0=100.0)
    >>> 
    >>> for price in prices:
    ...     result = pf.update(price)
    ...     print(f"Price: {result.price_mean:.2f}, Vol: {result.vol_mean:.4f}")
    """
    
    def __init__(
        self,
        n_particles: int = 4000,
        n_regimes: int = 4,
        lib_path: Optional[str] = None
    ):
        self.n_particles = n_particles
        self.n_regimes = n_regimes
        
        # Load library
        self._lib = _get_lib(lib_path)
        
        # Create filter
        self._pf = self._lib.pf2d_create(n_particles, n_regimes)
        if not self._pf:
            raise RuntimeError("Failed to create particle filter")
        
        self._owns_ptr = True
        
        # Default regime probs (uniform)
        self._regime_probs = PF2DRegimeProbs()
        for i in range(n_regimes):
            self._regime_probs.probs[i] = 1.0 / n_regimes
        
        self._initialized = False
    
    def __del__(self):
        """Clean up C resources"""
        if hasattr(self, '_owns_ptr') and self._owns_ptr and hasattr(self, '_pf') and self._pf:
            self._lib.pf2d_destroy(self._pf)
            self._pf = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        if self._owns_ptr and self._pf:
            self._lib.pf2d_destroy(self._pf)
            self._pf = None
            self._owns_ptr = False
    
    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------
    
    def set_observation_variance(self, var: float):
        """Set observation noise variance"""
        self._lib.pf2d_set_observation_variance(self._pf, var)
    
    def set_regime_params(
        self,
        regime: int,
        drift: float = 0.0,
        theta: float = 0.05,
        mu_vol: float = -4.6,  # log(0.01)
        sigma_vol: float = 0.05,
        rho: float = 0.0
    ):
        """
        Set parameters for a regime.
        
        Parameters
        ----------
        regime : int
            Regime index (0 to n_regimes-1)
        drift : float
            Price drift per tick
        theta : float
            Mean reversion speed for log-volatility
        mu_vol : float
            Long-run mean of log-volatility (e.g., log(0.01) = -4.6)
        sigma_vol : float
            Volatility of volatility
        rho : float
            Correlation between price and volatility shocks
        """
        if regime < 0 or regime >= self.n_regimes:
            raise ValueError(f"Regime must be 0-{self.n_regimes-1}")
        
        self._lib.pf2d_set_regime_params(
            self._pf, regime, drift, theta, mu_vol, sigma_vol, rho
        )
    
    def set_regime_probs(self, probs: List[float]):
        """Set regime transition probabilities"""
        if len(probs) != self.n_regimes:
            raise ValueError(f"Expected {self.n_regimes} probabilities")
        
        probs_arr = (c_double * self.n_regimes)(*probs)
        self._lib.pf2d_set_regime_probs(byref(self._regime_probs), probs_arr, self.n_regimes)
        self._lib.pf2d_build_regime_lut(self._pf, byref(self._regime_probs))
    
    def initialize(
        self,
        price0: float,
        log_vol0: float = -4.6,
        price_spread: float = 0.01,
        log_vol_spread: float = 0.5
    ):
        """
        Initialize particle cloud around initial state.
        
        Parameters
        ----------
        price0 : float
            Initial price
        log_vol0 : float
            Initial log-volatility (default: log(0.01))
        price_spread : float
            Spread for initial price distribution
        log_vol_spread : float
            Spread for initial log-vol distribution
        """
        self._lib.pf2d_initialize(
            self._pf, price0, price_spread, log_vol0, log_vol_spread
        )
        self._initialized = True
    
    def warmup(self):
        """Warmup MKL kernels to eliminate first-call latency"""
        self._lib.pf2d_warmup(self._pf)
    
    def enable_pcg(self, enable: bool = True):
        """Enable/disable PCG RNG (default: auto-select based on N)"""
        self._lib.pf2d_enable_pcg(self._pf, 1 if enable else 0)
    
    def print_config(self):
        """Print filter configuration to stdout"""
        self._lib.pf2d_print_config(self._pf)
    
    # -------------------------------------------------------------------------
    # Filtering
    # -------------------------------------------------------------------------
    
    def update(self, observation: float) -> PF2DResult:
        """
        Run one filter update step.
        
        Parameters
        ----------
        observation : float
            Observed price
        
        Returns
        -------
        PF2DResult
            Filter estimates and diagnostics
        """
        if not self._initialized:
            raise RuntimeError("Call initialize() before update()")
        
        out = self._lib.pf2d_update(
            self._pf, observation, byref(self._regime_probs)
        )
        
        return PF2DResult(
            price_mean=out.price_mean,
            price_variance=out.price_variance,
            price_std=np.sqrt(out.price_variance),
            log_vol_mean=out.log_vol_mean,
            log_vol_variance=out.log_vol_variance,
            log_vol_std=np.sqrt(out.log_vol_variance),
            vol_mean=out.vol_mean,
            ess=out.ess,
            ess_ratio=out.ess / self.n_particles,
            regime_probs=np.array([out.regime_probs[i] for i in range(self.n_regimes)]),
            dominant_regime=out.dominant_regime,
            resampled=bool(out.resampled),
        )
    
    # -------------------------------------------------------------------------
    # Batch Processing
    # -------------------------------------------------------------------------
    
    def update_batch(self, observations: np.ndarray) -> List[PF2DResult]:
        """
        Run filter on a batch of observations.
        
        Parameters
        ----------
        observations : array-like
            Array of observed prices
        
        Returns
        -------
        list of PF2DResult
            Results for each observation
        """
        return [self.update(float(obs)) for obs in observations]
    
    def run(
        self,
        observations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run filter on observations and return arrays.
        
        Parameters
        ----------
        observations : array-like
            Array of observed prices
        
        Returns
        -------
        price_est, vol_est, ess : np.ndarray
            Arrays of estimates
        """
        n = len(observations)
        price_est = np.zeros(n)
        vol_est = np.zeros(n)
        ess = np.zeros(n)
        
        for i, obs in enumerate(observations):
            result = self.update(float(obs))
            price_est[i] = result.price_mean
            vol_est[i] = result.vol_mean
            ess[i] = result.ess
        
        return price_est, vol_est, ess
    
    def filter(
        self,
        observations: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Run filter on observations with full output.
        
        Parameters
        ----------
        observations : array-like
            Array of observed prices
        
        Returns
        -------
        price_est, vol_est, ess, regimes : np.ndarray
            Arrays of estimates and dominant regimes
        """
        n = len(observations)
        price_est = np.zeros(n)
        vol_est = np.zeros(n)
        ess = np.zeros(n)
        regimes = np.zeros(n, dtype=np.int32)
        
        for i, obs in enumerate(observations):
            result = self.update(float(obs))
            price_est[i] = result.price_mean
            vol_est[i] = result.vol_mean
            ess[i] = result.ess
            regimes[i] = result.dominant_regime
        
        return price_est, vol_est, ess, regimes
    
    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------
    
    @property
    def ess(self) -> float:
        """Current effective sample size"""
        return self._lib.pf2d_effective_sample_size(self._pf)
    
    @property
    def ess_ratio(self) -> float:
        """ESS as fraction of N"""
        return self.ess / self.n_particles


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_default_filter(n_particles: int = 4000, lib_path: Optional[str] = None) -> ParticleFilter2D:
    """
    Create a particle filter with default regime configuration.
    
    Regimes:
        0: Trend (drift=0.001, low vol, slow MR)
        1: Mean-revert (no drift, stable vol)
        2: High-vol (no drift, elevated vol)
        3: Jump (no drift, high vol-of-vol)
    """
    pf = ParticleFilter2D(n_particles=n_particles, n_regimes=4, lib_path=lib_path)
    
    # Configure regimes
    pf.set_regime_params(0, drift=0.001, theta=0.02, mu_vol=np.log(0.01), sigma_vol=0.05)
    pf.set_regime_params(1, drift=0.0, theta=0.05, mu_vol=np.log(0.008), sigma_vol=0.03)
    pf.set_regime_params(2, drift=0.0, theta=0.10, mu_vol=np.log(0.03), sigma_vol=0.10)
    pf.set_regime_params(3, drift=0.0, theta=0.20, mu_vol=np.log(0.05), sigma_vol=0.20)
    
    # Set regime probabilities (mostly stable)
    pf.set_regime_probs([0.4, 0.3, 0.2, 0.1])
    
    # Observation variance
    pf.set_observation_variance(0.0001)
    
    return pf


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("PF2D Python Bindings Test")
    print("=" * 60)
    
    # Create filter
    try:
        pf = create_default_filter(n_particles=4000)
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    pf.initialize(price0=100.0, log_vol0=np.log(0.01))
    pf.warmup()
    
    print(f"\nFilter: {pf.n_particles} particles, {pf.n_regimes} regimes")
    pf.print_config()
    
    # Simulate prices
    np.random.seed(42)
    n_ticks = 1000
    true_vol = 0.01
    price = 100.0
    prices = [price]
    
    for _ in range(n_ticks - 1):
        true_vol = np.exp(0.95 * np.log(true_vol) + 0.05 * np.log(0.01) + 0.05 * np.random.randn())
        price += true_vol * np.random.randn()
        prices.append(price)
    
    prices = np.array(prices)
    
    # Run filter
    print(f"\nRunning {n_ticks} updates...")
    
    start = time.perf_counter()
    price_est, vol_est, ess = pf.run(prices)
    elapsed = time.perf_counter() - start
    
    print(f"\nPerformance:")
    print(f"  Total time:  {elapsed*1000:.1f} ms")
    print(f"  Per tick:    {elapsed/n_ticks*1e6:.1f} μs")
    print(f"  Throughput:  {n_ticks/elapsed:.0f} ticks/sec")
    
    print(f"\nAccuracy:")
    rmse = np.sqrt(np.mean((prices - price_est)**2))
    print(f"  Price RMSE:  {rmse:.6f}")
    print(f"  Mean ESS:    {np.mean(ess):.0f} ({np.mean(ess)/pf.n_particles:.1%})")
    
    print(f"\nFinal state:")
    print(f"  True price:  {prices[-1]:.4f}")
    print(f"  Est price:   {price_est[-1]:.4f}")
    print(f"  Est vol:     {vol_est[-1]:.6f}")
    
    print("\n" + "=" * 60)
    print("Test complete!")

