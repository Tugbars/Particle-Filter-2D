"""
PF2D Python Bindings
====================

Ctypes wrapper for particle_filter_2d.dll MKL-optimized particle filter.

CRITICAL: Struct definitions MUST match C header exactly or crashes occur.

Usage:
    from pf2d import create_default_filter
    
    with create_default_filter(n_particles=4000) as pf:
        pf.initialize(price0=100.0, log_vol0=np.log(0.01))
        pf.warmup()
        
        for price in prices:
            result = pf.update(price)
            print(f"Vol: {result.vol_mean:.4f}, ESS: {result.ess:.0f}")
"""

import ctypes
from ctypes import (
    Structure, POINTER, CDLL,
    c_int, c_float, c_double, c_uint8, c_void_p
)
import numpy as np
import os
import sys
import atexit
import weakref
from dataclasses import dataclass
from typing import Optional, List, Tuple
from pathlib import Path

# ============================================================================
# PRECISION CONFIGURATION - MUST MATCH COMPILED DLL
# ============================================================================

# Set this to match your DLL compilation:
#   True  = DLL compiled with PF2D_USE_FLOAT (default)
#   False = DLL compiled with PF2D_USE_DOUBLE
PF2D_USE_FLOAT = True

if PF2D_USE_FLOAT:
    pf2d_real = c_float
    pf2d_real_np = np.float32
else:
    pf2d_real = c_double
    pf2d_real_np = np.float64

# Constants from header
PF2D_MAX_REGIMES = 8
PF2D_REGIME_LUT_SIZE = 1024

# ============================================================================
# Windows MKL DLL paths
# ============================================================================

if sys.platform == "win32":
    mkl_paths = [
        r"C:\Program Files (x86)\Intel\oneAPI\mkl\latest\bin",
        r"C:\Program Files (x86)\Intel\oneAPI\compiler\latest\bin",
    ]
    for p in mkl_paths:
        if os.path.exists(p):
            os.add_dll_directory(p)

# ============================================================================
# Load DLL
# ============================================================================

def _load_library():
    """Find and load the PF2D shared library."""
    search_paths = [
        Path(__file__).parent / "particle_filter_2d.dll",
        Path(__file__).parent / "libpf2d.so",
        Path(".") / "particle_filter_2d.dll",
        Path(".") / "libpf2d.so",
        "particle_filter_2d.dll",
        "libpf2d.so",
    ]
    
    for path in search_paths:
        try:
            lib = CDLL(str(path))
            return lib
        except OSError:
            continue
    
    raise RuntimeError(
        "Could not load particle_filter_2d.dll/.so\n"
        "Make sure the DLL is in the current directory or same folder as pf2d.py"
    )

_lib = _load_library()

# ============================================================================
# Configure MKL from C side (most reliable)
# ============================================================================

try:
    _lib.pf2d_mkl_config_14900kf.argtypes = [c_int]
    _lib.pf2d_mkl_config_14900kf.restype = None
    _lib.pf2d_mkl_config_14900kf(0)  # 0 = silent
    _HAS_MKL_CONFIG = True
except AttributeError:
    _HAS_MKL_CONFIG = False

# ============================================================================
# STRUCTURE DEFINITIONS - MUST MATCH C HEADER EXACTLY
# ============================================================================

class PF2DRegimeProbs(Structure):
    """
    Matches C struct PF2DRegimeProbs exactly.
    
    typedef struct {
        pf2d_real probs[PF2D_MAX_REGIMES];
        pf2d_real cumprobs[PF2D_MAX_REGIMES];
        int n_regimes;
    } PF2DRegimeProbs;
    """
    _fields_ = [
        ("probs", pf2d_real * PF2D_MAX_REGIMES),
        ("cumprobs", pf2d_real * PF2D_MAX_REGIMES),
        ("n_regimes", c_int),
    ]


class PF2DOutput(Structure):
    """
    Matches C struct PF2DOutput exactly.
    
    typedef struct {
        pf2d_real price_mean;
        pf2d_real price_variance;
        pf2d_real log_vol_mean;
        pf2d_real log_vol_variance;
        pf2d_real vol_mean;
        pf2d_real ess;
        pf2d_real regime_probs[PF2D_MAX_REGIMES];
        int dominant_regime;
        int resampled;
        pf2d_real sigma_vol_scale;
        pf2d_real ess_ema;
        int high_vol_mode;
        int regime_feedback_active;
        int bocpd_cooldown_remaining;
    } PF2DOutput;
    """
    _fields_ = [
        ("price_mean", pf2d_real),
        ("price_variance", pf2d_real),
        ("log_vol_mean", pf2d_real),
        ("log_vol_variance", pf2d_real),
        ("vol_mean", pf2d_real),
        ("ess", pf2d_real),
        ("regime_probs", pf2d_real * PF2D_MAX_REGIMES),
        ("dominant_regime", c_int),
        ("resampled", c_int),
        # Adaptive diagnostics - CRITICAL: these were missing before!
        ("sigma_vol_scale", pf2d_real),
        ("ess_ema", pf2d_real),
        ("high_vol_mode", c_int),
        ("regime_feedback_active", c_int),
        ("bocpd_cooldown_remaining", c_int),
    ]


# Verify struct sizes for debugging
_EXPECTED_OUTPUT_SIZE = (
    6 * ctypes.sizeof(pf2d_real) +           # 6 scalars
    PF2D_MAX_REGIMES * ctypes.sizeof(pf2d_real) +  # regime_probs[8]
    2 * ctypes.sizeof(c_int) +                # dominant_regime, resampled
    2 * ctypes.sizeof(pf2d_real) +           # sigma_vol_scale, ess_ema
    3 * ctypes.sizeof(c_int)                  # high_vol_mode, regime_feedback_active, bocpd_cooldown_remaining
)

_EXPECTED_REGIME_PROBS_SIZE = (
    2 * PF2D_MAX_REGIMES * ctypes.sizeof(pf2d_real) +  # probs + cumprobs
    ctypes.sizeof(c_int)                               # n_regimes
)

# ============================================================================
# FUNCTION SIGNATURES
# ============================================================================

# Opaque pointer to PF2D struct
PF2D_PTR = c_void_p

# Create/Destroy
_lib.pf2d_create.argtypes = [c_int, c_int]
_lib.pf2d_create.restype = PF2D_PTR

_lib.pf2d_destroy.argtypes = [PF2D_PTR]
_lib.pf2d_destroy.restype = None

# Configuration
_lib.pf2d_set_observation_variance.argtypes = [PF2D_PTR, pf2d_real]
_lib.pf2d_set_observation_variance.restype = None

_lib.pf2d_set_regime_params.argtypes = [
    PF2D_PTR, c_int,
    pf2d_real, pf2d_real, pf2d_real, pf2d_real, pf2d_real
]
_lib.pf2d_set_regime_params.restype = None

_lib.pf2d_set_regime_probs.argtypes = [
    POINTER(PF2DRegimeProbs),
    POINTER(pf2d_real),
    c_int
]
_lib.pf2d_set_regime_probs.restype = None

_lib.pf2d_build_regime_lut.argtypes = [PF2D_PTR, POINTER(PF2DRegimeProbs)]
_lib.pf2d_build_regime_lut.restype = None

_lib.pf2d_enable_pcg.argtypes = [PF2D_PTR, c_int]
_lib.pf2d_enable_pcg.restype = None

# Initialization
_lib.pf2d_initialize.argtypes = [
    PF2D_PTR, pf2d_real, pf2d_real, pf2d_real, pf2d_real
]
_lib.pf2d_initialize.restype = None

_lib.pf2d_warmup.argtypes = [PF2D_PTR]
_lib.pf2d_warmup.restype = None

# Core update - RETURNS STRUCT BY VALUE
_lib.pf2d_update.argtypes = [PF2D_PTR, pf2d_real, POINTER(PF2DRegimeProbs)]
_lib.pf2d_update.restype = PF2DOutput

# Estimates
_lib.pf2d_price_mean.argtypes = [PF2D_PTR]
_lib.pf2d_price_mean.restype = pf2d_real

_lib.pf2d_price_variance.argtypes = [PF2D_PTR]
_lib.pf2d_price_variance.restype = pf2d_real

_lib.pf2d_log_vol_mean.argtypes = [PF2D_PTR]
_lib.pf2d_log_vol_mean.restype = pf2d_real

_lib.pf2d_log_vol_variance.argtypes = [PF2D_PTR]
_lib.pf2d_log_vol_variance.restype = pf2d_real

_lib.pf2d_vol_mean.argtypes = [PF2D_PTR]
_lib.pf2d_vol_mean.restype = pf2d_real

_lib.pf2d_effective_sample_size.argtypes = [PF2D_PTR]
_lib.pf2d_effective_sample_size.restype = pf2d_real

# Debug
_lib.pf2d_print_config.argtypes = [PF2D_PTR]
_lib.pf2d_print_config.restype = None

# ============================================================================
# PYTHON RESULT DATACLASS
# ============================================================================

@dataclass
class PF2DResult:
    """Python-friendly result from pf2d_update()."""
    price_mean: float
    price_std: float
    log_vol_mean: float
    log_vol_std: float
    vol_mean: float
    ess: float
    regime_probs: np.ndarray
    dominant_regime: int
    resampled: bool
    # Adaptive diagnostics
    sigma_vol_scale: float
    ess_ema: float
    high_vol_mode: bool
    regime_feedback_active: bool
    bocpd_cooldown_remaining: int

# ============================================================================
# CLEANUP TRACKING
# ============================================================================

_active_filters: weakref.WeakSet = weakref.WeakSet()

def _cleanup_all_filters():
    """Called at interpreter exit."""
    for pf in list(_active_filters):
        try:
            pf.close()
        except Exception:
            pass

atexit.register(_cleanup_all_filters)

# ============================================================================
# MAIN WRAPPER CLASS
# ============================================================================

class ParticleFilter2D:
    """
    Python wrapper for PF2D particle filter.
    
    Parameters
    ----------
    n_particles : int
        Number of particles (default 4000)
    n_regimes : int
        Number of regimes (default 4, max 8)
    
    Example
    -------
    >>> with ParticleFilter2D(n_particles=4000) as pf:
    ...     pf.initialize(price0=100.0, log_vol0=np.log(0.01))
    ...     pf.warmup()
    ...     result = pf.update(100.5)
    ...     print(f"Vol: {result.vol_mean:.4f}")
    """
    
    def __init__(self, n_particles: int = 4000, n_regimes: int = 4):
        if n_regimes > PF2D_MAX_REGIMES:
            raise ValueError(f"n_regimes must be <= {PF2D_MAX_REGIMES}")
        
        self.n_particles = n_particles
        self.n_regimes = n_regimes
        self._closed = False
        
        # Create filter
        self._pf = _lib.pf2d_create(n_particles, n_regimes)
        if not self._pf:
            raise RuntimeError("pf2d_create failed")
        
        # Create regime probs structure
        self._regime_probs = PF2DRegimeProbs()
        self._regime_probs.n_regimes = n_regimes
        
        # Default uniform probs
        uniform = 1.0 / n_regimes
        for i in range(n_regimes):
            self._regime_probs.probs[i] = pf2d_real(uniform)
        self._update_cumprobs()
        
        # Build LUT
        _lib.pf2d_build_regime_lut(self._pf, ctypes.byref(self._regime_probs))
        
        # Track for cleanup
        _active_filters.add(self)
    
    def _update_cumprobs(self):
        """Recompute cumulative probabilities."""
        cumsum = 0.0
        for i in range(self.n_regimes):
            cumsum += float(self._regime_probs.probs[i])
            self._regime_probs.cumprobs[i] = pf2d_real(cumsum)
    
    def _check_open(self):
        """Raise if filter has been closed."""
        if self._closed:
            raise RuntimeError("ParticleFilter2D has been closed")
    
    def close(self):
        """Explicitly release resources. Safe to call multiple times."""
        if not self._closed and self._pf:
            _lib.pf2d_destroy(self._pf)
            self._pf = None
            self._closed = True
    
    def __del__(self):
        self.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    # ========================================================================
    # Configuration
    # ========================================================================
    
    def set_observation_variance(self, var: float):
        """Set observation noise variance."""
        self._check_open()
        _lib.pf2d_set_observation_variance(self._pf, pf2d_real(var))
    
    def set_regime_params(self, regime: int, drift: float, theta: float,
                          mu_vol: float, sigma_vol: float, rho: float = 0.0):
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
            Long-term mean of log-volatility
        sigma_vol : float
            Volatility of volatility
        rho : float
            Price-vol correlation (leverage effect), -1 to 1
        """
        self._check_open()
        if regime < 0 or regime >= self.n_regimes:
            raise ValueError(f"regime must be 0 to {self.n_regimes-1}")
        
        _lib.pf2d_set_regime_params(
            self._pf, regime,
            pf2d_real(drift),
            pf2d_real(theta),
            pf2d_real(mu_vol),
            pf2d_real(sigma_vol),
            pf2d_real(rho)
        )
    
    def set_regime_probs(self, probs: List[float]):
        """Set regime prior probabilities."""
        self._check_open()
        if len(probs) != self.n_regimes:
            raise ValueError(f"probs must have {self.n_regimes} elements")
        
        # Normalize
        total = sum(probs)
        for i, p in enumerate(probs):
            self._regime_probs.probs[i] = pf2d_real(p / total)
        
        self._update_cumprobs()
        _lib.pf2d_build_regime_lut(self._pf, ctypes.byref(self._regime_probs))
    
    def enable_pcg(self, enable: bool = True):
        """Enable PCG random number generator (faster than MKL VSL)."""
        self._check_open()
        _lib.pf2d_enable_pcg(self._pf, 1 if enable else 0)
    
    # ========================================================================
    # Initialization
    # ========================================================================
    
    def initialize(self, price0: float, log_vol0: float,
                   price_spread: float = 0.01, log_vol_spread: float = 0.5):
        """
        Initialize particles around starting values.
        
        Parameters
        ----------
        price0 : float
            Initial price
        log_vol0 : float  
            Initial log-volatility (e.g., np.log(0.01) for 1% vol)
        price_spread : float
            Spread of initial price distribution
        log_vol_spread : float
            Spread of initial log-vol distribution
        """
        self._check_open()
        _lib.pf2d_initialize(
            self._pf,
            pf2d_real(price0),
            pf2d_real(price_spread),
            pf2d_real(log_vol0),
            pf2d_real(log_vol_spread)
        )
    
    def warmup(self):
        """Warmup RNG and prefetch memory."""
        self._check_open()
        _lib.pf2d_warmup(self._pf)
    
    # ========================================================================
    # Core Operations
    # ========================================================================
    
    def update(self, observation: float) -> PF2DResult:
        """
        Process one observation and return estimates.
        
        Parameters
        ----------
        observation : float
            Observed price
        
        Returns
        -------
        PF2DResult
            Estimates and diagnostics
        """
        self._check_open()
        
        # Call C function - returns struct by value
        out = _lib.pf2d_update(
            self._pf,
            pf2d_real(observation),
            ctypes.byref(self._regime_probs)
        )
        
        # Convert to Python types
        regime_probs = np.array([float(out.regime_probs[i]) 
                                  for i in range(self.n_regimes)], dtype=np.float64)
        
        return PF2DResult(
            price_mean=float(out.price_mean),
            price_std=float(np.sqrt(max(0, out.price_variance))),
            log_vol_mean=float(out.log_vol_mean),
            log_vol_std=float(np.sqrt(max(0, out.log_vol_variance))),
            vol_mean=float(out.vol_mean),
            ess=float(out.ess),
            regime_probs=regime_probs,
            dominant_regime=int(out.dominant_regime),
            resampled=bool(out.resampled),
            sigma_vol_scale=float(out.sigma_vol_scale),
            ess_ema=float(out.ess_ema),
            high_vol_mode=bool(out.high_vol_mode),
            regime_feedback_active=bool(out.regime_feedback_active),
            bocpd_cooldown_remaining=int(out.bocpd_cooldown_remaining),
        )
    
    # ========================================================================
    # Batch Processing
    # ========================================================================
    
    def run(self, observations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Process array of observations.
        
        Parameters
        ----------
        observations : array-like
            Price observations
        
        Returns
        -------
        price_est : ndarray
            Price estimates
        vol_est : ndarray
            Volatility estimates
        ess : ndarray
            Effective sample sizes
        """
        self._check_open()
        
        observations = np.asarray(observations, dtype=np.float64)
        n = len(observations)
        
        price_est = np.zeros(n, dtype=np.float64)
        vol_est = np.zeros(n, dtype=np.float64)
        ess = np.zeros(n, dtype=np.float64)
        
        for i, obs in enumerate(observations):
            result = self.update(float(obs))
            price_est[i] = result.price_mean
            vol_est[i] = result.vol_mean
            ess[i] = result.ess
        
        return price_est, vol_est, ess
    
    # ========================================================================
    # Direct Estimates
    # ========================================================================
    
    def price_mean(self) -> float:
        """Get current price estimate."""
        self._check_open()
        return float(_lib.pf2d_price_mean(self._pf))
    
    def vol_mean(self) -> float:
        """Get current volatility estimate."""
        self._check_open()
        return float(_lib.pf2d_vol_mean(self._pf))
    
    def ess(self) -> float:
        """Get effective sample size."""
        self._check_open()
        return float(_lib.pf2d_effective_sample_size(self._pf))
    
    # ========================================================================
    # Debug
    # ========================================================================
    
    def print_config(self):
        """Print filter configuration."""
        self._check_open()
        _lib.pf2d_print_config(self._pf)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_default_filter(n_particles: int = 4000, n_regimes: int = 4) -> ParticleFilter2D:
    """
    Create a ParticleFilter2D with sensible defaults.
    
    Default regime parameters are tuned for daily equity data.
    """
    pf = ParticleFilter2D(n_particles=n_particles, n_regimes=n_regimes)
    
    # Default observation variance
    pf.set_observation_variance(0.0001)
    
    # Default regime parameters (daily equity)
    defaults = [
        # (drift, theta, mu_vol, sigma_vol, rho)
        (0.0001, 0.02, np.log(0.01), 0.05, -0.3),   # Regime 0: Low vol
        (0.0000, 0.03, np.log(0.015), 0.08, -0.4),  # Regime 1: Normal
        (-0.0001, 0.05, np.log(0.025), 0.12, -0.5), # Regime 2: High vol
        (-0.0002, 0.08, np.log(0.04), 0.20, -0.6),  # Regime 3: Crisis
    ]
    
    for i in range(min(n_regimes, len(defaults))):
        drift, theta, mu_vol, sigma_vol, rho = defaults[i]
        pf.set_regime_params(i, drift, theta, mu_vol, sigma_vol, rho)
    
    # Equal prior
    pf.set_regime_probs([1.0 / n_regimes] * n_regimes)
    
    return pf


# ============================================================================
# STRUCT SIZE VERIFICATION
# ============================================================================

def verify_struct_sizes():
    """Verify Python struct sizes match expected C sizes."""
    output_size = ctypes.sizeof(PF2DOutput)
    regime_size = ctypes.sizeof(PF2DRegimeProbs)
    
    print(f"PF2DOutput size: {output_size} bytes (expected ~{_EXPECTED_OUTPUT_SIZE})")
    print(f"PF2DRegimeProbs size: {regime_size} bytes (expected ~{_EXPECTED_REGIME_PROBS_SIZE})")
    print(f"pf2d_real size: {ctypes.sizeof(pf2d_real)} bytes")
    
    # Check alignment
    print(f"\nPF2DOutput fields:")
    for name, ctype in PF2DOutput._fields_:
        print(f"  {name}: {ctypes.sizeof(ctype)} bytes")


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("PF2D Python Bindings Test")
    print("=" * 50)
    
    # Verify struct sizes
    verify_struct_sizes()
    print()
    
    # Test filter
    print("Creating filter...")
    with create_default_filter(n_particles=4000) as pf:
        pf.initialize(price0=100.0, log_vol0=np.log(0.01),
                      price_spread=0.01, log_vol_spread=0.5)
        pf.warmup()
        
        print("Running 100 updates...")
        np.random.seed(42)
        price = 100.0
        
        for i in range(100):
            price += 0.01 * np.random.randn()
            result = pf.update(price)
            
            if i % 25 == 0:
                print(f"  Tick {i}: price={result.price_mean:.2f}, "
                      f"vol={result.vol_mean:.4f}, ess={result.ess:.0f}")
        
        print(f"\nFinal: price={result.price_mean:.2f}, vol={result.vol_mean:.4f}")
    
    print("\n✓ All tests passed!")
