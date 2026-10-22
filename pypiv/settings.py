from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Optional, Tuple
 
import numpy as np
 
 
def _no_validation(u, v, s2n):
    flags = np.zeros_like(u, dtype=bool)
    return flags
 
 
@dataclass(init=True, repr=True, frozen=True, kw_only=True)
class CommonPIVSettings:
    # ------------------------------------------------------------------------------
    # CROSS-CORRELATION
    # ------------------------------------------------------------------------------
    backend: str = "auto"  # 'auto', 'scipy' or 'rust'. 'rust' needs the optional openpiv-rust package.
    correlation_method: str = "circular"  # 'circular' or 'linear' (pyprocess also accepts a few aliases such as 'fft').
    normalized_correlation: bool = False
 
    # ------------------------------------------------------------------------------
    # SUB-PIXEL PEAK
    # ------------------------------------------------------------------------------
    subpixel_method: str = "gaussian"
    use_vectorized: bool = False

    # ------------------------------------------------------------------------------
    # S/N RATIO
    # ------------------------------------------------------------------------------
    sig2noise_method: Optional[str] = "peak2peak"
    sig2noise_width: int = 2
 
    # ------------------------------------------------------------------------------
    # VECTOR VALIDATION
    # ------------------------------------------------------------------------------
    validation_function: Callable = _no_validation # must be a function that takes arguments u, v, and s2n
 
    # ------------------------------------------------------------------------------
    # OUTLIER REPLACEMENT
    # ------------------------------------------------------------------------------
    filter_method: str = "localmean"  #  'localmean', 'disk' or 'distance'.
    max_filter_iteration: int = 5
    filter_kernel_size: int = 2
 
 
@dataclass(init=True, repr=True, frozen=True, kw_only=True)
class SimplePIVSettings(CommonPIVSettings):
    # ------------------------------------------------------------------------------
    # PIV GRID
    # ------------------------------------------------------------------------------
    search_area_size: Optional[int] = None  # None: same as window_size
    window_size: int = 32
    overlap: int = 16

 
    def __post_init__(self):
        object.__setattr__(self, 'search_area_size',  self.window_size if self.search_area_size is None else self.search_area_size)

        if self.window_size <= 0 or self.overlap < 0:
            raise ValueError("window_size must be > 0 and overlap >= 0.")
 
        if self.search_area_size < self.window_size:
            raise ValueError("search_area_size cannot be smaller than window_size.")
        
        if self.overlap >= self.search_area_size:
            raise ValueError("overlap must be smaller than the search area "
                             "(window_size, or search_area_size if given).")
 
 
@dataclass(init=True, repr=True, frozen=True, kw_only=True)
class WindefPIVSettings(CommonPIVSettings):
    # ------------------------------------------------------------------------------
    # MULTIPASS GRID (one entry per pass, coarse to fine)
    # ------------------------------------------------------------------------------
    window_sizes: Tuple[int, ...] = (64, 32)
    overlaps: Tuple[int, ...] = (32, 16)

    @property
    def num_iterations(self) -> int:
        return len(self.window_sizes)
 
    # ------------------------------------------------------------------------------
    # Image deformation
    # ------------------------------------------------------------------------------
    interpolation_order: int = 3
 
    # ------------------------------------------------------------------------------
    # Smoothing
    # ------------------------------------------------------------------------------
    use_smoothn: bool = False
    smoothn_s: float = 0.05

 
    def __post_init__(self):
        if isinstance(self.interpolation_order, bool) or not isinstance(self.interpolation_order, int):
            raise TypeError("interpolation_order must be an integer.")
        
        if not 1 <= self.interpolation_order <= 5:
            raise ValueError("interpolation_order must be between 1 and 5.")

        if len(self.window_sizes) == 0:
            raise ValueError("window_sizes must contain at least one pass.")
        
        if len(self.overlaps) != len(self.window_sizes):
            raise ValueError(f"overlaps has {len(self.overlaps)} entries but "
                             f"window_sizes has {len(self.window_sizes)}; "
                             "one overlap value is required per pass.")


if __name__ == "__main__":
    settings = WindefPIVSettings()
    print(type(settings))
    print(settings)