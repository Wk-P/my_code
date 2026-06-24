"""
timer_utils.py — Simple timing decorator used by all run_all.py scripts.
"""

import time
import functools
from datetime import datetime


def timer(func):
    """Decorator: prints elapsed wall-clock time after the wrapped function returns."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.time()
        print(f"[timer] {func.__name__} started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        mins, secs = divmod(elapsed, 60)
        print(f"\n[timer] {func.__name__} finished in {int(mins)}m {secs:.1f}s [{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]")
        return result
    return wrapper
