from datetime import datetime
import time
import functools

"""
timer.py — Utility functions for timing and benchmarking.
"""

def timer(func):
    """Decorator to time a function and print its runtime."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()

        runtime = end - start
        runtime_strfmt = "HH:MM:SS" if runtime >= 3600 else "MM:SS" if runtime >= 60 else "SS.sss"

        if runtime_strfmt == "HH:MM:SS":
            runtime_fmt = time.strftime("%H:%M:%S", time.gmtime(runtime))
        elif runtime_strfmt == "MM:SS":
            runtime_fmt = time.strftime("%M:%S", time.gmtime(runtime))
        else:
            runtime_fmt = f"{runtime:.3f} seconds"

        print(f"Function '{func.__name__}' executed in {runtime_fmt}")
        return result
    return wrapper

if __name__ == "__main__":
    print(f"Current time: {datetime.now()}")
    time.sleep(1) 