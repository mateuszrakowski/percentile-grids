import hashlib
import inspect
import json
import os
import pickle
from datetime import datetime
from typing import Any, Callable


def disk_cache(cache_dir: str = ".cache"):
    os.makedirs(cache_dir, exist_ok=True)
    memory_cache: dict[str, Any] = {}

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            key = generate_cache_key(func, args, kwargs)
            cache_file = os.path.join(cache_dir, f"{key}.pickle")

            if key in memory_cache:
                return memory_cache[key]

            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "rb") as f:
                        result = pickle.load(f)
                    memory_cache[key] = result
                    return result
                except (pickle.PickleError, EOFError):
                    pass

            result = func(*args, **kwargs)
            memory_cache[key] = result

            try:
                with open(cache_file, "wb") as f:
                    pickle.dump(result, f)
            except (pickle.PickleError, EOFError):
                pass

            return result

        return wrapper

    return decorator


def generate_cache_key(func, args, kwargs):
    func_id = f"{func.__module__}.{func.__name__}"

    try:
        source_code = inspect.getsource(func)
    except (IOError, TypeError):
        source_code = ""

    try:
        args_str = str(args)
        kwargs_str = str(sorted(kwargs.items()))
    except Exception:
        args_str = str([str(arg) for arg in args])
        kwargs_str = str([(k, str(v)) for k, v in sorted(kwargs.items())])

    key_string = f"{func_id}:{source_code}:{args_str}:{kwargs_str}"

    return hashlib.md5(key_string.encode("utf-8")).hexdigest()
