import os
import functools
from contextlib import contextmanager


@contextmanager
def disable_flash_attention():
    old = os.environ.get("DISABLE_FLASH_ATTN")
    os.environ["DISABLE_FLASH_ATTN"] = "1"
    try:
        yield
    finally:
        if old is None:
            del os.environ["DISABLE_FLASH_ATTN"]
        else:
            os.environ["DISABLE_FLASH_ATTN"] = old


def disable_flash_attention_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with disable_flash_attention():
            return func(*args, **kwargs)

    return wrapper
