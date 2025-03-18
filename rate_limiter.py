from limits import strategies, parse, storage
import time
import asyncio

from adalflow import get_logger

logger = get_logger(__name__, level="INFO", enable_file=True)


fixed_window = strategies.FixedWindowRateLimiter(storage.MemoryStorage())
GOOGLE_GENAI_LIMITS = parse("15/minute")

async def async_rate_limited_call(identifier, fn, *args, **kwargs):
    """
    A simple rate limiter that enforces a minimum delay between calls.
    """
    if fixed_window.hit(GOOGLE_GENAI_LIMITS, identifier):
        return await fn(*args, **kwargs)
    else:
        reset_time, remaining = fixed_window.get_window_stats(GOOGLE_GENAI_LIMITS, identifier)
        logger.info(f"Rate limit exceeded for {identifier}. Reset time: {reset_time}, Remaining: {remaining}")
        time.sleep(reset_time - time.time())
        return await async_rate_limited_call(identifier, fn, *args, **kwargs)

def rate_limited_call(identifier, fn, *args, **kwargs):
    """
    A simple rate limiter that enforces a minimum delay between calls.
    """
    if fixed_window.hit(GOOGLE_GENAI_LIMITS, identifier):
        # if the function is a coroutine, we need to await it
        if asyncio.iscoroutinefunction(fn):
            result = fn(*args, **kwargs)
            return result
        else:
            return fn(*args, **kwargs)
    else:
        reset_time, remaining = fixed_window.get_window_stats(GOOGLE_GENAI_LIMITS, identifier)
        logger.info(f"Rate limit exceeded for {identifier}. Reset time: {reset_time}, Remaining: {remaining}")
        time.sleep(reset_time - time.time())
        return rate_limited_call(identifier, fn, *args, **kwargs)