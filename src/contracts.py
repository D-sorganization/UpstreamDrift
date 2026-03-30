from collections.abc import Callable


def require(condition: Callable[..., bool], message: str = "Precondition failed"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not condition(*args, **kwargs):
                raise ValueError(message)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def ensure(condition: Callable[..., bool], message: str = "Postcondition failed"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if not condition(result):
                raise RuntimeError(message)
            return result

        return wrapper

    return decorator
