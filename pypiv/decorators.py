import functools

def animatable(animate_func):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args, animate=False, animate_kwargs=None, **kwargs):
            gen = f(self, *args, **kwargs)
            if not animate:
                return gen
            return getattr(self, animate_func)(gen, **(animate_kwargs or {}))
        return wrapper
    return decorator


def saveable(save_func):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(self, *args, save_at=None, **kwargs):
            gen = f(self, *args, **kwargs)
            if save_at is None:
                return gen
            return getattr(self, save_func)(gen, save_at)
        return wrapper
    return decorator