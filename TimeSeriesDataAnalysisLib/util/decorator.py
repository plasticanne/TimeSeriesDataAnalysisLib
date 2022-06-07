from functools import wraps
def response_keys_filter(key_name:str):
    """filter return value to key_name list
    """
    def outterWrapper(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return [d[key_name] for d in result]
        return wrapper
    return outterWrapper