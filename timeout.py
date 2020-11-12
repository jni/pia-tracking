from multiprocessing import Process, Queue
import time


def timeout(func, timeout=3600):
    """
    Submit a function and its kwargs to a dask client. Wait for a
    number of seconds before cancelling the future and closing the
    client.

    Parameters
    ----------
    func: function
    kw: dict
        key word arguments required to run said function
    timeout: int
        number of seconds for which to wait before cancelling the future

    Notes
    -----
    Decorator
    """
    def inner(*args, **kwargs):
        q = Queue()
        p = Process(target=send_to_queue(func, q), args=args, kwargs=kwargs)
        p.start()
        t0 = time.time()
        result = None
        while time.time() - t0 <= timeout:
            time.sleep(1)
            if q.empty():
                pass
            else:
                result = q.get()
                break
        else:
            p.terminate()
            p.join()
            print('Your process took too long and was timed out')
        return result
    return inner


def send_to_queue(func, q):
    def inner(*args, **kwargs):
        result = func(*args, *kwargs)
        q.put(result)
        return result
    return inner


# this is a test
if __name__ == "__main__":
    from toolz import curry

    @timeout
    def add_sleep(a, b):
        time.sleep(11)
        return a + b