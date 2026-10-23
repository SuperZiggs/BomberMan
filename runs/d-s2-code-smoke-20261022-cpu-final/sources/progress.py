"""Progress bars that stay visible when a run's stdout/stderr go to a log (cmd `> log 2>&1`).

The bar is written to the console device (CON on Windows, /dev/tty elsewhere), not to the log;
without a console it falls back to stderr.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import queue
import sys

from tqdm import tqdm


def _console():
    for name in ('CON', '/dev/tty'):
        try:
            return open(name, 'w', encoding='utf-8')
        except OSError:
            pass
    return sys.stderr


def bar(total, desc, initial=0):
    return tqdm(total=total, desc=desc, initial=initial, file=_console(), dynamic_ncols=True,
                mininterval=1.0, smoothing=0.05)


def parallel_map(fn, jobs, total, desc, workers):
    """Run fn(*job, ticks) in worker processes; each ticks.put(n) advances one shared bar."""
    with multiprocessing.Manager() as manager, ProcessPoolExecutor(workers) as pool, bar(total, desc) as shown:
        ticks = manager.Queue()
        futures = [pool.submit(fn, *job, ticks) for job in jobs]
        while not all(f.done() for f in futures):
            try:
                shown.update(ticks.get(timeout=0.5))
            except queue.Empty:
                pass
        while not ticks.empty():
            shown.update(ticks.get())
        return [f.result() for f in futures]
