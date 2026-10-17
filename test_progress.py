import unittest
from unittest.mock import patch

import progress


def ticking(value, ticks):
    for _ in range(value):
        ticks.put(1)
    return value * 10


class FakeBar:
    def __init__(self, total, desc, initial=0):
        self.n, self.total = initial, total

    def update(self, n=1):
        self.n += n

    def __enter__(self):
        return self

    def __exit__(self, *args):
        FakeBar.last = self


class ProgressTests(unittest.TestCase):
    def test_parallel_map_keeps_job_order_and_counts_every_tick(self):
        with patch.object(progress, 'bar', FakeBar):
            results = progress.parallel_map(ticking, [(3,), (1,), (2,)], total=6, desc='x', workers=2)
        self.assertEqual(results, [30, 10, 20])
        self.assertEqual(FakeBar.last.n, 6)

    def test_bar_counts_from_the_resumed_position(self):
        with progress.bar(10, 'resume', initial=4) as b:
            b.update(2)
            self.assertEqual(b.n, 6)


if __name__ == '__main__':
    unittest.main()
