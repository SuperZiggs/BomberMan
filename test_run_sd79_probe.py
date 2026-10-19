import unittest

import run_sd79_probe as sd79


class ProbeTests(unittest.TestCase):
    def test_decide_threshold(self):
        def paired(diff):
            return {'S_search': {'active_win': {'difference_pp': diff}}}
        self.assertTrue(sd79.decide(paired(2.0))['open_test_47000'])
        self.assertFalse(sd79.decide(paired(1.75))['open_test_47000'])

    def test_arms(self):
        self.assertEqual({k: v[1] for k, v in sd79.ARMS.items()}, {'G_search': 1, 'S': 0, 'S_search': 1, 'G_search2': 2})
        self.assertTrue(sd79.decide({'G_search2': {'active_win': {'difference_pp': 2.5}}}, 'G_search2')['open_test_47000'])


if __name__ == '__main__':
    unittest.main()
