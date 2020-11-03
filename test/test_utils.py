import unittest
from greippi import utils


class TestUtils(unittest.TestCase):
    def test_mcnemar_same(self):
        t = utils.mcnemars_t(['1', '2', '3'], ['1', '1', '1'], ['1', '1', '1'])
        self.assertEqual(t, float('-inf'))

    def test_mcnemar_different(self):
        t = utils.mcnemars_t(['1', '2', '3'] * 50, ['1', '2', '3'] * 50, ['3', '1', '2'] * 50)
        self.assertGreater(t, 3.841459)

    def test_macro_acc(self):
        truth = ['a', 'b', 'b', 'b', 'b']
        predicted = ['a', 'a', 'a', 'a', 'a']
        self.assertEqual(utils.get_macro_acc(truth, predicted), 0.5)

    def test_accuracy_ci(self):
        acc1, ci1 = utils.get_accuracy_ci(0, 1)
        acc2, ci2 = utils.get_accuracy_ci(1, 1)
        self.assertGreater(acc1, 0)
        self.assertLess(acc2, 1)