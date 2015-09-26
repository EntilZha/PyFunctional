import unittest
from .. import seq


class TestStreams(unittest.TestCase):
    def test_open(self):
        with open('LICENSE.txt') as f:
            data = f.readlines()
        self.assertListEqual(data, seq.open('LICENSE.txt').to_list())

        text = ''.join(data).split(',')
        self.assertListEqual(text, seq.open('LICENSE.txt', delimiter=',').to_list())

        with self.assertRaises(ValueError):
            seq.open('LICENSE.txt', mode='w').to_list()

    def test_range(self):
        self.assertListEqual([0, 1, 2, 3], seq.range(4).to_list())

        data = [-5, -3, -1, 1, 3, 5, 7]
        self.assertListEqual(data, seq.range(-5, 8, 2).to_list())

    def test_csv(self):
        result = seq.csv('functional/test/data/test.csv').to_list()
        expect = [['1', '2', '3', '4'], ['a', 'b', 'c', 'd']]
        self.assertEqual(expect, result)
        with open('functional/test/data/test.csv', 'r') as csv_file:
            self.assertEqual(expect, seq.csv(csv_file).to_list())
        with self.assertRaises(ValueError):
            seq.csv(1)
