import unittest
from .. import seq
from .. import util


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

    def test_jsonl(self):
        result = seq.jsonl('functional/test/data/test.jsonl').to_list()
        expect = [[1, 2, 3], {'a': 1, 'b': 2, 'c': 3}]
        self.assertEqual(expect, result)
        result = seq.jsonl(['[1, 2, 3]', '[4, 5, 6]'])
        expect = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(expect, result)

    def test_json(self):
        list_test_path = 'functional/test/data/test_list.json'
        dict_test_path = 'functional/test/data/test_dict.json'
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list(util.dict_item_iter({u'a': 1, u'b': 2, u'c': 3}))

        result = seq.json(list_test_path).to_list()
        self.assertEqual(list_expect, result)
        result = seq.json(dict_test_path).to_list()
        self.assertEqual(dict_expect, result)

        with open(list_test_path) as file_handle:
            result = seq.json(file_handle).to_list()
            self.assertEqual(list_expect, result)
        with open(dict_test_path) as file_handle:
            result = seq.json(file_handle).to_list()
            self.assertEqual(dict_expect, result)

        with self.assertRaises(ValueError):
            seq.json(1)
