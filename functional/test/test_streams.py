from __future__ import absolute_import

import unittest
import six
from functional import seq


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
        result_0 = seq.jsonl('functional/test/data/test.jsonl').to_list()
        expect_0 = [[1, 2, 3], {'a': 1, 'b': 2, 'c': 3}]
        self.assertEqual(expect_0, result_0)
        result_1 = seq.jsonl(['[1, 2, 3]', '[4, 5, 6]'])
        expect_1 = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(expect_1, result_1)

    def test_json(self):
        list_test_path = 'functional/test/data/test_list.json'
        dict_test_path = 'functional/test/data/test_dict.json'
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list(six.viewitems({u'a': 1, u'b': 2, u'c': 3}))

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

    def test_to_file(self):
        tmp_path = 'functional/test/data/tmp/output.txt'
        sequence = seq(1, 2, 3, 4)
        sequence.to_file(tmp_path)
        with open(tmp_path, 'r') as output:
            self.assertEqual('[1, 2, 3, 4]', output.readlines()[0])

        sequence.to_file(tmp_path, delimiter=":")
        with open(tmp_path, 'r') as output:
            self.assertEqual('1:2:3:4', output.readlines()[0])

    def test_to_jsonl(self):
        tmp_path = 'functional/test/data/tmp/output.txt'
        elements = [{'a': 1, 'b': 2}, {'c': 3}, {'d': 4}]
        sequence = seq(elements)
        sequence.to_jsonl(tmp_path)
        result = seq.jsonl(tmp_path).to_list()
        self.assertEqual(elements, result)

    def test_to_json(self):
        tmp_path = 'functional/test/data/tmp/output.txt'
        elements = [[u'a', 1], [u'b', 2], [u'c', 3]]
        sequence = seq(elements)
        sequence.to_json(tmp_path)
        result = seq.json(tmp_path).to_list()
        self.assertEqual(elements, result)

        dict_expect = {u'a': 1, u'b': 2, u'c': 3}
        sequence.to_json(tmp_path, root_array=False)
        result = seq.json(tmp_path).to_dict()
        self.assertEqual(dict_expect, result)

    def test_to_csv(self):
        tmp_path = 'functional/test/data/tmp/output.txt'
        elements = [[1, 2, 3], [4, 5, 6], ['a', 'b', 'c']]
        expect = [['1', '2', '3'], ['4', '5', '6'], ['a', 'b', 'c']]
        sequence = seq(elements)
        sequence.to_csv(tmp_path)
        result = seq.csv(tmp_path).to_list()
        self.assertEqual(expect, result)
