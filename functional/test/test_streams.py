import sqlite3
import unittest
import collections
import sys
import gzip
from platform import system
import lzma
import bz2

from functional import seq, pseq
from functional.streams import Stream, ParallelStream


class TestStreams(unittest.TestCase):
    def setUp(self):
        self.seq = seq
        self.seq_c_disabled = Stream(disable_compression=True)

    def test_open(self):
        with open("LICENSE.txt", encoding="utf8") as f:
            data = f.readlines()
        self.assertListEqual(data, self.seq.open("LICENSE.txt").to_list())

        text = "".join(data).split(",")
        self.assertListEqual(
            text, self.seq.open("LICENSE.txt", delimiter=",").to_list()
        )

        with self.assertRaises(ValueError):
            self.seq.open("LICENSE.txt", mode="w").to_list()

    def test_open_gzip(self):
        expect = ["line0\n", "line1\n", "line2"]
        self.assertListEqual(
            expect,
            self.seq.open("functional/test/data/test.txt.gz", mode="rt").to_list(),
        )

    def test_open_bz2(self):
        expect = ["line0\n", "line1\n", "line2"]
        self.assertListEqual(
            expect,
            self.seq.open("functional/test/data/test.txt.bz2", mode="rt").to_list(),
        )

    def test_open_xz(self):
        expect = ["line0\n", "line1\n", "line2"]
        self.assertListEqual(
            expect,
            self.seq.open("functional/test/data/test.txt.xz", mode="rt").to_list(),
        )

    def test_disable_compression(self):
        file_name = "functional/test/data/test.txt.gz"
        with open(file_name, "rb") as f:
            expect = f.readlines()
        self.assertListEqual(
            expect, self.seq_c_disabled.open(file_name, mode="rb").to_list()
        )

    def test_range(self):
        self.assertListEqual([0, 1, 2, 3], self.seq.range(4).to_list())

        data = [-5, -3, -1, 1, 3, 5, 7]
        self.assertListEqual(data, self.seq.range(-5, 8, 2).to_list())

    def test_lazyiness(self):
        def yielder():
            nonlocal step
            step += 1
            yield 1
            step += 1
            yield 2

        step = 0
        sequence = iter(seq(yielder()).map(str))
        assert (
            step == 0
            and next(sequence) == "1"
            and step == 1
            and next(sequence) == "2"
            and step == 2
        )

        step = 0
        sequence = iter(seq.chain(yielder()).map(str))
        assert (
            step == 0
            and next(sequence) == "1"
            and step == 1
            and next(sequence) == "2"
            and step == 2
        )

    def test_chain(self):
        data_a = range(1, 5)
        data_b = range(6, 11)
        self.assertEqual(
            list(data_a) + list(data_b), self.seq.chain(data_a, data_b).to_list()
        )

        data_c = set(data_b)
        self.assertEqual(
            list(data_a) + list(data_c), self.seq.chain(data_a, data_c).to_list()
        )

        data_d = {"a": 1, "b": 2}
        self.assertEqual(
            list(data_a) + list(data_d.keys()), self.seq.chain(data_a, data_d).to_list()
        )

        self.assertEqual([], self.seq.chain().to_list())

        with self.assertRaises(TypeError):
            self.seq.chain(1, 2).to_list()

        self.assertEqual(list(data_a), self.seq.chain(data_a).to_list())

        self.assertEqual([1], self.seq.chain([1]).to_list())

    def test_csv(self):
        result = self.seq.csv("functional/test/data/test.csv").to_list()
        expect = [["1", "2", "3", "4"], ["a", "b", "c", "d"]]
        self.assertEqual(expect, result)
        with open("functional/test/data/test.csv", "r", encoding="utf8") as csv_file:
            self.assertEqual(expect, self.seq.csv(csv_file).to_list())
        with self.assertRaises(ValueError):
            self.seq.csv(1)

    def test_csv_dict_reader(self):
        result = self.seq.csv_dict_reader(
            "functional/test/data/test_header.csv"
        ).to_list()
        self.assertEqual(result[0]["a"], "1")
        self.assertEqual(result[0]["b"], "2")
        self.assertEqual(result[0]["c"], "3")
        self.assertEqual(result[1]["a"], "4")
        self.assertEqual(result[1]["b"], "5")
        self.assertEqual(result[1]["c"], "6")

        with open("functional/test/data/test_header.csv", "r", encoding="utf8") as f:
            result = self.seq.csv_dict_reader(f).to_list()
        self.assertEqual(result[0]["a"], "1")
        self.assertEqual(result[0]["b"], "2")
        self.assertEqual(result[0]["c"], "3")
        self.assertEqual(result[1]["a"], "4")
        self.assertEqual(result[1]["b"], "5")
        self.assertEqual(result[1]["c"], "6")

        with self.assertRaises(ValueError):
            self.seq.csv_dict_reader(1)

    def test_gzip_csv(self):
        result = self.seq.csv("functional/test/data/test.csv.gz").to_list()
        expect = [["1", "2", "3", "4"], ["a", "b", "c", "d"]]
        self.assertEqual(expect, result)
        with self.assertRaises(ValueError):
            self.seq.csv(1)

    def test_bz2_csv(self):
        result = self.seq.csv("functional/test/data/test.csv.bz2").to_list()
        expect = [["1", "2", "3", "4"], ["a", "b", "c", "d"]]
        self.assertEqual(expect, result)
        with self.assertRaises(ValueError):
            self.seq.csv(1)

    def test_xz_csv(self):
        result = self.seq.csv("functional/test/data/test.csv.xz").to_list()
        expect = [["1", "2", "3", "4"], ["a", "b", "c", "d"]]
        self.assertEqual(expect, result)
        with self.assertRaises(ValueError):
            self.seq.csv(1)

    def test_jsonl(self):
        result_0 = self.seq.jsonl("functional/test/data/test.jsonl").to_list()
        expect_0 = [[1, 2, 3], {"a": 1, "b": 2, "c": 3}]
        self.assertEqual(expect_0, result_0)
        result_1 = self.seq.jsonl(["[1, 2, 3]", "[4, 5, 6]"])
        expect_1 = [[1, 2, 3], [4, 5, 6]]
        self.assertEqual(expect_1, result_1)

    def test_gzip_jsonl(self):
        result_0 = self.seq.jsonl("functional/test/data/test.jsonl.gz").to_list()
        expect_0 = [[1, 2, 3], {"a": 1, "b": 2, "c": 3}]
        self.assertEqual(expect_0, result_0)

    def test_bz2_jsonl(self):
        result_0 = self.seq.jsonl("functional/test/data/test.jsonl.bz2").to_list()
        expect_0 = [[1, 2, 3], {"a": 1, "b": 2, "c": 3}]
        self.assertEqual(expect_0, result_0)

    def test_xz_jsonl(self):
        result_0 = self.seq.jsonl("functional/test/data/test.jsonl.xz").to_list()
        expect_0 = [[1, 2, 3], {"a": 1, "b": 2, "c": 3}]
        self.assertEqual(expect_0, result_0)

    def test_json(self):
        list_test_path = "functional/test/data/test_list.json"
        dict_test_path = "functional/test/data/test_dict.json"
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list({"a": 1, "b": 2, "c": 3}.items())

        result = self.seq.json(list_test_path).to_list()
        self.assertEqual(list_expect, result)
        result = self.seq.json(dict_test_path).to_list()
        self.assertEqual(dict_expect, result)

        with open(list_test_path, encoding="utf8") as file_handle:
            result = self.seq.json(file_handle).to_list()
            self.assertEqual(list_expect, result)
        with open(dict_test_path, encoding="utf8") as file_handle:
            result = self.seq.json(file_handle).to_list()
            self.assertEqual(dict_expect, result)

        with self.assertRaises(ValueError):
            self.seq.json(1)

    def test_gzip_json(self):
        list_test_path = "functional/test/data/test_list.json.gz"
        dict_test_path = "functional/test/data/test_dict.json.gz"
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list({"a": 1, "b": 2, "c": 3}.items())

        result = self.seq.json(list_test_path).to_list()
        self.assertEqual(list_expect, result)
        result = self.seq.json(dict_test_path).to_list()
        self.assertEqual(dict_expect, result)

        with self.assertRaises(ValueError):
            self.seq.json(1)

    def test_bz2_json(self):
        list_test_path = "functional/test/data/test_list.json.bz2"
        dict_test_path = "functional/test/data/test_dict.json.bz2"
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list({"a": 1, "b": 2, "c": 3}.items())

        result = self.seq.json(list_test_path).to_list()
        self.assertEqual(list_expect, result)
        result = self.seq.json(dict_test_path).to_list()
        self.assertEqual(dict_expect, result)

        with self.assertRaises(ValueError):
            self.seq.json(1)

    def test_xz_json(self):
        list_test_path = "functional/test/data/test_list.json.xz"
        dict_test_path = "functional/test/data/test_dict.json.xz"
        list_expect = [1, 2, 3, 4, 5]
        dict_expect = list({"a": 1, "b": 2, "c": 3}.items())

        result = self.seq.json(list_test_path).to_list()
        self.assertEqual(list_expect, result)
        result = self.seq.json(dict_test_path).to_list()
        self.assertEqual(dict_expect, result)

        with self.assertRaises(ValueError):
            self.seq.json(1)

    def test_sqlite3(self):
        db_file = "functional/test/data/test_sqlite3.db"

        # test failure case
        with self.assertRaises(ValueError):
            self.seq.sqlite3(1, "SELECT * from user").to_list()

        # test select from file path
        query_0 = "SELECT id, name FROM user;"
        result_0 = self.seq.sqlite3(db_file, query_0).to_list()
        expected_0 = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]
        self.assertListEqual(expected_0, result_0)

        # test select from connection
        with sqlite3.connect(db_file) as conn:
            result_0_1 = self.seq.sqlite3(conn, query_0).to_list()
            self.assertListEqual(expected_0, result_0_1)

        # test select from cursor
        with sqlite3.connect(db_file) as conn:
            cursor = conn.cursor()
            result_0_2 = self.seq.sqlite3(cursor, query_0).to_list()
            self.assertListEqual(expected_0, result_0_2)

        # test connection with kwds
        result_0_3 = self.seq.sqlite3(db_file, query_0, timeout=30).to_list()
        self.assertListEqual(expected_0, result_0_3)

        # test order by
        result_1 = self.seq.sqlite3(
            db_file, "SELECT id, name FROM user ORDER BY name;"
        ).to_list()
        expected_1 = [(2, "Jack"), (3, "Jane"), (4, "Stephan"), (1, "Tom")]
        self.assertListEqual(expected_1, result_1)

        # test query with params
        result_2 = self.seq.sqlite3(
            db_file, "SELECT id, name FROM user WHERE id = ?;", parameters=(1,)
        ).to_list()
        expected_2 = [(1, "Tom")]
        self.assertListEqual(expected_2, result_2)

    def test_pandas(self):
        try:
            import pandas

            data = pandas.DataFrame([[1, 3], [4, 5]])
            result = seq(data).list()
            self.assertEqual(result[0][0], 1)
            self.assertEqual(result[0][1], 3)
            self.assertEqual(result[1][0], 4)
            self.assertEqual(result[1][1], 5)
        except ImportError:
            pass

    def test_to_file(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        sequence = self.seq(1, 2, 3, 4)
        sequence.to_file(tmp_path)
        with open(tmp_path, "r", encoding="utf8") as output:
            self.assertEqual("[1, 2, 3, 4]", output.readlines()[0])

        sequence.to_file(tmp_path, delimiter=":")
        with open(tmp_path, "r", encoding="utf8") as output:
            self.assertEqual("1:2:3:4", output.readlines()[0])

    def test_to_file_compressed(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        sequence = self.seq(1, 2, 3, 4)
        sequence.to_file(tmp_path, compression="gzip")
        with gzip.open(tmp_path, "rt") as output:
            self.assertEqual("[1, 2, 3, 4]", output.readlines()[0])

        sequence.to_file(tmp_path, compression="lzma")
        with lzma.open(tmp_path, "rt") as output:
            self.assertEqual("[1, 2, 3, 4]", output.readlines()[0])

        sequence.to_file(tmp_path, compression="bz2")
        with bz2.open(tmp_path, "rt") as output:
            self.assertEqual("[1, 2, 3, 4]", output.readlines()[0])

    def test_to_jsonl(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [{"a": 1, "b": 2}, {"c": 3}, {"d": 4}]
        sequence = self.seq(elements)

        sequence.to_jsonl(tmp_path)
        result = self.seq.jsonl(tmp_path).to_list()
        self.assertEqual(elements, result)

    def test_to_jsonl_compressed(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [{"a": 1, "b": 2}, {"c": 3}, {"d": 4}]
        sequence = self.seq(elements)

        sequence.to_jsonl(tmp_path, compression="gzip")
        result = self.seq.jsonl(tmp_path).to_list()
        self.assertEqual(elements, result)

        sequence.to_jsonl(tmp_path, compression="lzma")
        result = self.seq.jsonl(tmp_path).to_list()
        self.assertEqual(elements, result)

        sequence.to_jsonl(tmp_path, compression="bz2")
        result = self.seq.jsonl(tmp_path).to_list()
        self.assertEqual(elements, result)

    def test_to_json(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [["a", 1], ["b", 2], ["c", 3]]
        sequence = self.seq(elements)

        sequence.to_json(tmp_path)
        result = self.seq.json(tmp_path).to_list()
        self.assertEqual(elements, result)

        dict_expect = {"a": 1, "b": 2, "c": 3}
        sequence.to_json(tmp_path, root_array=False)
        result = self.seq.json(tmp_path).to_dict()
        self.assertEqual(dict_expect, result)

    def test_to_json_compressed(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [["a", 1], ["b", 2], ["c", 3]]
        dict_expect = {"a": 1, "b": 2, "c": 3}
        sequence = self.seq(elements)

        sequence.to_json(tmp_path, compression="gzip")
        result = self.seq.json(tmp_path).to_list()
        self.assertEqual(elements, result)

        sequence.to_json(tmp_path, root_array=False, compression="gzip")
        result = self.seq.json(tmp_path).to_dict()
        self.assertEqual(dict_expect, result)

        sequence.to_json(tmp_path, compression="lzma")
        result = self.seq.json(tmp_path).to_list()
        self.assertEqual(elements, result)

        sequence.to_json(tmp_path, root_array=False, compression="lzma")
        result = self.seq.json(tmp_path).to_dict()
        self.assertEqual(dict_expect, result)

        sequence.to_json(tmp_path, compression="bz2")
        result = self.seq.json(tmp_path).to_list()
        self.assertEqual(elements, result)

        sequence.to_json(tmp_path, root_array=False, compression="bz2")
        result = self.seq.json(tmp_path).to_dict()
        self.assertEqual(dict_expect, result)

    def test_to_csv(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [[1, 2, 3], [4, 5, 6], ["a", "b", "c"]]
        expect = [["1", "2", "3"], ["4", "5", "6"], ["a", "b", "c"]]
        sequence = self.seq(elements)
        sequence.to_csv(tmp_path)
        result = self.seq.csv(tmp_path).to_list()
        self.assertEqual(expect, result)

    @unittest.skipUnless(system().startswith("Win"), "Skip CSV test if not on Windows")
    def test_to_csv_win(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [[1, 2, 3], [4, 5, 6], ["a", "b", "c"]]
        expect = [["1", "2", "3"], [], ["4", "5", "6"], [], ["a", "b", "c"], []]
        sequence = self.seq(elements)
        sequence.to_csv(tmp_path)
        result = self.seq.csv(tmp_path).to_list()
        self.assertNotEqual(expect, result)

    def test_to_csv_compressed(self):
        tmp_path = "functional/test/data/tmp/output.txt"
        elements = [[1, 2, 3], [4, 5, 6], ["a", "b", "c"]]
        expect = [["1", "2", "3"], ["4", "5", "6"], ["a", "b", "c"]]
        sequence = self.seq(elements)

        sequence.to_csv(tmp_path, compression="gzip")
        result = self.seq.csv(tmp_path).to_list()
        self.assertEqual(expect, result)

        sequence.to_csv(tmp_path, compression="lzma")
        result = self.seq.csv(tmp_path).to_list()
        self.assertEqual(expect, result)

        sequence.to_csv(tmp_path, compression="bz2")
        result = self.seq.csv(tmp_path).to_list()
        self.assertEqual(expect, result)

    def test_to_sqlite3_failure(self):
        insert_sql = "INSERT INTO user (id, name) VALUES (?, ?)"
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]
        with self.assertRaises(ValueError):
            self.seq(elements).to_sqlite3(1, insert_sql)

    def test_to_sqlite3_file(self):
        tmp_path = "functional/test/data/tmp/test.db"

        with sqlite3.connect(tmp_path) as conn:
            conn.execute("DROP TABLE IF EXISTS user;")
            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

        insert_sql = "INSERT INTO user (id, name) VALUES (?, ?)"
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]

        self.seq(elements).to_sqlite3(tmp_path, insert_sql)
        result = self.seq.sqlite3(tmp_path, "SELECT id, name FROM user;").to_list()
        self.assertListEqual(elements, result)

    def test_to_sqlite3_query(self):
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]

        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            insert_sql = "INSERT INTO user (id, name) VALUES (?, ?)"
            self.seq(elements).to_sqlite3(conn, insert_sql)
            result = self.seq.sqlite3(conn, "SELECT id, name FROM user;").to_list()
            self.assertListEqual(elements, result)

    def test_to_sqlite3_tuple(self):
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]

        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            table_name = "user"
            self.seq(elements).to_sqlite3(conn, table_name)
            result = self.seq.sqlite3(conn, "SELECT id, name FROM user;").to_list()
            self.assertListEqual(elements, result)

    def test_to_sqlite3_namedtuple(self):
        if self.seq is pseq:
            raise self.skipTest("pseq can't serialize all functions")
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]

        # test namedtuple with the same order as column
        with sqlite3.connect(":memory:") as conn:
            user = collections.namedtuple("user", ["id", "name"])

            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            table_name = "user"
            self.seq(elements).map(lambda u: user(u[0], u[1])).to_sqlite3(
                conn, table_name
            )
            result = self.seq.sqlite3(conn, "SELECT id, name FROM user;").to_list()
            self.assertListEqual(elements, result)

        # test namedtuple with different order
        with sqlite3.connect(":memory:") as conn:
            user = collections.namedtuple("user", ["name", "id"])

            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            table_name = "user"
            self.seq(elements).map(lambda u: user(u[1], u[0])).to_sqlite3(
                conn, table_name
            )
            result = self.seq.sqlite3(conn, "SELECT id, name FROM user;").to_list()
            self.assertListEqual(elements, result)

    def test_to_sqlite3_dict(self):
        elements = [(1, "Tom"), (2, "Jack"), (3, "Jane"), (4, "Stephan")]

        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            table_name = "user"
            self.seq(elements).map(lambda x: {"id": x[0], "name": x[1]}).to_sqlite3(
                conn, table_name
            )
            result = self.seq.sqlite3(conn, "SELECT id, name FROM user;").to_list()
            self.assertListEqual(elements, result)

    def test_to_sqlite3_typerror(self):
        elements = [1, 2, 3]
        with sqlite3.connect(":memory:") as conn:
            conn.execute("CREATE TABLE user (id INT, name TEXT);")
            conn.commit()

            table_name = "user"
            with self.assertRaises(TypeError):
                self.seq(elements).to_sqlite3(conn, table_name)

    def test_to_pandas(self):
        # pylint: disable=superfluous-parens
        try:
            import pandas as pd

            elements = [(1, "a"), (2, "b"), (3, "c")]
            df_expect = pd.DataFrame.from_records(elements)
            df_seq = self.seq(elements).to_pandas()
            self.assertTrue(df_seq.equals(df_expect))

            df_expect = pd.DataFrame.from_records(elements, columns=["id", "name"])
            df_seq = self.seq(elements).to_pandas(columns=["id", "name"])
            self.assertTrue(df_seq.equals(df_expect))

            elements = [
                dict(id=1, name="a"),
                dict(id=2, name="b"),
                dict(id=3, name="c"),
            ]
            df_expect = pd.DataFrame.from_records(elements)
            df_seq = self.seq(elements).to_pandas()
            self.assertTrue(df_seq.equals(df_expect))
        except ImportError:
            print("pandas not installed, skipping unit test")


# Skipping tests on pypy because of https://github.com/uqfoundation/dill/issues/73
@unittest.skipIf("__pypy__" in sys.builtin_module_names, "Skip parallel tests on pypy")
class TestParallelStreams(TestStreams):
    def setUp(self):
        self.seq = pseq
        self.seq_c_disabled = ParallelStream(disable_compression=True)
