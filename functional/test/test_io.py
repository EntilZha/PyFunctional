import unittest

from functional.io import ReusableFile, GZFile, BZ2File, XZFile, universal_write_open


class TestUtil(unittest.TestCase):
    def test_reusable_file(self):
        license_file_lf = ReusableFile("LICENSE.txt")
        with open("LICENSE.txt", encoding="utf8") as license_file:
            self.assertEqual(list(license_file), list(license_file_lf))
        iter_1 = iter(license_file_lf)
        iter_2 = iter(license_file_lf)
        self.assertEqual(list(iter_1), list(iter_2))

    def test_gzip_file(self):
        file_name = "functional/test/data/test.txt.gz"
        expect = [
            "line0\n",
            "line1\n",
            "line2",
        ]
        self.assertListEqual(
            expect, list(GZFile(file_name, mode="rt", encoding="utf-8"))
        )

        expect = [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]
        self.assertListEqual(expect, list(GZFile(file_name, mode="rb")))

    def test_bz2_file(self):
        file_name = "functional/test/data/test.txt.bz2"
        expect = [
            "line0\n",
            "line1\n",
            "line2",
        ]
        self.assertListEqual(
            expect, list(BZ2File(file_name, mode="rt", encoding="utf-8"))
        )

        expect = [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]
        self.assertListEqual(expect, list(BZ2File(file_name, mode="rb")))

    def test_xz_file(self):
        file_name = "functional/test/data/test.txt.xz"
        expect = [
            "line0\n",
            "line1\n",
            "line2",
        ]
        self.assertListEqual(
            expect, list(XZFile(file_name, mode="rt", encoding="utf-8"))
        )

        expect = [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]
        self.assertListEqual(expect, list(XZFile(file_name, mode="rb")))

    def test_universal_write_open(self):
        with self.assertRaises(ValueError):
            universal_write_open("", "", compression=1)
