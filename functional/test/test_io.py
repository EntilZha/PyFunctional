import unittest

from functional.io import ReusableFile, GZFile, BZ2File, XZFile, universal_write_open


class TestUtil(unittest.TestCase):
    def test_reusable_file(self):
        license_file_lf = ReusableFile("LICENSE.txt")
        with open("LICENSE.txt", encoding="utf8") as license_file:
            assert list(license_file) == list(license_file_lf)
        assert list(iter(license_file_lf)) == list(iter(license_file_lf))

    def test_gzip_file(self):
        file_name = "functional/test/data/test.txt.gz"
        assert list(GZFile(file_name, mode="rt", encoding="utf-8")) == [
            "line0\n",
            "line1\n",
            "line2",
        ]
        assert list(GZFile(file_name, mode="rb")) == [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]

    def test_bz2_file(self):
        file_name = "functional/test/data/test.txt.bz2"
        assert list(BZ2File(file_name, mode="rt", encoding="utf-8")) == [
            "line0\n",
            "line1\n",
            "line2",
        ]
        assert list(BZ2File(file_name, mode="rb")) == [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]

    def test_xz_file(self):
        file_name = "functional/test/data/test.txt.xz"
        assert list(XZFile(file_name, mode="rt", encoding="utf-8")) == [
            "line0\n",
            "line1\n",
            "line2",
        ]
        assert list(XZFile(file_name, mode="rb")) == [
            b"line0\n",
            b"line1\n",
            b"line2",
        ]

    def test_universal_write_open(self):
        with self.assertRaises(ValueError):
            universal_write_open("", "", compression=1)
