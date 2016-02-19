# -*- coding: utf-8 -*-
from __future__ import absolute_import

import unittest
from collections import namedtuple
import gzip
import codecs

from functional.util import ReusableFile, is_namedtuple, GZFile

Data = namedtuple('Tuple', 'x y')


class TestUtil(unittest.TestCase):
    def test_reusable_file(self):
        license_file_lf = ReusableFile('LICENSE.txt')
        with open('LICENSE.txt') as license_file:
            self.assertEqual(list(license_file), list(license_file_lf))
        iter_1 = iter(license_file_lf)
        iter_2 = iter(license_file_lf)
        self.assertEqual(list(iter_1), list(iter_2))

    def test_is_namedtuple(self):
        self.assertTrue(is_namedtuple(Data(1, 2)))
        self.assertFalse(is_namedtuple((1, 2, 3)))
        self.assertFalse(is_namedtuple([1, 2, 3]))
        self.assertFalse(is_namedtuple(1))

    def test_gzip_text_modes(self):
        TEXT_LINES = [
            b'root:x:0:0:root:/root:/bin/bash\n',
            b'bin:x:1:1:bin:/bin:\n',
            b'daemon:x:2:2:daemon:/sbin:\n',
            b'adm:x:3:4:adm:/var/adm:\n',
            b'lp:x:4:7:lp:/var/spool/lpd:\n',
            b'sync:x:5:0:sync:/sbin:/bin/sync\n',
            b'shutdown:x:6:0:shutdown:/sbin:/sbin/shutdown\n',
            b'halt:x:7:0:halt:/sbin:/sbin/halt\n',
            u'我愛拍桑'.encode("utf-8")
            ]
        TEXT = b''.join(TEXT_LINES)

        text = TEXT.decode("utf-8")

        TEXT_LINES = [i.decode("utf-8") for i in TEXT_LINES]

        # text_native_eol = text.replace("\n", os.linesep)

        filename = '/tmp/test_text.gz'

        with gzip.open(filename, 'wb') as f:
            with codecs.getwriter("utf-8")(f) as fp:
                fp.write(text)
        self.assertListEqual(TEXT_LINES, list(GZFile(filename, mode='rt', encoding="utf-8")))

    def test_gzip_binary_modes(self):
        TEXT_LINES = [
                b'root:x:0:0:root:/root:/bin/bash\n',
                b'bin:x:1:1:bin:/bin:\n',
                b'daemon:x:2:2:daemon:/sbin:\n',
                b'adm:x:3:4:adm:/var/adm:\n',
                b'lp:x:4:7:lp:/var/spool/lpd:\n',
                b'sync:x:5:0:sync:/sbin:/bin/sync\n',
                b'shutdown:x:6:0:shutdown:/sbin:/sbin/shutdown\n',
                b'halt:x:7:0:halt:/sbin:/sbin/halt\n'
                ]
        text = b''.join(TEXT_LINES)
        # text_native_eol = text.replace("\n", os.linesep)

        filename = '/tmp/test_text.gz'

        with gzip.open(filename, 'wb') as f:
            f.write(text)

        self.assertListEqual(TEXT_LINES, list(GZFile(filename, mode='rb')))
