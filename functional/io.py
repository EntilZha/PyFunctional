import gzip
import lzma
import bz2
import io
import builtins

from typing import Optional, Generic, TypeVar, Any


WRITE_MODE = "wt"

_FileConv_co = TypeVar("_FileConv_co", covariant=True)


class ReusableFile(Generic[_FileConv_co]):
    """
    Class which emulates the builtin file except that calling iter() on it will return separate
    iterators on different file handlers (which are automatically closed when iteration stops). This
    is useful for allowing a file object to be iterated over multiple times while keep evaluation
    lazy.
    """

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        delimiter: Optional[str] = None,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        """
        Constructor arguments are passed directly to builtins.open
        :param path: passed to open
        :param delimiter: passed to open
        :param mode: passed to open
        :param buffering: passed to open
        :param encoding: passed to open
        :param errors: passed to open
        :param newline: passed to open
        :return: ReusableFile from the arguments
        """
        self.path = path
        self.delimiter = delimiter
        self.mode = mode
        self.buffering = buffering
        self.encoding = encoding
        self.errors = errors
        self.newline = newline

    def __iter__(self):
        """
        Returns a new iterator over the file using the arguments from the constructor. Each call
        to __iter__ returns a new iterator independent of all others
        :return: iterator over file
        """
        # pylint: disable=no-member
        with builtins.open(
            self.path,
            mode=self.mode,
            buffering=self.buffering,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            yield from file_content

    def read(self):
        # pylint: disable=no-member
        with builtins.open(
            self.path,
            mode=self.mode,
            buffering=self.buffering,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


class CompressedFile(ReusableFile):
    magic_bytes: Optional[bytes] = None

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        delimiter: Optional[str] = None,
        mode: str = "rt",
        buffering: int = -1,
        compresslevel: int = 9,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        super(CompressedFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        self.compresslevel = compresslevel

    @classmethod
    def is_compressed(cls, data):
        return data.startswith(cls.magic_bytes)


class GZFile(CompressedFile):
    magic_bytes: bytes = b"\x1f\x8b\x08"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        delimiter: Optional[str] = None,
        mode: str = "rt",
        buffering: int = -1,
        compresslevel: int = 9,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        super(GZFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def __iter__(self):
        if "t" in self.mode:
            with gzip.GzipFile(self.path, compresslevel=self.compresslevel) as gz_file:
                gz_file.read1 = gz_file.read
                with io.TextIOWrapper(
                    gz_file,
                    encoding=self.encoding,
                    errors=self.errors,
                    newline=self.newline,
                ) as file_content:
                    yield from file_content
        else:
            with gzip.open(
                self.path, mode=self.mode, compresslevel=self.compresslevel
            ) as file_content:
                yield from file_content

    def read(self):
        with gzip.GzipFile(self.path, compresslevel=self.compresslevel) as gz_file:
            gz_file.read1 = gz_file.read
            with io.TextIOWrapper(
                gz_file,
                encoding=self.encoding,
                errors=self.errors,
                newline=self.newline,
            ) as file_content:
                return file_content.read()


class BZ2File(CompressedFile):
    magic_bytes: bytes = b"\x42\x5a\x68"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path: str,
        delimiter: Optional[str] = None,
        mode: str = "rt",
        buffering: int = -1,
        compresslevel: int = 9,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        super(BZ2File, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    def __iter__(self):
        with bz2.open(
            self.path,
            mode=self.mode,
            compresslevel=self.compresslevel,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            yield from file_content

    def read(self):
        with bz2.open(
            self.path,
            mode=self.mode,
            compresslevel=self.compresslevel,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


class XZFile(CompressedFile):
    magic_bytes: bytes = b"\xfd\x37\x7a\x58\x5a\x00"

    # pylint: disable=too-many-instance-attributes
    def __init__(
        self,
        path,
        delimiter=None,
        mode="rt",
        buffering=-1,
        compresslevel=9,
        encoding=None,
        errors=None,
        newline=None,
        check=-1,
        preset=None,
        filters=None,
        format=None,
    ):
        super(XZFile, self).__init__(
            path,
            delimiter=delimiter,
            mode=mode,
            buffering=buffering,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        self.check = check
        self.preset = preset
        self.format = format
        self.filters = filters

    def __iter__(self):
        with lzma.open(
            self.path,
            mode=self.mode,
            format=self.format,
            check=self.check,
            preset=self.preset,
            filters=self.filters,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            yield from file_content

    def read(self):
        with lzma.open(
            self.path,
            mode=self.mode,
            format=self.format,
            check=self.check,
            preset=self.preset,
            filters=self.filters,
            encoding=self.encoding,
            errors=self.errors,
            newline=self.newline,
        ) as file_content:
            return file_content.read()


COMPRESSION_CLASSES = [GZFile, BZ2File, XZFile]
N_COMPRESSION_CHECK_BYTES = max(len(cls.magic_bytes) for cls in COMPRESSION_CLASSES)  # type: ignore


def get_read_function(filename: str, disable_compression: bool):
    if disable_compression:
        return ReusableFile
    with open(filename, "rb") as f:
        start_bytes = f.read(N_COMPRESSION_CHECK_BYTES)
        for cls in COMPRESSION_CLASSES:
            if cls.is_compressed(start_bytes):  # type: ignore
                return cls
        return ReusableFile


def universal_write_open(
    path: str,
    mode: str,
    buffering: int = -1,
    encoding: Optional[str] = None,
    errors: Optional[str] = None,
    newline: Optional[str] = None,
    compresslevel: int = 9,
    format: Optional[int] = None,
    check: int = -1,
    preset: Optional[int] = None,
    filters: Any = None,
    compression: Optional[str] = None,
):
    # pylint: disable=unexpected-keyword-arg,no-member
    if compression is None:
        return builtins.open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression in ("gz", "gzip"):
        return gzip.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            errors=errors,
            newline=newline,
            encoding=encoding,
        )
    elif compression in ("lzma", "xz"):
        return lzma.open(
            path,
            mode=mode,
            format=format,
            check=check,
            preset=preset,
            filters=filters,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    elif compression == "bz2":
        return bz2.open(
            path,
            mode=mode,
            compresslevel=compresslevel,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
    else:
        raise ValueError(
            f"compression must be None, gz, gzip, lzma, or xz and was {compression}"
        )
