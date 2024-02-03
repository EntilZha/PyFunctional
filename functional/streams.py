from __future__ import annotations

import builtins
import csv as csvapi
import json as jsonapi
import re
import sqlite3 as sqlite3api
from itertools import chain
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Iterable,
    Mapping,
    Optional,
    TypeVar,
    overload,
    IO,
    AnyStr,
)
import typing

from functional.execution import ExecutionEngine, ParallelExecutionEngine
from functional.io import FileDescriptorOrPath, get_read_function
from functional.pipeline import Sequence
from functional.util import coalesce, is_primitive

T = TypeVar("T")

if TYPE_CHECKING:
    import pandas as pd


class Stream:
    """
    Represents and implements a stream which separates the responsibilities of Sequence and
    ExecutionEngine.

    An instance of Stream is normally accessed as `seq`
    """

    def __init__(
        self,
        disable_compression: bool = False,
        max_repr_items: int = 100,
        no_wrap: Optional[bool] = None,
    ):
        """
        Default stream constructor.
        :param disable_compression: Disable file compression detection
        :param no_wrap: default value of no_wrap for functions like first() or last()
        """
        self.disable_compression = disable_compression
        self.max_repr_items = max_repr_items
        self.no_wrap = no_wrap

    @overload
    def __call__(
        self, arg: pd.DataFrame, no_wrap: Optional[bool] = None
    ) -> Sequence[list[Any]]:
        ...

    @overload
    def __call__(self, arg: Iterable[T], no_wrap: Optional[bool] = None) -> Sequence[T]:
        ...

    @overload
    def __call__(self, *args: T, no_wrap: Optional[bool] = None) -> Sequence[T]:
        ...

    def __call__(self, *args, no_wrap: Optional[bool] = None):  # type: ignore
        """
        Create a Sequence using a sequential ExecutionEngine.

        If args has more than one argument then the argument list becomes the sequence.

        If args[0] is primitive, a sequence wrapping it is created.

        If args[0] is a list, tuple, iterable, or Sequence it is wrapped as a Sequence.

        :param args: Sequence to wrap
        :return: Wrapped sequence
        """
        engine = ExecutionEngine()
        return self._parse_args(
            args, engine, no_wrap=coalesce(no_wrap, self.no_wrap, False)
        )

    def chain(self, *args: Iterable[T], no_wrap: Optional[bool] = None) -> Sequence[T]:
        """
        Create a Sequence chaining multiple iterators.
        """
        for arg in args:
            if not isinstance(arg, Iterable):
                raise TypeError("The type of arg should be iterator.")

        return self(chain(*args), no_wrap=no_wrap)

    def _parse_args(
        self, args, engine: Optional[ExecutionEngine], no_wrap: Optional[bool] = None
    ) -> Sequence[T]:
        _no_wrap = coalesce(no_wrap, self.no_wrap, False)
        if len(args) == 0:
            return Sequence(
                [], engine=engine, max_repr_items=self.max_repr_items, no_wrap=_no_wrap
            )
        if len(args) == 1:
            try:
                if type(args[0]).__name__ == "DataFrame":
                    import pandas

                    if isinstance(args[0], pandas.DataFrame):
                        return Sequence(
                            args[0].values,
                            engine=engine,
                            max_repr_items=self.max_repr_items,
                            no_wrap=_no_wrap,
                        )
            except ImportError:  # pragma: no cover
                pass

        if len(args) > 1:
            return Sequence(
                list(args),
                engine=engine,
                max_repr_items=self.max_repr_items,
                no_wrap=_no_wrap,
            )
        elif is_primitive(args[0]):
            return Sequence(
                [args[0]],
                engine=engine,
                max_repr_items=self.max_repr_items,
                no_wrap=_no_wrap,
            )
        else:
            return Sequence(
                args[0],
                engine=engine,
                max_repr_items=self.max_repr_items,
                no_wrap=_no_wrap,
            )

    def open(
        self,
        path: FileDescriptorOrPath,
        delimiter: Optional[str] = None,
        mode: str = "r",
        buffering: int = -1,
        encoding: Optional[str] = None,
        errors: Optional[str] = None,
        newline: Optional[str] = None,
    ):
        """
        Reads and parses input files as defined.

        If delimiter is not None, then the file is read in bulk then split on it. If it is None
        (the default), then the file is parsed as sequence of lines. The rest of the options are
        passed directly to builtins.open with the exception that write/append file modes is not
        allowed.

        >>> seq.open('examples/gear_list.txt').take(1)
        [u'tent\\n']

        :param path: path to file
        :param delimiter: delimiter to split joined text on. if None, defaults to per line split
        :param mode: file open mode
        :param buffering: passed to builtins.open
        :param encoding: passed to builtins.open
        :param errors: passed to builtins.open
        :param newline: passed to builtins.open
        :return: output of file depending on options wrapped in a Sequence via seq
        """
        if not re.match("^[rbt]{1,3}$", mode):
            raise ValueError("mode argument must be only have r, b, and t")

        file_open = get_read_function(path, self.disable_compression)
        file = file_open(
            path,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )
        if delimiter is None:
            return self(file)
        else:
            return self("".join(list(file)).split(delimiter))

    def range(self, *args: int) -> Sequence[int]:
        """
        Alias to range function where seq.range(args) is equivalent to seq(range(args)).

        >>> seq.range(1, 8, 2)
        [1, 3, 5, 7]

        :param args: args to range function
        :return: range(args) wrapped by a sequence
        """
        return self(builtins.range(*args))  # pylint: disable=no-member

    def csv(
        self,
        csv_file: FileDescriptorOrPath | Iterable[str],
        dialect: str = "excel",
        **fmt_params,
    ):
        """
        Reads and parses the input of a csv stream or file.

        csv_file can be a filepath or an object that implements the iterator interface
        (defines next() or __next__() depending on python version).

        >>> seq.csv('examples/camping_purchases.csv').take(2)
        [['1', 'tent', '300'], ['2', 'food', '100']]

        :param csv_file: path to file or iterator object
        :param dialect: dialect of csv, passed to csv.reader
        :param fmt_params: options passed to csv.reader
        :return: Sequence wrapping csv file
        """
        if isinstance(csv_file, FileDescriptorOrPath):
            file_open = get_read_function(csv_file, self.disable_compression)
            input_file = file_open(csv_file)
        elif hasattr(csv_file, "next") or hasattr(csv_file, "__next__"):
            input_file = csv_file
        else:
            raise TypeError(
                "csv_file must be a file descriptor or implement the iterator interface"
            )

        csv_input = csvapi.reader(input_file, dialect=dialect, **fmt_params)
        return self(csv_input).cache(delete_lineage=True)

    def csv_dict_reader(
        self,
        csv_file: FileDescriptorOrPath | Iterable[str],
        fieldnames=None,
        restkey=None,
        restval=None,
        dialect="excel",
        **kwds,
    ):
        if isinstance(csv_file, FileDescriptorOrPath):
            file_open = get_read_function(csv_file, self.disable_compression)
            input_file = file_open(csv_file)
        elif hasattr(csv_file, "next") or hasattr(csv_file, "__next__"):
            input_file = csv_file
        else:
            raise TypeError(
                "csv_file must be a file path or implement the iterator interface"
            )

        csv_input = csvapi.DictReader(
            input_file,
            fieldnames=fieldnames,
            restkey=restkey,
            restval=restval,
            dialect=dialect,
            **kwds,
        )
        return self(csv_input).cache(delete_lineage=True)

    def jsonl(self, jsonl_file: FileDescriptorOrPath | Iterable[str]):
        """
        Reads and parses the input of a jsonl file stream or file.

        Jsonl formatted files must have a single valid json value on each line which is parsed by
        the python json module.

        >>> seq.jsonl('examples/chat_logs.jsonl').first()
        {u'date': u'10/09', u'message': u'hello anyone there?', u'user': u'bob'}

        :param jsonl_file: path or file containing jsonl content
        :return: Sequence wrapping jsonl file
        """
        if isinstance(jsonl_file, FileDescriptorOrPath):
            file_open = get_read_function(jsonl_file, self.disable_compression)
            input_file = file_open(jsonl_file)
        else:
            input_file = jsonl_file
        return self(input_file).map(jsonapi.loads).cache(delete_lineage=True)

    def json(self, json_file: FileDescriptorOrPath | IO) -> Sequence[Any]:
        """
        Reads and parses the input of a json file handler or file.

        Json files are parsed differently depending on if the root is a dictionary or an array.

        1) If the json's root is a dictionary, these are parsed into a sequence of (Key, Value)
        pairs

        2) If the json's root is an array, these are parsed into a sequence
        of entries

        >>> seq.json('examples/users.json').first()
        [u'sarah', {u'date_created': u'08/08', u'news_email': True, u'email': u'sarah@gmail.com'}]

        :param json_file: path or file containing json content
        :return: Sequence wrapping jsonl file
        """
        if isinstance(json_file, FileDescriptorOrPath):
            file_open = get_read_function(json_file, self.disable_compression)
            input_file = file_open(json_file)
            json_input = jsonapi.load(input_file)
        elif hasattr(json_file, "read"):
            json_input = jsonapi.load(json_file)
        else:
            raise TypeError(
                "json_file must be a file path or implement the iterator interface"
            )

        if isinstance(json_input, list):
            return self(json_input)
        else:
            return self(json_input.items())

    # pylint: disable=keyword-arg-before-vararg
    def sqlite3(
        self,
        conn: sqlite3api.Connection | sqlite3api.Cursor | str,
        sql: str,
        parameters: typing.Sequence[Any] | Mapping[str, Any] | None = None,
        *args,
        **kwargs,
    ) -> Sequence[Any]:
        """
        Reads input by querying from a sqlite database.

        >>> seq.sqlite3('examples/users.db', 'select id, name from users where id = 1;').first()
        [(1, 'Tom')]

        :param conn: path or sqlite connection, cursor
        :param sql: SQL query string
        :param parameters: Parameters for sql query
        :return: Sequence wrapping SQL cursor
        """

        if parameters is None:
            parameters = ()

        if isinstance(conn, (sqlite3api.Connection, sqlite3api.Cursor)):
            return self(conn.execute(sql, parameters))
        elif isinstance(conn, str):
            with sqlite3api.connect(conn, *args, **kwargs) as input_conn:
                return self(input_conn.execute(sql, parameters))
        else:
            raise TypeError(
                "conn must be a file path or sqlite3 Connection/Cursor, got:",
                type(conn),
                conn,
            )


class ParallelStream(Stream):
    """
    Parallelized version of functional.streams.Stream normally accessible as `pseq`
    """

    def __init__(
        self,
        processes: Optional[int] = None,
        partition_size: Optional[int] = None,
        disable_compression: bool = False,
        no_wrap: bool | None = None,
    ):
        """
        Configure Stream for parallel processing and file compression detection
        :param processes: Number of parallel processes
        :param disable_compression: Disable file compression detection
        :param no_wrap: default value of no_wrap for functions like first() or last()
        """
        super().__init__(disable_compression=disable_compression, no_wrap=no_wrap)
        self.processes = processes
        self.partition_size = partition_size

    def __call__(
        self,
        *args,
        no_wrap: Optional[bool] = None,
        processes: Optional[int] = None,
        partition_size: Optional[int] = None,
    ):
        """
        Create a Sequence using a parallel ExecutionEngine.

        If args has more than one argument then the argument list becomes the sequence.

        If args[0] is primitive, a sequence wrapping it is created.

        If args[0] is a list, tuple, iterable, or Sequence it is wrapped as a Sequence.

        :param args: Sequence to wrap
        :return: Wrapped sequence
        """
        engine = ParallelExecutionEngine(
            processes=processes if processes is not None else self.processes,
            partition_size=(
                partition_size if partition_size is not None else self.partition_size
            ),
        )
        return self._parse_args(
            args, engine, no_wrap=coalesce(no_wrap, self.no_wrap, False)
        )


# pylint: disable=invalid-name
seq = Stream()
pseq = ParallelStream()
