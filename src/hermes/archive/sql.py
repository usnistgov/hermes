from dataclasses import dataclass
import numpy as np
import io
import sqlite3
from contextlib import closing
import multiprocessing
import copy

compressor = "zlib"  # zlib, bz2


def adapt_array(arr):  # add docstring
    return arr.tobytes()
    # """
    # http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    # """
    # # zlib uses similar disk size that Matlab v5 .mat files
    # # bz2 compress 4 times zlib, but storing process is 20 times slower.
    # out = io.BytesIO()
    # np.save(out, arr)
    # out.seek(0)
    # return sqlite3.Binary(out.read().encode(compressor))  # zlib, bz2


def convert_array(text):  # add docstring
    return np.frombuffer(text)
    # out = io.BytesIO(text)
    # out.seek(0)
    # out = io.BytesIO(out.read().decode(compressor))
    # return np.load(out)


sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)


@dataclass
class DataDeamon:
    """Base Class for Data Archives."""
