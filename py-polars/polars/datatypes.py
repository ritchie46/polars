import ctypes
import typing as tp
from typing import Any, Dict, Type

from _ctypes import _SimpleCData

__pdoc__ = {
    "dtype_to_ctype": False,
    "dtype_to_int": False,
    "dtype_to_primitive": False,
    "pytype_to_polars_type": False,
}

__all__ = [
    "DataType",
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
    "Boolean",
    "Utf8",
    "List",
    "Date32",
    "Date64",
    "Time32Millisecond",
    "Time32Second",
    "Time64Nanosecond",
    "Time64Microsecond",
    "DurationNanosecond",
    "DurationMicrosecond",
    "DurationMillisecond",
    "DurationSecond",
    "TimestampNanosecond",
    "TimestampMicrosecond",
    "TimestampMillisecond",
    "TimestampSecond",
    "Object",
    "Categorical",
    "DTYPES",
    "DTYPE_TO_FFINAME",
    "dtype_to_primitive",
    "dtype_to_ctype",
    "pytype_to_polars_type",
]


class DataType:
    pass


class Int8(DataType):
    pass


class Int16(DataType):
    pass


class Int32(DataType):
    pass


class Int64(DataType):
    pass


class UInt8(DataType):
    pass


class UInt16(DataType):
    pass


class UInt32(DataType):
    pass


class UInt64(DataType):
    pass


class Float32(DataType):
    pass


class Float64(DataType):
    pass


class Boolean(DataType):
    pass


class Utf8(DataType):
    pass


class List(DataType):
    pass


class Date32(DataType):
    pass


class Date64(DataType):
    pass


class Time32Millisecond(DataType):
    pass


class Time32Second(DataType):
    pass


class Time64Nanosecond(DataType):
    pass


class Time64Microsecond(DataType):
    pass


class DurationNanosecond(DataType):
    pass


class DurationMicrosecond(DataType):
    pass


class DurationMillisecond(DataType):
    pass


class DurationSecond(DataType):
    pass


class TimestampNanosecond(DataType):
    pass


class TimestampMicrosecond(DataType):
    pass


class TimestampMillisecond(DataType):
    pass


class TimestampSecond(DataType):
    pass


class Object(DataType):
    pass


class Categorical(DataType):
    pass


# Don't change the order of these!
DTYPES: tp.List[Type[DataType]] = [
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Boolean,
    Utf8,
    List,
    Date32,
    Date64,
    Time64Nanosecond,
    DurationNanosecond,
    DurationMillisecond,
    Object,
    Categorical,
]
DTYPE_TO_FFINAME: Dict[Type[DataType], str] = {
    Int8: "i8",
    Int16: "i16",
    Int32: "i32",
    Int64: "i64",
    UInt8: "u8",
    UInt16: "u16",
    UInt32: "u32",
    UInt64: "u64",
    Float32: "f32",
    Float64: "f64",
    Boolean: "bool",
    Utf8: "str",
    List: "list",
    Date32: "date32",
    Date64: "date64",
    Time64Nanosecond: "time64_nanosecond",
    DurationNanosecond: "duration_nanosecond",
    DurationMillisecond: "duration_millisecond",
    Object: "object",
    Categorical: "categorical",
}


def dtype_to_primitive(dtype: Type[DataType]) -> Type[DataType]:
    #  TODO: add more
    if dtype == Date32:
        return Int32
    if dtype == Date64:
        return Int64
    ffi_name = DTYPE_TO_FFINAME[dtype]
    if "duration" in ffi_name:
        return Int64
    return dtype


def dtype_to_ctype(dtype: Type[DataType]) -> Type[_SimpleCData]:  # noqa: F821
    ptr_type: Type[_SimpleCData]
    if dtype == UInt8:
        ptr_type = ctypes.c_uint8
    elif dtype == UInt16:
        ptr_type = ctypes.c_uint16
    elif dtype == UInt32:
        ptr_type = ctypes.c_uint
    elif dtype == UInt64:
        ptr_type = ctypes.c_ulong
    elif dtype == Int8:
        ptr_type = ctypes.c_int8
    elif dtype == Int16:
        ptr_type = ctypes.c_int16
    elif dtype == Int32:
        ptr_type = ctypes.c_int
    elif dtype == Int64:
        ptr_type = ctypes.c_long
    elif dtype == Float32:
        ptr_type = ctypes.c_float
    elif dtype == Float64:
        ptr_type = ctypes.c_double
    elif dtype == Date32:
        ptr_type = ctypes.c_int
    elif dtype == Date64:
        ptr_type = ctypes.c_long
    else:
        raise NotImplementedError
    return ptr_type


def pytype_to_polars_type(data_type: Type[Any]) -> Type[DataType]:
    polars_type: Type[DataType]
    if data_type == int:
        polars_type = Int64
    elif data_type == str:
        polars_type = Utf8
    elif data_type == float:
        polars_type = Float64
    else:
        polars_type = data_type
    return polars_type
