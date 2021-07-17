from builtins import range
from datetime import datetime
from io import BytesIO

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.datatypes import *
from polars.lazy import *


def test_version():
    pl.__version__


def test_init():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)

    # length mismatch
    with pytest.raises(RuntimeError):
        pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})

    df = pl.DataFrame(np.random.randn(3, 5))
    assert df.shape == (3, 5)
    assert df.columns == ["0", "1", "2", "3", "4"]


def test_selection():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    # column selection by string(s) in first dimension
    assert df["a"] == [1, 2, 3]
    assert df["b"] == [1.0, 2.0, 3.0]
    assert df["c"] == ["a", "b", "c"]

    # row selection by integers(s) in first dimension
    assert df[0].frame_equal(pl.DataFrame({"a": [1], "b": [1.0], "c": ["a"]}))
    assert df[-1].frame_equal(pl.DataFrame({"a": [3], "b": [3.0], "c": ["c"]}))

    # row, column selection when using two dimensions
    assert df[:, 0] == [1, 2, 3]
    assert df[:, 1] == [1.0, 2.0, 3.0]
    assert df[:2, 2] == ["a", "b"]

    assert df[[1, 2]].frame_equal(
        pl.DataFrame({"a": [2, 3], "b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[[-1, -2]].frame_equal(
        pl.DataFrame({"a": [3, 2], "b": [3.0, 2.0], "c": ["c", "b"]})
    )

    assert df[[True, False, True]].frame_equal(
        pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    )
    assert df[["a", "b"]].columns == ["a", "b"]
    assert df[[1, 2], [1, 2]].frame_equal(
        pl.DataFrame({"b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[1, 2] == "b"
    assert df[1, 1] == 2.0
    assert df[2, 0] == 3

    assert df[[True, False, True], "b"].shape == (2, 1)
    assert df[[True, False, False], ["a", "b"]].shape == (1, 2)

    assert df[[0, 1], "b"].shape == (2, 1)
    assert df[[2], ["a", "b"]].shape == (1, 2)
    assert df.select_at_idx(0).name == "a"
    assert (df.a == df["a"]).sum() == 3
    assert (df.c == df["a"]).sum() == 0
    assert df[:, "a":"b"].shape == (3, 2)
    assert df[:, "a":"c"].columns == ["a", "b", "c"]
    expect = pl.DataFrame({"c": ["b"]})
    assert df[1, [2]].frame_equal(expect)
    expect = pl.DataFrame({"b": [1.0, 3.0]})
    assert df[[0, 2], [1]].frame_equal(expect)
    assert df[0, "c"] == "a"
    assert df[1, "c"] == "b"
    assert df[2, "c"] == "c"
    assert df[0, "a"] == 1


def test_from_arrow():
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 2)),
        }
    )
    assert pl.from_arrow_table(tbl).shape == (2, 5)


def test_downsample():
    s = pl.Series(
        "datetime",
        [
            946684800000,
            946684860000,
            946684920000,
            946684980000,
            946685040000,
            946685100000,
            946685160000,
            946685220000,
            946685280000,
            946685340000,
            946685400000,
            946685460000,
            946685520000,
            946685580000,
            946685640000,
            946685700000,
            946685760000,
            946685820000,
            946685880000,
            946685940000,
        ],
    ).cast(Date64)
    s2 = s.clone()
    df = pl.DataFrame({"a": s, "b": s2})
    out = df.downsample("a", rule="minute", n=5).first()
    assert out.shape == (4, 2)

    # OLHC
    out = df.downsample("a", rule="minute", n=5).agg(
        {"b": ["first", "min", "max", "last"]}
    )
    assert out.shape == (4, 5)

    # test to_pandas as well.
    out = df.to_pandas()
    assert out["a"].dtype == "datetime64[ns]"


def test_sort():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    df.sort("a", in_place=True)
    assert df.frame_equal(pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}))


def test_replace():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = pl.Series("c", [True, False, True])
    df.replace("a", s)
    assert df.frame_equal(pl.DataFrame({"c": [True, False, True], "b": [1, 2, 3]}))


def test_assignment():
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [2, 3, 4]})
    df["foo"] = df["foo"]
    # make sure that assignment does not change column order
    assert df.columns == ["foo", "bar"]


def test_slice():
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df = df.slice(1, 2)
    assert df.frame_equal(pl.DataFrame({"a": [1, 3], "b": ["b", "c"]}))


def test_null_count():
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)


def test_head_tail():
    df = pl.DataFrame({"a": range(10), "b": range(10)})
    assert df.head(5).height == 5
    assert df.tail(5).height == 5

    assert not df.head(5).frame_equal(df.tail(5))
    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.tail(100).height == 10


def test_groupby():
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    # use __getitem__ to map to select
    assert (
        df.groupby("a")["b"]
        .sum()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [4, 11, 6]}))
    )

    assert (
        df.groupby("a")
        .select("b")
        .sum()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [4, 11, 6]}))
    )
    assert (
        df.groupby("a")
        .select("c")
        .sum()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [10, 10, 1]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .min()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [1, 2, 6]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .max()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [3, 5, 6]}))
    )
    assert (
        df.groupby("a")
        .select("b")
        .mean()
        .sort(by="a")
        .frame_equal(
            pl.DataFrame({"a": ["a", "b", "c"], "": [2.0, (2 + 4 + 5) / 3, 6.0]})
        )
    )
    assert (
        df.groupby("a")
        .select("b")
        .last()
        .sort(by="a")
        .frame_equal(pl.DataFrame({"a": ["a", "b", "c"], "": [3, 5, 6]}))
    )
    # check if it runs
    (df.groupby("a").select("b").n_unique())

    (df.groupby("a").select("b").quantile(0.3))
    (df.groupby("a").select("b").agg_list())

    gb_df = df.groupby("a").agg({"b": ["sum", "min"], "c": "count"})
    assert "b_sum" in gb_df.columns
    assert "b_min" in gb_df.columns

    #
    # # TODO: is false because count is u32
    # df.groupby(by="a", select="b", agg="count").frame_equal(
    #     pl.DataFrame({"a": ["a", "b", "c"], "": [2, 3, 1]})
    # )
    assert df.groupby("a").apply(lambda df: df[["c"]].sum()).sort("c")["c"][0] == 1

    assert (
        df.groupby("a").groups().sort("a")["a"].series_equal(pl.Series(["a", "b", "c"]))
    )

    for subdf in df.groupby("a"):
        if subdf["a"][0] == "b":
            assert subdf.shape == (3, 3)

    assert df.groupby("a").get_group("c").shape == (1, 3)
    assert df.groupby("a").get_group("b").shape == (3, 3)
    assert df.groupby("a").get_group("a").shape == (2, 3)

    # Use lazy API in eager groupby
    assert df.groupby("a").agg([pl.sum("b")]).shape == (3, 2)
    # test if it accepts a single expression
    assert df.groupby("a").agg(pl.sum("b")).shape == (3, 2)

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )

    # check if this query runs and thus column names propagate
    df.groupby("b").agg(col("c").forward_fill()).explode("c")


def test_join():
    df_left = pl.DataFrame(
        {
            "a": ["a", "b", "a", "z"],
            "b": [1, 2, 3, 4],
            "c": [6, 5, 4, 3],
        }
    )
    df_right = pl.DataFrame(
        {
            "a": ["b", "c", "b", "a"],
            "k": [0, 3, 9, 6],
            "c": [1, 0, 2, 1],
        }
    )

    joined = df_left.join(df_right, left_on="a", right_on="a").sort("a")
    assert joined["b"].series_equal(pl.Series("", [1, 3, 2, 2]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="left").sort("a")
    assert joined["c_right"].is_null().sum() == 1
    assert joined["b"].series_equal(pl.Series("", [1, 3, 2, 2, 4]))
    joined = df_left.join(df_right, left_on="a", right_on="a", how="outer").sort("a")
    assert joined["c_right"].null_count() == 1
    assert joined["c"].null_count() == 2
    assert joined["b"].null_count() == 2

    df_a = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    df_b = pl.DataFrame(
        {"foo": [1, 1, 1], "bar": ["a", "c", "c"], "ham": ["let", "var", "const"]}
    )

    # just check if join on multiple columns runs
    df_a.join(df_b, left_on=["a", "b"], right_on=["foo", "bar"])

    eager_join = df_a.join(df_b, left_on="a", right_on="foo")

    lazy_join = df_a.lazy().join(df_b.lazy(), left_on="a", right_on="foo").collect()
    assert lazy_join.shape == eager_join.shape


def test_hstack():
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df.hstack([pl.Series("stacked", [-1, -1, -1])], in_place=True)
    assert df.shape == (3, 3)
    assert df.columns == ["a", "b", "stacked"]


def test_drop():
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop("a")
    assert df.shape == (3, 2)
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    s = df.drop_in_place("a")
    assert s.name == "a"


def test_file_buffer():
    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = pl.DataFrame.read_csv(f, has_headers=False)
    assert df.shape == (2, 6)
    f.seek(0)

    # check if not fails on TryClone and Length impl in file.rs
    with pytest.raises(RuntimeError) as e:
        df.read_parquet(f)
    assert "Invalid Parquet file" in str(e.value)


def test_set():
    np.random.seed(1)
    df = pl.DataFrame(
        {"foo": np.random.rand(10), "bar": np.arange(10), "ham": ["h"] * 10}
    )
    df["new"] = np.random.rand(10)
    df[df["new"] > 0.5, "new"] = 1


def test_melt():
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    melted = df.melt(id_vars="A", value_vars=["B", "C"])
    assert melted["value"] == [1, 3, 4, 2, 4, 6]


def test_shift():
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    a = df.shift(1)
    b = pl.DataFrame({"A": [None, "a", "b"], "B": [None, 1, 3]}, nullable=True)
    assert a.frame_equal(b, null_equal=True)


def test_to_dummies():
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    dummies = df.to_dummies()
    assert dummies["A_a"].to_list() == [1, 0, 0]
    assert dummies["A_b"].to_list() == [0, 1, 0]
    assert dummies["A_c"].to_list() == [0, 0, 1]


def test_from_pandas():
    df = pd.DataFrame(
        {
            "bools": [False, True, False],
            "bools_nulls": [None, True, False],
            "int": [1, 2, 3],
            "int_nulls": [1, None, 3],
            "floats": [1.0, 2.0, 3.0],
            "floats_nulls": [1.0, None, 3.0],
            "strings": ["foo", "bar", "ham"],
            "strings_nulls": ["foo", None, "ham"],
            "strings-cat": ["foo", "bar", "ham"],
        }
    )
    df["strings-cat"] = df["strings-cat"].astype("category")

    out = pl.from_pandas(df)
    assert out.shape == (3, 9)


def test_custom_groupby():
    df = pl.DataFrame({"A": ["a", "a", "c", "c"], "B": [1, 3, 5, 2]})
    assert df.groupby("A").select("B").apply(lambda x: x.sum()).shape == (2, 2)
    assert df.groupby("A").select("B").apply(
        lambda x: pl.Series("", np.array(x))
    ).shape == (
        2,
        2,
    )

    df = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})

    out = (
        df.lazy()
        .groupby("b")
        .agg([col("a").apply(lambda x: x.sum(), return_dtype=int)])
        .collect()
    )
    assert out.shape == (3, 2)


def test_multiple_columns_drop():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    out = df.drop(["a", "b"])
    assert out.columns == ["c"]


def test_concat():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    assert pl.concat([df, df]).shape == (6, 3)

    # check if a remains unchanged
    a = pl.DataFrame.from_rows(((1, 2), (1, 2)))
    _ = pl.concat([a, a, a])
    assert a.shape == (2, 2)


def test_to_pandas(df):
    df.to_arrow()
    df.to_pandas()
    # test shifted df
    df.shift(2).to_pandas()
    df = pl.DataFrame({"col": pl.Series([True, False, True])})
    df.shift(2).to_pandas()


def test_from_arrow_table():
    data = {"a": [1, 2], "b": [1, 2]}
    tbl = pa.table(data)

    df = pl.from_arrow_table(tbl)
    df.frame_equal(pl.DataFrame(data))


def test_df_stats(df):
    df.var()
    df.std()
    df.min()
    df.max()
    df.sum()
    df.mean()
    df.median()
    df.quantile(0.4)


def test_from_pandas_datetime():
    df = pd.DataFrame({"datetime": ["2021-01-01", "2021-01-02"], "foo": [1, 2]})
    df["datetime"] = pd.to_datetime(df["datetime"])
    pl.from_pandas(df)


def test_df_fold():
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    assert df.fold(lambda s1, s2: s1 + s2).series_equal(pl.Series("a", [4.0, 5.0, 9.0]))
    assert df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2)).series_equal(
        pl.Series("a", [1.0, 1.0, 3.0])
    )

    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.fold(lambda s1, s2: s1 + s2)
    out.series_equal(pl.Series("", ["foo11", "bar22", "233"]))

    df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    # just check dispatch. values are tested on rust side.
    assert df.sum(axis=1).shape == (3, 1)
    assert df.mean(axis=1).shape == (3, 1)
    assert df.min(axis=1).shape == (3, 1)
    assert df.max(axis=1).shape == (3, 1)


def test_row_tuple():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)


def test_read_csv_categorical():
    f = BytesIO()
    f.write(b"col1,col2,col3,col4,col5,col6\n'foo',2,3,4,5,6\n'bar',8,9,10,11,12")
    f.seek(0)
    df = pl.DataFrame.read_csv(f, has_headers=True, dtype={"col1": pl.Categorical})
    assert df["col1"].dtype == pl.Categorical


def test_df_apply():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.apply(lambda x: len(x), None)
    assert out.sum() == 9


def test_column_names():
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5], pa.decimal128(38, 2)),
            "b": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )
    df = pl.from_arrow(tbl)
    assert df.columns == ["a", "b"]


def test_lazy_functions():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df[[pl.count("a")]]
    assert out["a"] == 3
    assert pl.count(df["a"]) == 3
    out = df[
        [
            pl.var("b"),
            pl.std("b"),
            pl.max("b"),
            pl.min("b"),
            pl.sum("b"),
            pl.mean("b"),
            pl.median("b"),
            pl.n_unique("b"),
            pl.first("b"),
            pl.last("b"),
        ]
    ]
    expected = 1.0
    assert np.isclose(out.select_at_idx(0), expected)
    assert np.isclose(pl.var(df["b"]), expected)
    expected = 1.0
    assert np.isclose(out.select_at_idx(1), expected)
    assert np.isclose(pl.std(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(2), expected)
    assert np.isclose(pl.max(df["b"]), expected)
    expected = 1
    assert np.isclose(out.select_at_idx(3), expected)
    assert np.isclose(pl.min(df["b"]), expected)
    expected = 6
    assert np.isclose(out.select_at_idx(4), expected)
    assert np.isclose(pl.sum(df["b"]), expected)
    expected = 2
    assert np.isclose(out.select_at_idx(5), expected)
    assert np.isclose(pl.mean(df["b"]), expected)
    expected = 2
    assert np.isclose(out.select_at_idx(6), expected)
    assert np.isclose(pl.median(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(7), expected)
    assert np.isclose(pl.n_unique(df["b"]), expected)
    expected = 1
    assert np.isclose(out.select_at_idx(8), expected)
    assert np.isclose(pl.first(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(9), expected)
    assert np.isclose(pl.last(df["b"]), expected)
    expected = 3
    assert np.isclose(out.select_at_idx(9), expected)
    assert np.isclose(pl.last(df["b"]), expected)


def test_multiple_column_sort():
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [2, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.sort([col("b"), col("c").reverse()])
    assert out["c"] == [2, 3, 1]
    assert out["b"] == [2, 2, 3]

    df = pl.DataFrame({"a": np.arange(1, 4), "b": ["a", "a", "b"]})

    df.sort("a", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 2, 1], "b": ["b", "a", "a"]})
    )

    df.sort("b", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 1, 2], "b": ["b", "a", "a"]})
    )

    df.sort(["b", "a"], reverse=[False, True]).frame_equal(
        pl.DataFrame({"a": [2, 1, 3], "b": ["a", "a", "b"]})
    )


def test_describe():
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, 6],
            "c": [True, False, True],
            "d": ["a", "b", "c"],
        }
    )
    assert df.describe().shape != df.shape
    assert set(df.describe().select_at_idx(2)) == set([1.0, 4.0, 5.0, 6.0])


def test_string_cache_eager_lazy():
    # tests if the global string cache is really global and not interfered by the lazy execution.
    # first the global settings was thread-local and this breaks with the parallel execution of lazy
    with pl.StringCache():
        df1 = pl.DataFrame(
            {"region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"]}
        ).select([pl.col("region_ids").cast(pl.Categorical)])
        df2 = pl.DataFrame(
            {"seq_name": ["reg4", "reg2", "reg1"], "score": [3.0, 1.0, 2.0]}
        ).select([pl.col("seq_name").cast(pl.Categorical), pl.col("score")])

    expected = pl.DataFrame(
        {
            "region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"],
            "score": [2.0, 1.0, None, 3.0, None],
        }
    )
    assert df1.join(
        df2, left_on="region_ids", right_on="seq_name", how="left"
    ).frame_equal(expected, null_equal=True)


def test_assign():
    # check if can assign in case of a single column
    df = pl.DataFrame({"a": [1, 2, 3]})
    # test if we can assign in case of single column
    df["a"] = df["a"] * 2
    assert df["a"] == [2, 4, 6]


def test_to_numpy():
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.to_numpy().shape == (3, 2)


def test_argsort_by(df):
    a = df[pl.argsort_by(["int_nulls", "floats"], reverse=[False, True])]["int_nulls"]
    assert a == [1, 0, 3]


def test_literal_series():
    df = pl.DataFrame(
        {
            "a": np.array([21.7, 21.8, 21], dtype=np.float32),
            "b": np.array([1, 3, 2], dtype=np.int64),
            "c": ["reg1", "reg2", "reg3"],
        }
    )
    out = (
        df.lazy()
        .with_column(pl.Series("e", [2, 1, 3]))
        .with_column(pl.col("e").cast(pl.Float32))
        .collect()
    )
    assert out["e"] == [2, 1, 3]


def test_to_html(df):
    # check if it does not panic/ error
    df._repr_html_()


def test_rows():
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    assert df.rows() == [(1, 1), (2, 2)]


def test_rename(df):
    out = df.rename({"strings": "bars", "int": "foos"})
    # check if wel can select these new columns
    _ = out[["foos", "bars"]]


def test_to_json(df):
    s = df.to_json(to_string=True)
    out = pl.read_json(s)
    assert df.frame_equal(out, null_equal=True)


def test_from_rows():
    df = pl.DataFrame.from_rows(
        [[1, 2, "foo"], [2, 3, "bar"]], column_name_mapping={1: "foo"}
    )
    assert df.frame_equal(
        pl.DataFrame({"column_0": [1, 2], "foo": [2, 3], "column_2": ["foo", "bar"]})
    )

    df = pl.DataFrame.from_rows(
        [[1, datetime.fromtimestamp(100)], [2, datetime.fromtimestamp(2398754908)]],
        column_name_mapping={1: "foo"},
    )
    assert df.dtypes == [pl.Int64, pl.Date64]


def test_repeat_by():
    df = pl.DataFrame({"name": ["foo", "bar"], "n": [2, 3]})

    out = df[col("n").repeat_by("n")]
    s = out["n"]
    assert s[0] == [2, 2]
    assert s[1] == [3, 3, 3]


def test_join_dates():
    date_times = pd.date_range(
        "2021-06-24 00:00:00", "2021-06-24 10:00:00", freq="1H", closed="left"
    )
    dts = (
        pl.from_pandas(date_times)
        .apply(lambda x: x + np.random.randint(1_000 * 60, 60_000 * 60))
        .cast(pl.Date64)
    )

    # some df with sensor id, (randomish) datetime and some value
    df = pl.DataFrame(
        {
            "sensor": ["a"] * 5 + ["b"] * 5,
            "datetime": dts,
            "value": [2, 3, 4, 1, 2, 3, 5, 1, 2, 3],
        }
    )
    df.join(df, on="datetime")


def test_asof_cross_join():
    left = pl.DataFrame({"a": [-10, 5, 10], "left_val": ["a", "b", "c"]})
    right = pl.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

    # only test dispatch of asof join
    out = left.join(right, on="a", how="asof")
    assert out.shape == (3, 4)

    left.lazy().join(right.lazy(), on="a", how="asof").collect()
    assert out.shape == (3, 4)

    # only test dispatch of cross join
    out = left.join(right, how="cross")
    assert out.shape == (15, 4)

    left.lazy().join(right.lazy(), how="cross").collect()
    assert out.shape == (15, 4)


def test_str_concat():
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4],
            "name": ["ham", "spam", "foo", None],
        }
    )
    out = df.with_column((pl.lit("Dr. ") + pl.col("name")).alias("graduated_name"))
    assert out["graduated_name"][0] == "Dr. ham"
    assert out["graduated_name"][1] == "Dr. spam"


def dot_product():
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})

    assert df["a"].dot(df["b"]) == 20
    assert df[[col("a").dot("b")]][0, "a"] == 20


def test_hash_rows():
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})
    assert df.hash_rows().dtype == pl.UInt64
    assert df["a"].hash().dtype == pl.UInt64
    assert df[[col("a").hash().alias("foo")]]["foo"].dtype == pl.UInt64
