use std::convert::TryFrom;

use pyo3::types::{PyList, PyTuple};
use pyo3::{exceptions::PyRuntimeError, prelude::*};

use polars::frame::groupby::GroupBy;
use polars::prelude::*;
#[cfg(feature = "downsample")]
use polars_core::frame::groupby::resample::SampleRule;

use crate::apply::dataframe::{
    apply_lambda_unknown, apply_lambda_with_bool_out_type, apply_lambda_with_primitive_out_type,
    apply_lambda_with_utf8_out_type,
};
use crate::conversion::{ObjectValue, Wrap};
use crate::datatypes::PyDataType;
use crate::file::get_mmap_bytes_reader;
use crate::lazy::dataframe::PyLazyFrame;
use crate::utils::{downsample_str_to_rule, str_to_polarstype};
use crate::{
    arrow_interop,
    error::PyPolarsEr,
    file::{get_either_file, get_file_like, EitherRustPythonFile},
    series::{to_pyseries_collection, to_series_collection, PySeries},
};
use polars::frame::row::{rows_to_schema, Row};

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataFrame {
    pub df: DataFrame,
}

impl PyDataFrame {
    pub(crate) fn new(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

impl From<DataFrame> for PyDataFrame {
    fn from(df: DataFrame) -> Self {
        PyDataFrame { df }
    }
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PyDataFrame {
    #[new]
    pub fn __init__(columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = DataFrame::new(columns).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[allow(clippy::too_many_arguments)]
    pub fn read_csv(
        py_f: &PyAny,
        infer_schema_length: usize,
        chunk_size: usize,
        has_header: bool,
        ignore_errors: bool,
        stop_after_n_rows: Option<usize>,
        skip_rows: usize,
        projection: Option<Vec<usize>>,
        sep: &str,
        rechunk: bool,
        columns: Option<Vec<String>>,
        encoding: &str,
        n_threads: Option<usize>,
        path: Option<String>,
        overwrite_dtype: Option<Vec<(&str, &PyAny)>>,
        low_memory: bool,
        comment_char: Option<&str>,
        null_values: Option<Wrap<NullValues>>,
    ) -> PyResult<Self> {
        let null_values = null_values.map(|w| w.0);
        let comment_char = comment_char.map(|s| s.as_bytes()[0]);
        let encoding = match encoding {
            "utf8" => CsvEncoding::Utf8,
            "utf8-lossy" => CsvEncoding::LossyUtf8,
            e => {
                return Err(
                    PyPolarsEr::Other(format!("encoding not {} not implemented.", e)).into(),
                )
            }
        };

        let overwrite_dtype = overwrite_dtype.map(|overwrite_dtype| {
            let fields = overwrite_dtype
                .iter()
                .map(|(name, dtype)| {
                    let str_repr = dtype.str().unwrap().to_str().unwrap();
                    let dtype = str_to_polarstype(str_repr);
                    Field::new(name, dtype)
                })
                .collect();
            Schema::new(fields)
        });

        let mmap_bytes_r = get_mmap_bytes_reader(py_f)?;
        let df = CsvReader::new(mmap_bytes_r)
            .infer_schema(Some(infer_schema_length))
            .has_header(has_header)
            .with_stop_after_n_rows(stop_after_n_rows)
            .with_delimiter(sep.as_bytes()[0])
            .with_skip_rows(skip_rows)
            .with_ignore_parser_errors(ignore_errors)
            .with_projection(projection)
            .with_rechunk(rechunk)
            .with_chunk_size(chunk_size)
            .with_encoding(encoding)
            .with_columns(columns)
            .with_n_threads(n_threads)
            .with_path(path)
            .with_dtypes(overwrite_dtype.as_ref())
            .low_memory(low_memory)
            .with_comment_char(comment_char)
            .with_null_values(null_values)
            .finish()
            .map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    #[staticmethod]
    #[cfg(feature = "parquet")]
    pub fn read_parquet(py_f: PyObject, stop_after_n_rows: Option<usize>) -> PyResult<Self> {
        use EitherRustPythonFile::*;

        let result = match get_either_file(py_f, false)? {
            Py(f) => {
                let buf = f.as_slicable_buffer();
                ParquetReader::new(buf)
                    .with_stop_after_n_rows(stop_after_n_rows)
                    .finish()
            }
            Rust(f) => ParquetReader::new(f)
                .with_stop_after_n_rows(stop_after_n_rows)
                .finish(),
        };
        let df = result.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "ipc")]
    pub fn read_ipc(py_f: PyObject) -> PyResult<Self> {
        let file = get_file_like(py_f, false)?;
        let df = IpcReader::new(file).finish().map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[staticmethod]
    #[cfg(feature = "json")]
    pub fn read_json(json: &str) -> PyResult<Self> {
        // it is faster to first read to memory and then parse: https://github.com/serde-rs/json/issues/160
        // so don't bother with files.
        let df: DataFrame =
            serde_json::from_str(json).map_err(|e| PyPolarsEr::Other(format!("{:?}", e)))?;
        Ok(df.into())
    }

    #[cfg(feature = "json")]
    pub fn to_json(&self, py_f: PyObject, pretty: bool) -> PyResult<()> {
        let file = get_file_like(py_f, false)?;
        let r = match pretty {
            true => serde_json::to_writer_pretty(file, &self.df),
            false => serde_json::to_writer(file, &self.df),
        };
        r.map_err(|e| PyPolarsEr::Other(format!("{:?}", e)))?;
        Ok(())
    }

    #[staticmethod]
    pub fn from_arrow_record_batches(rb: Vec<&PyAny>) -> PyResult<Self> {
        let batches = arrow_interop::to_rust::to_rust_rb(&rb)?;
        let df = DataFrame::try_from(batches).map_err(PyPolarsEr::from)?;
        Ok(Self::from(df))
    }

    // somehow from_rows did not work
    #[staticmethod]
    pub fn read_rows(rows: Vec<Wrap<Row>>) -> PyResult<Self> {
        // safety:
        // wrap is transparent
        let rows: Vec<Row> = unsafe { std::mem::transmute(rows) };

        // replace inferred nulls with boolean
        let schema = rows_to_schema(&rows);
        let fields = schema
            .fields()
            .iter()
            .map(|fld| match fld.data_type() {
                DataType::Null => Field::new(fld.name(), DataType::Boolean),
                _ => fld.clone(),
            })
            .collect();
        let schema = Schema::new(fields);

        let df = DataFrame::from_rows_and_schema(&rows, &schema).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn to_csv(&self, py_f: PyObject, has_headers: bool, delimiter: u8) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        CsvWriter::new(&mut buf)
            .has_headers(has_headers)
            .with_delimiter(delimiter)
            .finish(&self.df)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    #[cfg(feature = "ipc")]
    pub fn to_ipc(&self, py_f: PyObject) -> PyResult<()> {
        let mut buf = get_file_like(py_f, true)?;
        IpcWriter::new(&mut buf)
            .finish(&self.df)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn row_tuple(&self, idx: i64) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let idx = if idx < 0 {
            (self.df.height() as i64 + idx) as usize
        } else {
            idx as usize
        };
        PyTuple::new(
            py,
            self.df.get_columns().iter().map(|s| match s.dtype() {
                DataType::Object(_) => {
                    let any = s.get_as_any(idx);
                    let obj: &ObjectValue = any.into();
                    obj.to_object(py)
                }
                _ => Wrap(s.get(idx)).into_py(py),
            }),
        )
        .into_py(py)
    }

    pub fn row_tuples(&self) -> PyObject {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let df = &self.df;
        PyList::new(
            py,
            (0..df.height()).map(|idx| {
                PyTuple::new(
                    py,
                    self.df.get_columns().iter().map(|s| match s.dtype() {
                        DataType::Object(_) => {
                            let any = s.get_as_any(idx);
                            let obj: &ObjectValue = any.into();
                            obj.to_object(py)
                        }
                        _ => Wrap(s.get(idx)).into_py(py),
                    }),
                )
            }),
        )
        .into_py(py)
    }

    #[cfg(feature = "parquet")]
    pub fn to_parquet(&self, path: &str) -> PyResult<()> {
        let f = std::fs::File::create(path).expect("to open a new file");
        ParquetWriter::new(f)
            .finish(&self.df)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn to_arrow(&self) -> PyResult<Vec<PyObject>> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let pyarrow = py.import("pyarrow")?;

        // Arrow does not know about our categorical type implementation, so we cast to utf8
        let cols = self
            .df
            .get_columns()
            .iter()
            .map(|s| match s.dtype() {
                DataType::Categorical => s.cast::<Utf8Type>().unwrap(),
                _ => s.clone(),
            })
            .collect::<Vec<_>>();
        let df = DataFrame::new_no_checks(cols);

        let rbs = df
            .iter_record_batches()
            .map(|rb| arrow_interop::to_py::to_py_rb(&rb, py, pyarrow))
            .collect::<PyResult<_>>()?;
        Ok(rbs)
    }

    pub fn add(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df + &s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn sub(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df - &s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn div(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df / &s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn mul(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df * &s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn rem(&self, s: &PySeries) -> PyResult<Self> {
        let df = (&self.df % &s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn sample_n(&self, n: usize, with_replacement: bool) -> PyResult<Self> {
        let df = self
            .df
            .sample_n(n, with_replacement)
            .map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn sample_frac(&self, frac: f64, with_replacement: bool) -> PyResult<Self> {
        let df = self
            .df
            .sample_frac(frac, with_replacement)
            .map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn rechunk(&mut self) -> Self {
        self.df.agg_chunks().into()
    }

    /// Format `DataFrame` as String
    pub fn as_str(&self) -> String {
        format!("{:?}", self.df)
    }

    pub fn fill_none(&self, strategy: &str) -> PyResult<Self> {
        let strat = match strategy {
            "backward" => FillNoneStrategy::Backward,
            "forward" => FillNoneStrategy::Forward,
            "min" => FillNoneStrategy::Min,
            "max" => FillNoneStrategy::Max,
            "mean" => FillNoneStrategy::Mean,
            "one" => FillNoneStrategy::One,
            "zero" => FillNoneStrategy::Zero,
            s => return Err(PyPolarsEr::Other(format!("Strategy {} not supported", s)).into()),
        };
        let df = self.df.fill_none(strat).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn join(
        &self,
        other: &PyDataFrame,
        left_on: Vec<&str>,
        right_on: Vec<&str>,
        how: &str,
    ) -> PyResult<Self> {
        let how = match how {
            "left" => JoinType::Left,
            "inner" => JoinType::Inner,
            "outer" => JoinType::Outer,
            "asof" => JoinType::AsOf,
            "cross" => JoinType::Cross,
            _ => panic!("not supported"),
        };

        let df = self
            .df
            .join(&other.df, left_on, right_on, how)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn get_columns(&self) -> Vec<PySeries> {
        let cols = self.df.get_columns().clone();
        to_pyseries_collection(cols)
    }

    /// Get column names
    pub fn columns(&self) -> Vec<&str> {
        self.df.get_column_names()
    }

    /// set column names
    pub fn set_column_names(&mut self, names: Vec<&str>) -> PyResult<()> {
        self.df.set_column_names(&names).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn with_column(&mut self, s: PySeries) -> PyResult<Self> {
        let mut df = self.df.clone();
        df.with_column(s.series).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    /// Get datatypes
    pub fn dtypes(&self) -> Vec<u8> {
        self.df
            .dtypes()
            .iter()
            .map(|arrow_dtype| {
                let dt: PyDataType = arrow_dtype.into();
                dt as u8
            })
            .collect()
    }

    pub fn n_chunks(&self) -> PyResult<usize> {
        let n = self.df.n_chunks().map_err(PyPolarsEr::from)?;
        Ok(n)
    }

    pub fn shape(&self) -> (usize, usize) {
        self.df.shape()
    }

    pub fn height(&self) -> usize {
        self.df.height()
    }

    pub fn width(&self) -> usize {
        self.df.width()
    }

    pub fn hstack_mut(&mut self, columns: Vec<PySeries>) -> PyResult<()> {
        let columns = to_series_collection(columns);
        self.df.hstack_mut(&columns).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn hstack(&self, columns: Vec<PySeries>) -> PyResult<Self> {
        let columns = to_series_collection(columns);
        let df = self.df.hstack(&columns).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn vstack_mut(&mut self, df: &PyDataFrame) -> PyResult<()> {
        self.df.vstack_mut(&df.df).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn vstack(&mut self, df: &PyDataFrame) -> PyResult<Self> {
        let df = self.df.vstack(&df.df).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn drop_in_place(&mut self, name: &str) -> PyResult<PySeries> {
        let s = self.df.drop_in_place(name).map_err(PyPolarsEr::from)?;
        Ok(PySeries { series: s })
    }

    pub fn drop_nulls(&self, subset: Option<Vec<String>>) -> PyResult<Self> {
        let df = self
            .df
            .drop_nulls(subset.as_ref().map(|s| s.as_ref()))
            .map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn drop(&self, name: &str) -> PyResult<Self> {
        let df = self.df.drop(name).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn select_at_idx(&self, idx: usize) -> Option<PySeries> {
        self.df.select_at_idx(idx).map(|s| PySeries::new(s.clone()))
    }

    pub fn find_idx_by_name(&self, name: &str) -> Option<usize> {
        self.df.find_idx_by_name(name)
    }

    pub fn column(&self, name: &str) -> PyResult<PySeries> {
        let series = self
            .df
            .column(name)
            .map(|s| PySeries::new(s.clone()))
            .map_err(PyPolarsEr::from)?;
        Ok(series)
    }

    pub fn select(&self, selection: Vec<&str>) -> PyResult<Self> {
        let df = self.df.select(&selection).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn filter(&self, mask: &PySeries) -> PyResult<Self> {
        let filter_series = &mask.series;
        if let Ok(ca) = filter_series.bool() {
            let df = self.df.filter(ca).map_err(PyPolarsEr::from)?;
            Ok(PyDataFrame::new(df))
        } else {
            Err(PyRuntimeError::new_err("Expected a boolean mask"))
        }
    }

    pub fn take(&self, indices: Vec<usize>) -> Self {
        let df = self.df.take_iter(indices.iter().copied());
        PyDataFrame::new(df)
    }

    pub fn take_with_series(&self, indices: &PySeries) -> PyResult<Self> {
        let idx = indices.series.u32().map_err(PyPolarsEr::from)?;
        let df = self.df.take(idx);
        Ok(PyDataFrame::new(df))
    }

    pub fn sort(&self, by_column: &str, reverse: bool) -> PyResult<Self> {
        let df = self.df.sort(by_column, reverse).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn sort_in_place(&mut self, by_column: &str, reverse: bool) -> PyResult<()> {
        self.df
            .sort_in_place(by_column, reverse)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn replace(&mut self, column: &str, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace(column, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn rename(&mut self, column: &str, new_col: &str) -> PyResult<()> {
        self.df.rename(column, new_col).map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn replace_at_idx(&mut self, index: usize, new_col: PySeries) -> PyResult<()> {
        self.df
            .replace_at_idx(index, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn insert_at_idx(&mut self, index: usize, new_col: PySeries) -> PyResult<()> {
        self.df
            .insert_at_idx(index, new_col.series)
            .map_err(PyPolarsEr::from)?;
        Ok(())
    }

    pub fn slice(&self, offset: usize, length: usize) -> Self {
        let df = self.df.slice(offset as i64, length);
        df.into()
    }

    pub fn head(&self, length: Option<usize>) -> Self {
        let df = self.df.head(length);
        PyDataFrame::new(df)
    }

    pub fn tail(&self, length: Option<usize>) -> Self {
        let df = self.df.tail(length);
        PyDataFrame::new(df)
    }

    pub fn is_unique(&self) -> PyResult<PySeries> {
        let mask = self.df.is_unique().map_err(PyPolarsEr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn is_duplicated(&self) -> PyResult<PySeries> {
        let mask = self.df.is_duplicated().map_err(PyPolarsEr::from)?;
        Ok(mask.into_series().into())
    }

    pub fn frame_equal(&self, other: &PyDataFrame, null_equal: bool) -> bool {
        if null_equal {
            self.df.frame_equal_missing(&other.df)
        } else {
            self.df.frame_equal(&other.df)
        }
    }

    #[cfg(feature = "downsample")]
    pub fn downsample_agg(
        &self,
        by: &str,
        rule: &str,
        n: u32,
        column_to_agg: Vec<(&str, Vec<&str>)>,
    ) -> PyResult<Self> {
        let rule = downsample_str_to_rule(rule, n)?;
        let gb = self.df.downsample(by, rule).map_err(PyPolarsEr::from)?;
        let df = gb.agg(&column_to_agg).map_err(PyPolarsEr::from)?;
        let out = df.sort(by, false).map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    #[cfg(feature = "downsample")]
    pub fn downsample(&self, by: &str, rule: &str, n: u32, agg: &str) -> PyResult<Self> {
        let rule = match rule {
            "second" => SampleRule::Second(n),
            "minute" => SampleRule::Minute(n),
            "day" => SampleRule::Day(n),
            "hour" => SampleRule::Hour(n),
            a => {
                return Err(PyPolarsEr::Other(format!("rule {} not supported", a)).into());
            }
        };
        let gb = self.df.downsample(by, rule).map_err(PyPolarsEr::from)?;
        let df = finish_groupby(gb, agg)?;
        let out = df.df.sort(by, false).map_err(PyPolarsEr::from)?;
        Ok(out.into())
    }

    pub fn groupby(&self, by: Vec<&str>, select: Option<Vec<String>>, agg: &str) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let selection = match select.as_ref() {
            Some(s) => gb.select(s),
            None => gb,
        };
        finish_groupby(selection, agg)
    }

    pub fn groupby_agg(
        &self,
        by: Vec<&str>,
        column_to_agg: Vec<(&str, Vec<&str>)>,
    ) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let df = gb.agg(&column_to_agg).map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn groupby_apply(&self, by: Vec<&str>, lambda: PyObject) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let function = move |df: DataFrame| {
            let gil = Python::acquire_gil();
            let py = gil.python();
            // get the pypolars module
            let pypolars = PyModule::import(py, "polars").unwrap();

            // create a PyDataFrame struct/object for Python
            let pydf = PyDataFrame::new(df);

            // Wrap this PySeries object in the python side DataFrame wrapper
            let python_df_wrapper = pypolars.call1("wrap_df", (pydf,)).unwrap();

            // call the lambda and get a python side DataFrame wrapper
            let result_df_wrapper = match lambda.call1(py, (python_df_wrapper,)) {
                Ok(pyobj) => pyobj,
                Err(e) => panic!("UDF failed: {}", e.pvalue(py).to_string()),
            };
            // unpack the wrapper in a PyDataFrame
            let py_pydf = result_df_wrapper.getattr(py, "_df").expect(
                "Could net get DataFrame attribute '_df'. Make sure that you return a DataFrame object.",
            );
            // Downcast to Rust
            let pydf = py_pydf.extract::<PyDataFrame>(py).unwrap();
            // Finally get the actual DataFrame
            Ok(pydf.df)
        };

        let gil = Python::acquire_gil();
        let py = gil.python();
        let df = py.allow_threads(|| gb.apply(function).map_err(PyPolarsEr::from))?;
        Ok(df.into())
    }

    pub fn groupby_quantile(
        &self,
        by: Vec<&str>,
        select: Vec<String>,
        quantile: f64,
    ) -> PyResult<Self> {
        let gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let selection = gb.select(&select);
        let df = selection.quantile(quantile);
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    #[cfg(feature = "pivot")]
    pub fn pivot(
        &self,
        by: Vec<String>,
        pivot_column: &str,
        values_column: &str,
        agg: &str,
    ) -> PyResult<Self> {
        let mut gb = self.df.groupby(&by).map_err(PyPolarsEr::from)?;
        let pivot = gb.pivot(pivot_column, values_column);
        let df = match agg {
            "first" => pivot.first(),
            "min" => pivot.min(),
            "max" => pivot.max(),
            "mean" => pivot.mean(),
            "median" => pivot.median(),
            "sum" => pivot.sum(),
            "count" => pivot.count(),
            a => Err(PolarsError::Other(
                format!("agg fn {} does not exists", a).into(),
            )),
        };
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn clone(&self) -> Self {
        PyDataFrame::new(self.df.clone())
    }

    pub fn explode(&self, columns: Vec<String>) -> PyResult<Self> {
        let df = self.df.explode(&columns);
        let df = df.map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn melt(&self, id_vars: Vec<&str>, value_vars: Vec<&str>) -> PyResult<Self> {
        let df = self
            .df
            .melt(id_vars, value_vars)
            .map_err(PyPolarsEr::from)?;
        Ok(PyDataFrame::new(df))
    }

    pub fn shift(&self, periods: i64) -> Self {
        self.df.shift(periods).into()
    }

    pub fn drop_duplicates(
        &self,
        maintain_order: bool,
        subset: Option<Vec<String>>,
    ) -> PyResult<Self> {
        let df = self
            .df
            .drop_duplicates(maintain_order, subset.as_ref().map(|v| v.as_ref()))
            .map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn lazy(&self) -> PyLazyFrame {
        self.df.clone().lazy().into()
    }

    pub fn max(&self) -> Self {
        self.df.max().into()
    }

    pub fn min(&self) -> Self {
        self.df.min().into()
    }

    pub fn sum(&self) -> Self {
        self.df.sum().into()
    }

    pub fn mean(&self) -> Self {
        self.df.mean().into()
    }

    pub fn std(&self) -> Self {
        self.df.std().into()
    }

    pub fn var(&self) -> Self {
        self.df.var().into()
    }

    pub fn median(&self) -> Self {
        self.df.median().into()
    }

    pub fn hmean(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hmean().map_err(PyPolarsEr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hmax(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hmax().map_err(PyPolarsEr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hmin(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hmin().map_err(PyPolarsEr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn hsum(&self) -> PyResult<Option<PySeries>> {
        let s = self.df.hsum().map_err(PyPolarsEr::from)?;
        Ok(s.map(|s| s.into()))
    }

    pub fn quantile(&self, quantile: f64) -> PyResult<Self> {
        let df = self.df.quantile(quantile).map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn to_dummies(&self) -> PyResult<Self> {
        let df = self.df.to_dummies().map_err(PyPolarsEr::from)?;
        Ok(df.into())
    }

    pub fn null_count(&self) -> Self {
        let df = self.df.null_count();
        df.into()
    }

    pub fn apply(&self, lambda: &PyAny, output_type: &PyAny) -> PyResult<PySeries> {
        let gil = Python::acquire_gil();
        let py = gil.python();
        let df = &self.df;

        let output_type = match output_type.is_none() {
            true => None,
            false => {
                let str_repr = output_type.str().unwrap().to_str().unwrap();
                Some(str_to_polarstype(str_repr))
            }
        };
        let out = match output_type {
            Some(DataType::Int32) => {
                apply_lambda_with_primitive_out_type::<Int32Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Int64) => {
                apply_lambda_with_primitive_out_type::<Int64Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::UInt32) => {
                apply_lambda_with_primitive_out_type::<UInt32Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::UInt64) => {
                apply_lambda_with_primitive_out_type::<UInt64Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Float32) => {
                apply_lambda_with_primitive_out_type::<Float32Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Float64) => {
                apply_lambda_with_primitive_out_type::<Float64Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Boolean) => {
                apply_lambda_with_bool_out_type(df, py, lambda, 0, None).into_series()
            }
            Some(DataType::Date32) => {
                apply_lambda_with_primitive_out_type::<Date32Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Date64) => {
                apply_lambda_with_primitive_out_type::<Date64Type>(df, py, lambda, 0, None)
                    .into_series()
            }
            Some(DataType::Utf8) => {
                apply_lambda_with_utf8_out_type(df, py, lambda, 0, None).into_series()
            }

            _ => apply_lambda_unknown(df, py, lambda)?,
        };

        Ok(out.into())
    }

    pub fn shrink_to_fit(&mut self) {
        self.df.shrink_to_fit();
    }

    pub fn hash_rows(&self, k0: u64, k1: u64, k2: u64, k3: u64) -> PyResult<PySeries> {
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        let hash = self.df.hash_rows(Some(hb)).map_err(PyPolarsEr::from)?;
        Ok(hash.into_series().into())
    }
}

fn finish_groupby(gb: GroupBy, agg: &str) -> PyResult<PyDataFrame> {
    let df = match agg {
        "min" => gb.min(),
        "max" => gb.max(),
        "mean" => gb.mean(),
        "first" => gb.first(),
        "last" => gb.last(),
        "sum" => gb.sum(),
        "count" => gb.count(),
        "n_unique" => gb.n_unique(),
        "median" => gb.median(),
        "agg_list" => gb.agg_list(),
        "groups" => gb.groups(),
        "std" => gb.std(),
        "var" => gb.var(),
        a => Err(PolarsError::Other(
            format!("agg fn {} does not exists", a).into(),
        )),
    };
    let df = df.map_err(PyPolarsEr::from)?;
    Ok(PyDataFrame::new(df))
}
