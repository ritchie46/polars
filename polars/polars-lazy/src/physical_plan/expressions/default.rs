use crate::logical_plan::Context;
use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::{fmt_groupby_column, GroupByMethod, GroupTuples};
use polars_core::prelude::*;
use polars_core::utils::slice_offsets;
use rayon::prelude::*;
use std::borrow::Cow;
use std::sync::Arc;

pub struct IsNotNullExpr {
    physical_expr: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl IsNotNullExpr {
    pub fn new(physical_expr: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self {
            physical_expr,
            expr,
        }
    }
}

impl PhysicalExpr for IsNotNullExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.physical_expr.evaluate(df, state)?;
        Ok(series.is_not_null().into_series())
    }
    fn to_field(&self, _input_schema: &Schema) -> Result<Field> {
        Ok(Field::new("is_not_null", DataType::Boolean))
    }
}

pub(crate) struct AggregationExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: GroupByMethod,
}

impl AggregationExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self { expr, agg_type }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), self.agg_type);
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct AggQuantileExpr {
    pub(crate) expr: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: f64,
}

impl AggQuantileExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, quantile: f64) -> Self {
        Self { expr, quantile }
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> Result<Series> {
        unimplemented!()
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field = self.expr.to_field(input_schema)?;
        let new_name = fmt_groupby_column(field.name(), GroupByMethod::Quantile(self.quantile));
        Ok(Field::new(&new_name, field.data_type().clone()))
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct CastExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) data_type: DataType,
}

impl CastExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, data_type: DataType) -> Self {
        Self { input, data_type }
    }
}

impl PhysicalExpr for CastExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        series.cast_with_datatype(&self.data_type)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct TernaryExpr {
    pub predicate: Arc<dyn PhysicalExpr>,
    pub truthy: Arc<dyn PhysicalExpr>,
    pub falsy: Arc<dyn PhysicalExpr>,
    pub expr: Expr,
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mask_series = self.predicate.evaluate(df, state)?;
        let mask = mask_series.bool()?;
        let truthy = self.truthy.evaluate(df, state)?;
        let falsy = self.falsy.evaluate(df, state)?;
        truthy.zip_with(&mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.truthy.to_field(input_schema)
    }
}

pub struct ApplyExpr {
    pub input: Arc<dyn PhysicalExpr>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub output_type: Option<DataType>,
    pub expr: Expr,
}

impl ApplyExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        function: NoEq<Arc<dyn SeriesUdf>>,
        output_type: Option<DataType>,
        expr: Expr,
    ) -> Self {
        ApplyExpr {
            input,
            function,
            output_type,
            expr,
        }
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let input = self.input.evaluate(df, state)?;
        let in_name = input.name().to_string();
        let mut out = self.function.call_udf(input)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        match &self.output_type {
            Some(output_type) => {
                let input_field = self.input.to_field(input_schema)?;
                Ok(Field::new(input_field.name(), output_type.clone()))
            }
            None => self.input.to_field(input_schema),
        }
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: i64,
    pub(crate) len: usize,
}

impl PhysicalExpr for SliceExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        Ok(series.slice(self.offset, self.len))
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let s = self.input.evaluate(df, state)?;

        let groups = groups
            .iter()
            .map(|(first, idx)| {
                let (offset, len) = slice_offsets(self.offset, self.len, s.len());
                (*first, idx[offset..offset + len].to_vec())
            })
            .collect();

        Ok((s, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub(crate) struct BinaryFunctionExpr {
    pub(crate) input_a: Arc<dyn PhysicalExpr>,
    pub(crate) input_b: Arc<dyn PhysicalExpr>,
    pub(crate) function: NoEq<Arc<dyn SeriesBinaryUdf>>,
    pub(crate) output_field: NoEq<Arc<dyn BinaryUdfOutputField>>,
}

impl PhysicalExpr for BinaryFunctionExpr {
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series_a = self.input_a.evaluate(df, state)?;
        let series_b = self.input_b.evaluate(df, state)?;

        self.function.call_udf(series_a, series_b).map(|mut s| {
            s.rename("binary_function");
            s
        })
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        let field_a = self.input_a.to_field(input_schema)?;
        let field_b = self.input_b.to_field(input_schema)?;
        self.output_field
            .get_field(input_schema, Context::Default, &field_a, &field_b)
            .ok_or_else(|| PolarsError::UnknownSchema("no field found".into()))
    }
    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

pub struct FilterExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl FilterExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, by, expr }
    }
}

impl PhysicalExpr for FilterExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let series = self.input.evaluate(df, state)?;
        let predicate = self.by.evaluate(df, state)?;
        series.filter(predicate.bool()?)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<(Series, Cow<'a, GroupTuples>)> {
        let s = self.input.evaluate(df, state)?;
        let predicate_s = self.by.evaluate(df, state)?;
        let predicate = predicate_s.bool()?;

        let groups = groups
            .par_iter()
            .map(|(first, idx)| {
                let idx: Vec<u32> = idx
                    .iter()
                    .filter_map(|i| match predicate.get(*i as usize) {
                        Some(true) => Some(*i),
                        _ => None,
                    })
                    .collect();

                (*idx.get(0).unwrap_or(first), idx)
            })
            .collect();

        Ok((s, Cow::Owned(groups)))
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }
}
