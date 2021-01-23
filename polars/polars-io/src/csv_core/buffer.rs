use polars_core::prelude::*;

trait ToPolarsError {
    fn to_polars_err(&self) -> PolarsError {
        PolarsError::Other("Could not parse primitive type during csv parsing".into())
    }
}

impl ToPolarsError for lexical::Error {}
impl ToPolarsError for fast_float::Error {}

pub(crate) trait PrimitiveParser: ArrowPrimitiveType {
    fn parse(bytes: &[u8]) -> Result<Self::Native>;
}

impl PrimitiveParser for Float32Type {
    fn parse(bytes: &[u8]) -> Result<f32> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Float64Type {
    fn parse(bytes: &[u8]) -> Result<f64> {
        let a = fast_float::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}

impl PrimitiveParser for UInt8Type {
    fn parse(bytes: &[u8]) -> Result<u8> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt16Type {
    fn parse(bytes: &[u8]) -> Result<u16> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt32Type {
    fn parse(bytes: &[u8]) -> Result<u32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for UInt64Type {
    fn parse(bytes: &[u8]) -> Result<u64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int8Type {
    fn parse(bytes: &[u8]) -> Result<i8> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int16Type {
    fn parse(bytes: &[u8]) -> Result<i16> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int32Type {
    fn parse(bytes: &[u8]) -> Result<i32> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}
impl PrimitiveParser for Int64Type {
    fn parse(bytes: &[u8]) -> Result<i64> {
        let a = lexical::parse(bytes).map_err(|e| e.to_polars_err())?;
        Ok(a)
    }
}

trait ParsedBuffer<T> {
    fn parse_bytes(&mut self, bytes: &[u8], ignore_errors: bool, start_pos: usize) -> Result<()>;
}

impl<T> ParsedBuffer<T> for Vec<Option<T::Native>>
where
    T: PolarsNumericType + PrimitiveParser,
{
    fn parse_bytes(&mut self, bytes: &[u8], ignore_errors: bool, _start_pos: usize) -> Result<()> {
        let result = T::parse(bytes);

        match (result, ignore_errors) {
            (Ok(value), _) => self.push(Some(value)),
            (Err(_), true) => self.push(None),
            (Err(err), _) => return Err(err),
        };
        Ok(())
    }
}

/// To prevent over-allocating string buffers and expensive reallocation that lead to high peak heap
/// memory we first store the utf8 locations and only create the utf8 arrays on the end.
#[derive(Debug)]
pub(crate) struct Utf8Field {
    start_pos: usize,
    len: u32,
}

impl Utf8Field {
    pub(crate) fn parse_str<'a>(&self, bytes: &'a [u8]) -> Result<&'a str> {
        match std::str::from_utf8(&bytes[self.start_pos..self.start_pos + self.len as usize]) {
            Ok(s) => Ok(s),
            Err(_) => Err(PolarsError::Other("utf8_error".into())),
        }
    }
}

impl ParsedBuffer<Utf8Type> for Vec<Utf8Field> {
    fn parse_bytes(&mut self, bytes: &[u8], _ignore_errors: bool, start_pos: usize) -> Result<()> {
        self.push(Utf8Field {
            start_pos,
            len: bytes.len() as u32,
        });

        Ok(())
    }
}

impl ParsedBuffer<BooleanType> for Vec<Option<bool>> {
    fn parse_bytes(&mut self, bytes: &[u8], ignore_errors: bool, start_pos: usize) -> Result<()> {
        if bytes.eq_ignore_ascii_case(b"false") {
            self.push(Some(false));
        } else if bytes.eq_ignore_ascii_case(b"true") {
            self.push(Some(true));
        } else if ignore_errors {
            self.push(None);
        } else {
            return Err(PolarsError::Other(
                format!(
                    "Error while parsing value {} at byte position {} as boolean",
                    start_pos,
                    String::from_utf8_lossy(bytes)
                )
                .into(),
            ));
        }
        Ok(())
    }
}

pub(crate) fn init_buffers(
    projection: &[usize],
    capacity: usize,
    schema: &SchemaRef,
) -> Result<Vec<Buffer>> {
    projection
        .iter()
        .map(|&i| field_to_builder(i, capacity, schema))
        .collect()
}

fn field_to_builder(i: usize, capacity: usize, schema: &SchemaRef) -> Result<Buffer> {
    let field = schema.field(i).unwrap();

    let builder = match field.data_type() {
        &DataType::Boolean => Buffer::Boolean(Vec::with_capacity(capacity)),
        &DataType::Int32 => Buffer::Int32(Vec::with_capacity(capacity)),
        &DataType::Int64 => Buffer::Int64(Vec::with_capacity(capacity)),
        &DataType::UInt32 => Buffer::UInt32(Vec::with_capacity(capacity)),
        &DataType::UInt64 => Buffer::UInt64(Vec::with_capacity(capacity)),
        &DataType::Float32 => Buffer::Float32(Vec::with_capacity(capacity)),
        &DataType::Float64 => Buffer::Float64(Vec::with_capacity(capacity)),
        &DataType::Utf8 => Buffer::Utf8(Vec::with_capacity(capacity), 0),
        other => {
            return Err(PolarsError::Other(
                format!("Unsupported data type {:?} when reading a csv", other).into(),
            ))
        }
    };
    Ok(builder)
}

#[derive(Debug)]
pub(crate) enum Buffer {
    Boolean(Vec<Option<bool>>),
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    UInt64(Vec<Option<u64>>),
    UInt32(Vec<Option<u32>>),
    Float32(Vec<Option<f32>>),
    Float64(Vec<Option<f64>>),
    /// Stores the Utf8 fields and the total string length seen for that column
    Utf8(Vec<Utf8Field>, usize),
}

impl Default for Buffer {
    fn default() -> Self {
        Buffer::Boolean(vec![])
    }
}

impl Buffer {
    pub(crate) fn add(
        &mut self,
        bytes: &[u8],
        ignore_errors: bool,
        start_pos: usize,
    ) -> Result<()> {
        use Buffer::*;
        match self {
            Boolean(buf) => <Vec<Option<bool>> as ParsedBuffer<BooleanType>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            Int32(buf) => <Vec<Option<i32>> as ParsedBuffer<Int32Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            Int64(buf) => <Vec<Option<i64>> as ParsedBuffer<Int64Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            UInt64(buf) => <Vec<Option<u64>> as ParsedBuffer<UInt64Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            UInt32(buf) => <Vec<Option<u32>> as ParsedBuffer<UInt32Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            Float32(buf) => <Vec<Option<f32>> as ParsedBuffer<Float32Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            Float64(buf) => <Vec<Option<f64>> as ParsedBuffer<Float64Type>>::parse_bytes(
                buf,
                bytes,
                ignore_errors,
                start_pos,
            ),
            Utf8(buf, len) => {
                *len += bytes.len();

                <Vec<Utf8Field> as ParsedBuffer<Utf8Type>>::parse_bytes(
                    buf,
                    bytes,
                    ignore_errors,
                    start_pos,
                )
            }
        }
    }
}

pub(crate) fn buffers_to_series<I>(buffers: I, bytes: &[u8], ignore_errors: bool) -> Result<Series>
where
    I: IntoIterator<Item = Buffer>,
{
    let buffers: Vec<_> = buffers.into_iter().collect();

    match &buffers[0] {
        Buffer::Boolean(_) => {
            let ca: BooleanChunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Boolean(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::Int32(_) => {
            let ca: Int32Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Int32(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::Int64(_) => {
            let ca: Int64Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Int64(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::UInt64(_) => {
            let ca: UInt64Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::UInt64(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::UInt32(_) => {
            let ca: UInt32Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::UInt32(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::Float32(_) => {
            let ca: Float32Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Float32(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::Float64(_) => {
            let ca: Float64Chunked = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Float64(buf) => Some(buf),
                    _ => None,
                })
                .flat_map(|v| v.into_iter())
                .collect();
            Ok(ca.into_series())
        }
        Buffer::Utf8(_, _) => {
            let buffers = buffers
                .into_iter()
                .filter_map(|buf| match buf {
                    Buffer::Utf8(buf, size) => Some((buf, size)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            let values_size = buffers.iter().map(|(_, size)| *size).sum::<usize>();
            let row_size = buffers.iter().map(|(v, _)| v.len()).sum::<usize>();
            let mut builder = Utf8ChunkedBuilder::new("", row_size, values_size);

            buffers
                .into_iter()
                .flat_map(|(v, _)| v.into_iter().map(|utf8_field| utf8_field.parse_str(bytes)))
                .try_for_each(|parse_result| {
                    match parse_result {
                        Ok(s) => {
                            builder.append_value(s);
                        }
                        Err(err) => {
                            if ignore_errors {
                                builder.append_null();
                            } else {
                                return Err(err);
                            }
                        }
                    }
                    Ok(())
                })?;
            Ok(builder.finish().into_series())
        }
    }
}