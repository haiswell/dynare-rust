// SPDX-License-Identifier: GPL-3.0-or-later
//! Minimal MATLAB Level-5 MAT-file reader/writer, enough for Dynare's mode
//! files (`<fname>_mode.mat`: `xparam1`, `hh`, `fval`, `parameter_names`) and
//! for handing results back to MATLAB.
//!
//! Reads: real numeric arrays of any storage type, char arrays, cell arrays,
//! compressed (`miCOMPRESSED`) or not; v7.3 (HDF5) files through `matfile_h5`. Writes: uncompressed double matrices
//! and cell columns of strings, which every MATLAB version loads.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, bail};
use flate2::read::ZlibDecoder;

use crate::linalg::Mat;

#[derive(Debug, Clone, PartialEq)]
pub enum MatValue {
    /// Real numeric array, column-major, with its dimensions.
    Numeric {
        dims: Vec<usize>,
        data: Vec<f64>,
    },
    Char(String),
    Cell {
        dims: Vec<usize>,
        items: Vec<MatValue>,
    },
    /// Struct array: `elems[k][f]` is field `fields[f]` of element `k` (column-major over `dims`).
    Struct {
        dims: Vec<usize>,
        fields: Vec<String>,
        elems: Vec<Vec<MatValue>>,
    },
    /// Anything else (objects, sparse, complex): kept by class only.
    Unsupported(u8),
}

impl MatValue {
    pub fn as_matrix(&self) -> Option<Mat> {
        match self {
            MatValue::Numeric { dims, data } if dims.len() == 2 => Some(Mat::from_column_slice(dims[0], dims[1], data)),
            _ => None,
        }
    }

    pub fn as_vec(&self) -> Option<Vec<f64>> {
        match self {
            MatValue::Numeric { data, .. } => Some(data.clone()),
            _ => None,
        }
    }

    pub fn as_scalar(&self) -> Option<f64> {
        match self {
            MatValue::Numeric { data, .. } if data.len() == 1 => Some(data[0]),
            _ => None,
        }
    }

    /// A column vector (`n x 1`) of doubles.
    pub fn column(v: &[f64]) -> MatValue {
        MatValue::Numeric { dims: vec![v.len(), 1], data: v.to_vec() }
    }

    /// A row vector (`1 x n`) of doubles.
    pub fn row(v: &[f64]) -> MatValue {
        MatValue::Numeric { dims: vec![1, v.len()], data: v.to_vec() }
    }

    pub fn scalar(x: f64) -> MatValue {
        MatValue::Numeric { dims: vec![1, 1], data: vec![x] }
    }

    pub fn matrix(m: &Mat) -> MatValue {
        MatValue::Numeric { dims: vec![m.nrows(), m.ncols()], data: m.as_slice().to_vec() }
    }

    /// A 1x1 struct from (field, value) pairs.
    pub fn structure(fields: Vec<(&str, MatValue)>) -> MatValue {
        let (names, vals): (Vec<String>, Vec<MatValue>) = fields.into_iter().map(|(k, v)| (k.to_string(), v)).unzip();
        MatValue::Struct { dims: vec![1, 1], fields: names, elems: vec![vals] }
    }

    /// Field `name` of element `k` of a struct array.
    pub fn field_of(&self, k: usize, name: &str) -> Option<&MatValue> {
        match self {
            MatValue::Struct { fields, elems, .. } => {
                let f = fields.iter().position(|x| x == name)?;
                elems.get(k).map(|e| &e[f])
            }
            _ => None,
        }
    }

    /// Field `name` of a 1x1 struct.
    pub fn field(&self, name: &str) -> Option<&MatValue> {
        self.field_of(0, name)
    }

    /// Number of elements of a struct array (0 for anything else).
    pub fn struct_len(&self) -> usize {
        match self {
            MatValue::Struct { elems, .. } => elems.len(),
            _ => 0,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MatValue::Char(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_strings(&self) -> Option<Vec<String>> {
        match self {
            MatValue::Cell { items, .. } => items
                .iter()
                .map(|i| match i {
                    MatValue::Char(s) => Some(s.clone()),
                    _ => None,
                })
                .collect(),
            MatValue::Char(s) => Some(vec![s.clone()]),
            _ => None,
        }
    }
}

const MI_INT8: u32 = 1;
const MI_UINT8: u32 = 2;
const MI_INT16: u32 = 3;
const MI_UINT16: u32 = 4;
const MI_INT32: u32 = 5;
const MI_UINT32: u32 = 6;
const MI_SINGLE: u32 = 7;
const MI_DOUBLE: u32 = 9;
const MI_INT64: u32 = 12;
const MI_UINT64: u32 = 13;
const MI_MATRIX: u32 = 14;
const MI_COMPRESSED: u32 = 15;
const MI_UTF8: u32 = 16;
const MI_UTF16: u32 = 17;

const MX_CELL: u8 = 1;
const MX_STRUCT: u8 = 2;
const MX_OBJECT: u8 = 3;
const MX_CHAR: u8 = 4;
const MX_OPAQUE: u8 = 17;
const MX_DOUBLE: u8 = 6;

/// Reads every variable of a MAT-file.
pub fn read(path: &Path) -> Result<BTreeMap<String, MatValue>> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    parse(&bytes).with_context(|| format!("parsing {}", path.display()))
}

pub fn parse(bytes: &[u8]) -> Result<BTreeMap<String, MatValue>> {
    if bytes.len() < 128 {
        bail!("not a MAT-file (too short)");
    }
    if bytes.get(512..520) == Some(b"\x89HDF\r\n\x1a\n") {
        // v7.3: HDF5
        return crate::matfile_h5::parse(bytes);
    }
    if &bytes[126..128] != b"IM" {
        bail!("only little-endian Level 5 MAT-files are supported (v7.3/HDF5 is not)");
    }
    let mut out = BTreeMap::new();
    let mut pos = 128;
    while pos + 8 <= bytes.len() {
        let (ty, data, next) = element(bytes, pos)?;
        pos = next;
        match ty {
            MI_MATRIX => {
                let (name, v) = matrix(data)?;
                out.insert(name, v);
            }
            MI_COMPRESSED => {
                let mut inflated = Vec::new();
                ZlibDecoder::new(data).read_to_end(&mut inflated).context("inflating compressed variable")?;
                let (ty, data, _) = element(&inflated, 0)?;
                if ty == MI_MATRIX {
                    let (name, v) = matrix(data)?;
                    out.insert(name, v);
                }
            }
            _ => {}
        }
    }
    Ok(out)
}

/// One data element at `pos`: (type, payload, position of the next element).
fn element(b: &[u8], pos: usize) -> Result<(u32, &[u8], usize)> {
    let w0 = u32::from_le_bytes(b[pos..pos + 4].try_into().unwrap());
    if w0 >> 16 != 0 {
        // small data element: type and size share the first word, data in the next 4 bytes
        let n = (w0 >> 16) as usize;
        if n > 4 || pos + 8 > b.len() {
            bail!("corrupt small data element at byte {pos}");
        }
        return Ok((w0 & 0xffff, &b[pos + 4..pos + 4 + n], pos + 8));
    }
    let n = u32::from_le_bytes(b[pos + 4..pos + 8].try_into().unwrap()) as usize;
    let start = pos + 8;
    if start + n > b.len() {
        bail!("data element at byte {pos} runs past the end of the file");
    }
    // compressed elements are not padded; everything else is padded to 8 bytes
    let next = if w0 == MI_COMPRESSED { start + n } else { start + n.div_ceil(8) * 8 };
    Ok((w0, &b[start..start + n], next.min(b.len())))
}

fn numbers(ty: u32, d: &[u8]) -> Result<Vec<f64>> {
    macro_rules! conv {
        ($t:ty) => {
            d.chunks_exact(std::mem::size_of::<$t>())
                .map(|c| <$t>::from_le_bytes(c.try_into().unwrap()) as f64)
                .collect()
        };
    }
    Ok(match ty {
        MI_DOUBLE => conv!(f64),
        MI_SINGLE => conv!(f32),
        MI_INT8 => conv!(i8),
        MI_UINT8 | MI_UTF8 => conv!(u8),
        MI_INT16 => conv!(i16),
        MI_UINT16 | MI_UTF16 => conv!(u16),
        MI_INT32 => conv!(i32),
        MI_UINT32 => conv!(u32),
        MI_INT64 => conv!(i64),
        MI_UINT64 => conv!(u64),
        _ => bail!("unsupported numeric storage type {ty}"),
    })
}

/// An miMATRIX payload: (name, value).
fn matrix(d: &[u8]) -> Result<(String, MatValue)> {
    if d.is_empty() {
        return Ok((String::new(), MatValue::Numeric { dims: vec![0, 0], data: vec![] }));
    }
    let (_, flags, p) = element(d, 0)?;
    let class = flags[0];
    let complex = flags[1] & 0x08 != 0;
    if class == MX_OPAQUE {
        return Ok((String::new(), MatValue::Unsupported(class))); // MCOS objects (RandStream, ...)
    }
    let (_, dims_raw, p) = element(d, p)?;
    let dims: Vec<usize> = numbers(MI_INT32, dims_raw)?.into_iter().map(|v| v as usize).collect();
    let (_, name_raw, mut p) = element(d, p)?;
    let name = String::from_utf8_lossy(name_raw).into_owned();
    let count: usize = dims.iter().product();
    let value = match class {
        MX_CELL => {
            let mut items = Vec::with_capacity(count);
            for _ in 0..count {
                let (_, sub, next) = element(d, p)?;
                p = next;
                items.push(matrix(sub)?.1);
            }
            MatValue::Cell { dims, items }
        }
        MX_STRUCT | MX_OBJECT => {
            if class == MX_OBJECT {
                let (_, _classname, next) = element(d, p)?;
                p = next;
            }
            let (_, len_raw, next) = element(d, p)?;
            p = next;
            let len = numbers(MI_INT32, len_raw)?.first().copied().unwrap_or(0.0) as usize;
            let (_, names_raw, next) = element(d, p)?;
            p = next;
            let fields: Vec<String> = if len == 0 {
                vec![]
            } else {
                names_raw.chunks(len).map(|c| String::from_utf8_lossy(c).trim_end_matches('\0').to_string()).collect()
            };
            let mut elems = Vec::with_capacity(count);
            for _ in 0..count {
                let mut e = Vec::with_capacity(fields.len());
                for _ in 0..fields.len() {
                    let (_, sub, next) = element(d, p)?;
                    p = next;
                    e.push(matrix(sub)?.1);
                }
                elems.push(e);
            }
            if class == MX_OBJECT { MatValue::Unsupported(class) } else { MatValue::Struct { dims, fields, elems } }
        }
        MX_CHAR => {
            let (ty, raw, _) = element(d, p)?;
            let s = if ty == MI_UTF8 {
                String::from_utf8_lossy(raw).into_owned()
            } else {
                let units: Vec<u16> = numbers(ty, raw)?.into_iter().map(|v| v as u16).collect();
                String::from_utf16_lossy(&units)
            };
            MatValue::Char(s)
        }
        6..=15 if !complex => {
            let (ty, raw, _) = element(d, p)?;
            let data = numbers(ty, raw)?;
            if data.len() != count {
                bail!("variable {name}: {} values for dimensions {dims:?}", data.len());
            }
            MatValue::Numeric { dims, data }
        }
        c => MatValue::Unsupported(c),
    };
    Ok((name, value))
}

// ---------------------------------------------------------------------------
// writing
// ---------------------------------------------------------------------------

/// A variable to write.
pub enum MatOut<'a> {
    Matrix(&'a Mat),
    Scalar(f64),
    /// Column vector.
    Vector(&'a [f64]),
    /// Cell column of strings (`parameter_names`).
    Strings(&'a [String]),
}

pub fn write(path: &Path, vars: &[(&str, MatOut)]) -> Result<()> {
    let mut b = header();
    for (name, v) in vars {
        let m = match v {
            MatOut::Matrix(m) => numeric_matrix(name, m.nrows(), m.ncols(), m.as_slice()),
            MatOut::Scalar(x) => numeric_matrix(name, 1, 1, &[*x]),
            MatOut::Vector(x) => numeric_matrix(name, x.len(), 1, x),
            MatOut::Strings(s) => cell_of_strings(name, s),
        };
        push_element(&mut b, MI_MATRIX, &m);
    }
    std::fs::write(path, b).with_context(|| format!("writing {}", path.display()))
}

/// Writes general values (numeric arrays, strings, cells, struct arrays), uncompressed.
/// The file is written under a temporary name and renamed, so an interrupted run never
/// leaves a half-written file under `path`.
pub fn write_values(path: &Path, vars: &[(&str, &MatValue)]) -> Result<()> {
    let mut b = header();
    for (name, v) in vars {
        let m = encode(name, v)?;
        push_element(&mut b, MI_MATRIX, &m);
    }
    let tmp = path.with_extension("mat.tmp");
    std::fs::write(&tmp, b).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("renaming {} to {}", tmp.display(), path.display()))
}

fn encode(name: &str, v: &MatValue) -> Result<Vec<u8>> {
    Ok(match v {
        MatValue::Numeric { dims, data } => {
            let mut b = Vec::with_capacity(64 + 8 * data.len());
            matrix_header_dims(&mut b, MX_DOUBLE, dims, name);
            let n = 8 * data.len();
            b.extend_from_slice(&MI_DOUBLE.to_le_bytes());
            b.extend_from_slice(&(n as u32).to_le_bytes());
            for v in data {
                b.extend_from_slice(&v.to_le_bytes());
            }
            b
        }
        MatValue::Char(s) => char_matrix(name, s),
        MatValue::Cell { dims, items } => {
            let mut b = Vec::new();
            matrix_header_dims(&mut b, MX_CELL, dims, name);
            for it in items {
                push_element(&mut b, MI_MATRIX, &encode("", it)?);
            }
            b
        }
        MatValue::Struct { dims, fields, elems } => {
            let mut b = Vec::new();
            matrix_header_dims(&mut b, MX_STRUCT, dims, name);
            let len = fields.iter().map(|f| f.len()).max().unwrap_or(0) + 1;
            if len > 64 {
                bail!("struct field names longer than 63 characters cannot be written");
            }
            push_element(&mut b, MI_INT32, &(len as i32).to_le_bytes());
            let mut names = vec![0u8; len * fields.len()];
            for (k, f) in fields.iter().enumerate() {
                names[k * len..k * len + f.len()].copy_from_slice(f.as_bytes());
            }
            push_element(&mut b, MI_INT8, &names);
            for e in elems {
                if e.len() != fields.len() {
                    bail!("struct element with {} values for {} fields", e.len(), fields.len());
                }
                for v in e {
                    push_element(&mut b, MI_MATRIX, &encode("", v)?);
                }
            }
            b
        }
        MatValue::Unsupported(c) => bail!("cannot write a value of MATLAB class {c}"),
    })
}

fn header() -> Vec<u8> {
    let mut b = Vec::with_capacity(4096);
    let mut header = format!(
        "MATLAB 5.0 MAT-file, Platform: dynare-rust, Created on: {}",
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
    )
    .into_bytes();
    header.resize(116, b' ');
    b.extend_from_slice(&header);
    b.extend_from_slice(&[0u8; 8]);
    b.extend_from_slice(&0x0100u16.to_le_bytes());
    b.extend_from_slice(b"IM");
    b
}

fn push_element(b: &mut Vec<u8>, ty: u32, data: &[u8]) {
    b.extend_from_slice(&ty.to_le_bytes());
    b.extend_from_slice(&(data.len() as u32).to_le_bytes());
    b.extend_from_slice(data);
    b.resize(b.len().div_ceil(8) * 8, 0);
}

fn matrix_header(b: &mut Vec<u8>, class: u8, rows: usize, cols: usize, name: &str) {
    matrix_header_dims(b, class, &[rows, cols], name);
}

fn matrix_header_dims(b: &mut Vec<u8>, class: u8, dims: &[usize], name: &str) {
    let mut flags = [0u8; 8];
    flags[0] = class;
    push_element(b, MI_UINT32, &flags);
    let dims: Vec<u8> = dims.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect();
    push_element(b, MI_INT32, &dims);
    push_element(b, MI_INT8, name.as_bytes());
}

fn numeric_matrix(name: &str, rows: usize, cols: usize, data: &[f64]) -> Vec<u8> {
    let mut b = Vec::new();
    matrix_header(&mut b, MX_DOUBLE, rows, cols, name);
    let raw: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    push_element(&mut b, MI_DOUBLE, &raw);
    b
}

fn char_matrix(name: &str, s: &str) -> Vec<u8> {
    let units: Vec<u16> = s.encode_utf16().collect();
    let mut b = Vec::new();
    matrix_header(&mut b, MX_CHAR, 1, units.len(), name);
    let raw: Vec<u8> = units.iter().flat_map(|u| u.to_le_bytes()).collect();
    push_element(&mut b, MI_UINT16, &raw);
    b
}

fn cell_of_strings(name: &str, s: &[String]) -> Vec<u8> {
    let mut b = Vec::new();
    matrix_header(&mut b, MX_CELL, s.len(), 1, name);
    for item in s {
        push_element(&mut b, MI_MATRIX, &char_matrix("", item));
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let dir = std::env::temp_dir().join(format!("dynare_rust_mat_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.mat");
        let m = Mat::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, -6.5]);
        let names = vec!["crpi".to_string(), "SE_é".to_string()];
        write(
            &p,
            &[("hh", MatOut::Matrix(&m)), ("fval", MatOut::Scalar(3.5)), ("parameter_names", MatOut::Strings(&names))],
        )
        .unwrap();
        let r = read(&p).unwrap();
        assert_eq!(r["hh"].as_matrix().unwrap(), m);
        assert_eq!(r["fval"].as_scalar(), Some(3.5));
        assert_eq!(r["parameter_names"].as_strings().unwrap(), names);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn roundtrip_structs() {
        let dir = std::env::temp_dir().join(format!("dynare_rust_mats_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.mat");
        let seeds = MatValue::Struct {
            dims: vec![1, 2],
            fields: vec!["Unifor".into(), "algo".into()],
            elems: vec![
                vec![MatValue::column(&[1.0, 2.0]), MatValue::Char("chacha8".into())],
                vec![MatValue::column(&[3.0]), MatValue::Char("x".into())],
            ],
        };
        let rec = MatValue::structure(vec![
            ("Nblck", MatValue::scalar(2.0)),
            ("MhDraws", MatValue::Numeric { dims: vec![2, 3], data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0] }),
            ("Sampler", MatValue::Char("random_walk_metropolis_hastings".into())),
            ("LastSeeds", seeds),
            ("Empty", MatValue::Numeric { dims: vec![0, 0], data: vec![] }),
        ]);
        write_values(&p, &[("record", &rec), ("version", &MatValue::scalar(2.0))]).unwrap();
        let r = read(&p).unwrap();
        assert_eq!(r["record"], rec);
        assert_eq!(r["record"].field("LastSeeds").unwrap().field_of(1, "algo").unwrap().as_str(), Some("x"));
        assert_eq!(r["version"].as_scalar(), Some(2.0));
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn reads_dynare_mode_file() {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("reference/sw07/sw07/Output/sw07_mode.mat");
        let r = read(&p).unwrap();
        assert_eq!(r["xparam1"].as_vec().unwrap().len(), 40);
        assert_eq!(r["hh"].as_matrix().unwrap().shape(), (40, 40));
        let names = r["parameter_names"].as_strings().unwrap();
        assert_eq!((names[0].as_str(), names[39].as_str()), ("stderr ea", "calfa"));
        // stage-2 newrat mode (docs/REFERENCES.md)
        assert!((r["fval"].as_scalar().unwrap() - 349.856047).abs() < 1e-6);
    }
}
