// SPDX-License-Identifier: GPL-3.0-or-later
//! MATLAB v7.3 MAT-files (`save -v7.3`): an HDF5 file after a 512-byte user block. Read-only,
//! enough for the data files of Dynare's tests: superblock version 0/1, version 1 object
//! headers, groups as symbol tables (the root group: the variables; a sub-group: a struct),
//! datasets of integer/float/string/reference types with compact, contiguous or chunked
//! (deflate/shuffle) layout, and the `MATLAB_class` / `MATLAB_empty` attributes. Cells are
//! datasets of object references into `#refs#`.
//!
//! HDF5 stores the dimensions in C order: MATLAB's are the reverse, with the same data
//! (MATLAB's column-major order).

use std::collections::BTreeMap;
use std::io::Read;

use anyhow::{Context, Result, bail};
use flate2::read::ZlibDecoder;

use crate::matfile::MatValue;

struct H5<'a> {
    b: &'a [u8],
    base: usize,
    so: usize,
    sl: usize,
}

/// The messages of an object header: (type, payload).
type Messages<'a> = Vec<(u16, &'a [u8])>;

fn uint(d: &[u8], n: usize) -> u64 {
    d[..n].iter().rev().fold(0u64, |a, &x| (a << 8) | x as u64)
}

impl<'a> H5<'a> {
    fn at(&self, addr: u64, len: usize) -> Result<&'a [u8]> {
        let p = self.base + addr as usize;
        self.b.get(p..p + len).with_context(|| format!("HDF5 address {addr:#x} out of the file"))
    }

    fn addr_undefined(&self, a: u64) -> bool {
        a == if self.so >= 8 { u64::MAX } else { (1u64 << (8 * self.so)) - 1 }
    }

    /// Version 1 object header at `addr` (continuation blocks followed).
    fn messages(&self, addr: u64) -> Result<Messages<'a>> {
        let h = self.at(addr, 16)?;
        if h[0] != 1 {
            bail!("HDF5 object header version {} is not supported", h[0]);
        }
        let nmsg = uint(&h[2..], 2) as usize;
        let size = uint(&h[8..], 4) as usize;
        let mut blocks = vec![(addr + 16, size)];
        let mut out = Vec::new();
        let mut k = 0;
        while k < blocks.len() && out.len() < nmsg {
            let (a, len) = blocks[k];
            k += 1;
            let blk = self.at(a, len)?;
            let mut p = 0;
            while p + 8 <= blk.len() {
                let ty = uint(&blk[p..], 2) as u16;
                let sz = uint(&blk[p + 2..], 2) as usize;
                let data = blk.get(p + 8..p + 8 + sz).context("HDF5 message runs past its header")?;
                if ty == 0x10 {
                    blocks.push((uint(data, self.so), uint(&data[self.so..], self.sl) as usize));
                } else {
                    out.push((ty, data));
                }
                p += 8 + sz.div_ceil(8) * 8;
            }
        }
        Ok(out)
    }

    /// The links of a group (symbol table message): name -> object header address.
    fn group(&self, btree: u64, heap: u64) -> Result<Vec<(String, u64)>> {
        let hp = self.at(heap, 8 + 2 * self.sl + self.so)?;
        if &hp[..4] != b"HEAP" {
            bail!("HDF5: local heap signature not found");
        }
        let heap_data = uint(&hp[8 + 2 * self.sl..], self.so);
        let mut out = Vec::new();
        self.group_node(btree, heap_data, &mut out)?;
        Ok(out)
    }

    fn group_node(&self, node: u64, heap_data: u64, out: &mut Vec<(String, u64)>) -> Result<()> {
        let h = self.at(node, 8 + 2 * self.so)?;
        if &h[..4] == b"SNOD" {
            let n = uint(&h[6..], 2) as usize;
            let ent = 2 * self.so + 24;
            let d = self.at(node + 8, n * ent)?;
            for e in d.chunks(ent) {
                let name_off = uint(e, self.so);
                let obj = uint(&e[self.so..], self.so);
                let s =
                    self.at(heap_data + name_off, 0).map(|_| &self.b[self.base + (heap_data + name_off) as usize..])?;
                let end = s.iter().position(|&c| c == 0).unwrap_or(s.len());
                out.push((String::from_utf8_lossy(&s[..end]).into_owned(), obj));
            }
            return Ok(());
        }
        if &h[..4] != b"TREE" || h[4] != 0 {
            bail!("HDF5: group B-tree node expected");
        }
        let used = uint(&h[6..], 2) as usize;
        let body = self.at(node + 8 + 2 * self.so as u64, (used + 1) * self.sl + used * self.so)?;
        for k in 0..used {
            let child = uint(&body[self.sl + k * (self.sl + self.so)..], self.so);
            self.group_node(child, heap_data, out)?;
        }
        Ok(())
    }

    /// The MATLAB value of the object at `addr`.
    fn value(&self, addr: u64) -> Result<MatValue> {
        let msgs = self.messages(addr)?;
        let mut class = String::new();
        let mut empty = false;
        for (_, d) in msgs.iter().filter(|(t, _)| *t == 0x0c) {
            let (name, val) = attribute(d)?;
            match name.as_str() {
                "MATLAB_class" => class = String::from_utf8_lossy(val).trim_end_matches('\0').to_string(),
                "MATLAB_empty" => empty = val.iter().any(|&x| x != 0),
                _ => {}
            }
        }
        if let Some((_, st)) = msgs.iter().find(|(t, _)| *t == 0x11) {
            // a group: a struct (1x1)
            let links = self.group(uint(st, self.so), uint(&st[self.so..], self.so))?;
            let mut fields = Vec::new();
            let mut vals = Vec::new();
            for (n, a) in links {
                vals.push(self.value(a)?);
                fields.push(n);
            }
            return Ok(MatValue::Struct { dims: vec![1, 1], fields, elems: vec![vals] });
        }
        let space = msgs.iter().find(|(t, _)| *t == 1).context("HDF5 dataset without a dataspace")?.1;
        let dtype = msgs.iter().find(|(t, _)| *t == 3).context("HDF5 dataset without a datatype")?.1;
        let layout = msgs.iter().find(|(t, _)| *t == 8).context("HDF5 dataset without a layout")?.1;
        let filters = msgs.iter().find(|(t, _)| *t == 0x0b).map(|m| m.1);
        let hdims = dataspace(space, self.sl)?;
        let count: usize = hdims.iter().product();
        let esize = uint(&dtype[4..], 4) as usize;
        let raw = self.data(layout, filters, &hdims, esize)?;
        let mut dims: Vec<usize> = hdims.iter().rev().copied().collect();
        while dims.len() < 2 {
            dims.push(1);
        }
        let tclass = dtype[0] & 0x0f;
        if tclass == 7 {
            // object references (a cell array)
            let items = raw.chunks(esize).take(count).map(|r| self.value(uint(r, self.so))).collect::<Result<_>>()?;
            return Ok(MatValue::Cell { dims, items });
        }
        let nums: Vec<f64> = match tclass {
            0 => {
                let signed = dtype[1] & 0x08 != 0;
                raw.chunks(esize)
                    .take(count)
                    .map(|c| {
                        let u = uint(c, esize);
                        if signed {
                            let sh = 64 - 8 * esize as u32;
                            ((u << sh) as i64 >> sh) as f64
                        } else {
                            u as f64
                        }
                    })
                    .collect()
            }
            1 if esize == 8 => raw.chunks(8).take(count).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect(),
            1 if esize == 4 => {
                raw.chunks(4).take(count).map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64).collect()
            }
            3 => return Ok(MatValue::Char(String::from_utf8_lossy(&raw).trim_end_matches('\0').to_string())),
            c => bail!("HDF5 datatype class {c} is not supported"),
        };
        if empty {
            // MATLAB_empty: the data are the dimensions
            let dims: Vec<usize> = nums.iter().map(|&v| v as usize).collect();
            return Ok(if class == "char" {
                MatValue::Char(String::new())
            } else {
                MatValue::Numeric { dims, data: vec![] }
            });
        }
        match class.as_str() {
            "char" => {
                let units: Vec<u16> = nums.iter().map(|&v| v as u16).collect();
                Ok(MatValue::Char(String::from_utf16_lossy(&units)))
            }
            "cell" | "struct" | "function_handle" => Ok(MatValue::Unsupported(0)),
            _ if dtype[0] & 0x0f == 6 => Ok(MatValue::Unsupported(0)), // complex (compound)
            _ => Ok(MatValue::Numeric { dims, data: nums }),
        }
    }

    /// The raw bytes of a dataset (`count * esize`).
    fn data(&self, layout: &[u8], filters: Option<&[u8]>, hdims: &[usize], esize: usize) -> Result<Vec<u8>> {
        let count: usize = hdims.iter().product();
        if layout[0] != 3 {
            bail!("HDF5 data layout version {} is not supported", layout[0]);
        }
        match layout[1] {
            0 => {
                let n = uint(&layout[2..], 2) as usize;
                Ok(layout[4..4 + n].to_vec())
            }
            1 => {
                let a = uint(&layout[2..], self.so);
                if self.addr_undefined(a) {
                    return Ok(vec![0; count * esize]);
                }
                Ok(self.at(a, count * esize)?.to_vec())
            }
            2 => {
                let r = layout[2] as usize; // rank + 1 (the element size)
                let btree = uint(&layout[3..], self.so);
                let cdims: Vec<usize> = (0..r - 1).map(|k| uint(&layout[3 + self.so + 4 * k..], 4) as usize).collect();
                let pipeline = match filters {
                    Some(f) => pipeline(f)?,
                    None => vec![],
                };
                let mut out = vec![0u8; count * esize];
                if !self.addr_undefined(btree) {
                    self.chunks(btree, r, &cdims, hdims, esize, &pipeline, &mut out)?;
                }
                Ok(out)
            }
            c => bail!("HDF5 layout class {c} is not supported"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn chunks(
        &self,
        node: u64,
        r: usize,
        cdims: &[usize],
        hdims: &[usize],
        esize: usize,
        pipeline: &[u16],
        out: &mut [u8],
    ) -> Result<()> {
        let h = self.at(node, 8 + 2 * self.so)?;
        if &h[..4] != b"TREE" || h[4] != 1 {
            bail!("HDF5: chunk B-tree node expected");
        }
        let level = h[5];
        let used = uint(&h[6..], 2) as usize;
        let key = 8 + 8 * r;
        let body = self.at(node + 8 + 2 * self.so as u64, (used + 1) * key + used * self.so)?;
        for k in 0..used {
            let kd = &body[k * (key + self.so)..];
            let child = uint(&kd[key..], self.so);
            if level > 0 {
                self.chunks(child, r, cdims, hdims, esize, pipeline, out)?;
                continue;
            }
            let size = uint(kd, 4) as usize;
            let mask = uint(&kd[4..], 4) as u32;
            let offs: Vec<usize> = (0..r - 1).map(|j| uint(&kd[8 + 8 * j..], 8) as usize).collect();
            let mut c = self.at(child, size)?.to_vec();
            for (i, &f) in pipeline.iter().enumerate().rev() {
                if mask & (1 << i) != 0 {
                    continue;
                }
                c = match f {
                    1 => {
                        let mut v = Vec::new();
                        ZlibDecoder::new(&c[..]).read_to_end(&mut v).context("inflating an HDF5 chunk")?;
                        v
                    }
                    2 => {
                        // shuffle: byte k of every element stored together
                        let n = c.len() / esize;
                        let mut v = vec![0u8; c.len()];
                        for e in 0..n {
                            for b in 0..esize {
                                v[e * esize + b] = c[b * n + e];
                            }
                        }
                        v
                    }
                    f => bail!("HDF5 filter {f} is not supported"),
                };
            }
            // copy the chunk's elements (row-major in the chunk) into the dataset
            let nc: usize = cdims.iter().product();
            let mut idx = vec![0usize; cdims.len()];
            for e in 0..nc {
                if e > 0 {
                    for j in (0..cdims.len()).rev() {
                        idx[j] += 1;
                        if idx[j] < cdims[j] {
                            break;
                        }
                        idx[j] = 0;
                    }
                }
                let mut g = 0usize;
                let mut inside = true;
                for j in 0..cdims.len() {
                    let x = offs[j] + idx[j];
                    inside &= x < hdims[j];
                    g = g * hdims[j] + x;
                }
                if inside && (e + 1) * esize <= c.len() {
                    out[g * esize..(g + 1) * esize].copy_from_slice(&c[e * esize..(e + 1) * esize]);
                }
            }
        }
        Ok(())
    }
}

/// Dataspace message: the dimensions (C order).
fn dataspace(d: &[u8], sl: usize) -> Result<Vec<usize>> {
    let rank = d[1] as usize;
    let start = match d[0] {
        1 => 8,
        2 => 4,
        v => bail!("HDF5 dataspace version {v} is not supported"),
    };
    Ok((0..rank).map(|k| uint(&d[start + k * sl..], sl) as usize).collect())
}

/// Filter pipeline message: the filter ids in order.
fn pipeline(d: &[u8]) -> Result<Vec<u16>> {
    let v = d[0];
    let n = d[1] as usize;
    let mut p = if v == 1 { 8 } else { 2 };
    let mut out = Vec::new();
    for _ in 0..n {
        let id = uint(&d[p..], 2) as u16;
        let (name_len, q) = if v == 1 || id >= 256 { (uint(&d[p + 2..], 2) as usize, p + 4) } else { (0, p + 2) };
        let nvals = uint(&d[q + 2..], 2) as usize;
        p = q + 4;
        p += if v == 1 { name_len.div_ceil(8) * 8 } else { name_len };
        p += 4 * nvals;
        if v == 1 && nvals % 2 == 1 {
            p += 4;
        }
        out.push(id);
    }
    Ok(out)
}

/// Attribute message: (name, raw value bytes).
fn attribute(d: &[u8]) -> Result<(String, &[u8])> {
    let v = d[0];
    let nlen = uint(&d[2..], 2) as usize;
    let tlen = uint(&d[4..], 2) as usize;
    let slen = uint(&d[6..], 2) as usize;
    let (pad, mut p) = match v {
        1 => (true, 8),
        2 => (false, 8),
        3 => (false, 9),
        _ => bail!("HDF5 attribute version {v} is not supported"),
    };
    let r = |n: usize| if pad { n.div_ceil(8) * 8 } else { n };
    let name = String::from_utf8_lossy(&d[p..p + nlen]).trim_end_matches('\0').to_string();
    p += r(nlen) + r(tlen) + r(slen);
    Ok((name, &d[p.min(d.len())..]))
}

/// Reads every variable of a v7.3 MAT-file (`bytes` is the whole file).
pub fn parse(bytes: &[u8]) -> Result<BTreeMap<String, MatValue>> {
    // the superblock is at 0, 512, 1024, ...
    let mut base = 0;
    while bytes.get(base..base + 8).context("HDF5 superblock not found")? != b"\x89HDF\r\n\x1a\n" {
        base = if base == 0 { 512 } else { base * 2 };
    }
    let sb = &bytes[base..];
    let version = sb[8];
    if version > 1 {
        bail!("HDF5 superblock version {version} is not supported");
    }
    let so = sb[13] as usize;
    let sl = sb[14] as usize;
    let mut p = 24 + if version == 1 { 4 } else { 0 };
    let base_addr = uint(&sb[p..], so) as usize;
    p += 4 * so; // base, free-space, end of file, driver info
    let root = &sb[p..];
    let h = H5 { b: bytes, base: base_addr, so, sl };
    let root_obj = uint(&root[so..], so);
    // the root group's symbol table: from the cache of its entry, else from its header
    let (btree, heap) = if uint(&root[2 * so..], 4) == 1 {
        (uint(&root[2 * so + 8..], so), uint(&root[3 * so + 8..], so))
    } else {
        let msgs = h.messages(root_obj)?;
        let st = msgs.iter().find(|(t, _)| *t == 0x11).context("HDF5 root group without a symbol table")?.1;
        (uint(st, so), uint(&st[so..], so))
    };
    let mut out = BTreeMap::new();
    for (name, a) in h.group(btree, heap)? {
        if name.starts_with('#') {
            continue;
        }
        let v = h.value(a).with_context(|| format!("variable {name}"))?;
        out.insert(name, v);
    }
    Ok(out)
}
