// SPDX-License-Identifier: GPL-3.0-or-later
//! MATLAB's default random stream (`RandStream('mt19937ar','Seed',0)`, what Dynare's
//! `set_dynare_seed('default')` installs at start-up), so that the GSA samples that Dynare
//! draws with `rand`/`randperm` (Latin hypercube columns of the shock parameters, scrambled
//! LPTAU for more than 30 parameters, Morris trajectories) are reproduced draw for draw.
//!
//! `rand` is Mersenne Twister `genrand_res53` (MATLAB maps seed 0 to MT's default 5489);
//! `randperm(n)` is the sort permutation of `rand(1, n)` (checked against MATLAB R2025b).

const N: usize = 624;
const M: usize = 397;

#[derive(Clone)]
pub struct MatlabRng {
    mt: [u32; N],
    mti: usize,
}

impl MatlabRng {
    /// `RandStream('mt19937ar', 'Seed', seed)`.
    pub fn new(seed: u32) -> Self {
        let s = if seed == 0 { 5489 } else { seed };
        let mut mt = [0u32; N];
        mt[0] = s;
        for i in 1..N {
            mt[i] = 1812433253u32.wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30)).wrapping_add(i as u32);
        }
        MatlabRng { mt, mti: N }
    }

    fn next_u32(&mut self) -> u32 {
        const UPPER: u32 = 0x8000_0000;
        const LOWER: u32 = 0x7fff_ffff;
        const MATRIX_A: u32 = 0x9908_b0df;
        if self.mti >= N {
            for k in 0..N {
                let y = (self.mt[k] & UPPER) | (self.mt[(k + 1) % N] & LOWER);
                let mag = if y & 1 == 1 { MATRIX_A } else { 0 };
                self.mt[k] = self.mt[(k + M) % N] ^ (y >> 1) ^ mag;
            }
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// MATLAB `rand` (one double in (0, 1)).
    pub fn rand(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67108864.0 + b) / 9007199254740992.0
    }

    /// `rand(n, 1)`.
    pub fn rand_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.rand()).collect()
    }

    /// MATLAB `randperm(n)`, 0-based: `[~, p] = sort(rand(1, n))`.
    pub fn randperm(&mut self, n: usize) -> Vec<usize> {
        let r = self.rand_vec(n);
        let mut p: Vec<usize> = (0..n).collect();
        p.sort_by(|&a, &b| r[a].total_cmp(&r[b]));
        p
    }
}

thread_local! {
    /// MATLAB's global stream (`RandStream.getGlobalStream`), for the `rand` draws Dynare
    /// takes from it across commands (csminwel's cliff perturbations).
    static GLOBAL: std::cell::RefCell<MatlabRng> = std::cell::RefCell::new(MatlabRng::new(0));
}

/// `set_dynare_seed(seed)`: a fresh global `mt19937ar` stream (`'default'` is seed 0, what
/// `global_initialization.m` installs at the start of every run).
pub fn reset_global(seed: u32) {
    GLOBAL.with(|g| *g.borrow_mut() = MatlabRng::new(seed));
}

/// Draws from the global stream.
pub fn with_global<R>(f: impl FnOnce(&mut MatlabRng) -> R) -> R {
    GLOBAL.with(|g| f(&mut g.borrow_mut()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_matlab_default_stream() {
        // rng('default'); rand(1,5) and randperm(10) in MATLAB R2025b
        let mut r = MatlabRng::new(0);
        let want =
            [0.81472368639317894, 0.90579193707561922, 0.12698681629350606, 0.91337585613901939, 0.63235924622540951];
        for w in want {
            assert_eq!(r.rand(), w);
        }
        let mut r = MatlabRng::new(0);
        let p: Vec<usize> = r.randperm(10).iter().map(|k| k + 1).collect();
        assert_eq!(p, vec![6, 3, 7, 8, 5, 1, 2, 4, 9, 10]);
        assert_eq!(r.rand(), 0.15761308167754828);
        let mut r = MatlabRng::new(0);
        let p = r.randperm(2048);
        assert_eq!(&p[..6].iter().map(|k| k + 1).collect::<Vec<_>>(), &[1974, 609, 452, 1582, 1517, 1836]);
    }
}
