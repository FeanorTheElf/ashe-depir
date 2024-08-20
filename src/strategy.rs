use std::collections::{hash_map::Entry, HashMap};
use std::fmt::Display;
use std::hash::Hash;
use std::mem::size_of;
use std::cmp::{min, max};

use feanor_math::algorithms;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;

use crate::{component_count, reduced_m};
use crate::polynomial::io::EvaluationsUInt;

const ZZbig: BigIntRing = BigIntRing::RING;
const ZZ: StaticRing<i64> = StaticRing::<i64>::RING;

pub fn sample_primes_arithmetic_progression(a: i64, n: i64, min_bits: usize) -> impl Clone + Iterator<Item = i64> {
    let start = ZZ.euclidean_div(ZZ.power_of_two(min_bits), &n);
    return (start..).map(move |i| i * n + a).filter(move |p| algorithms::miller_rabin::is_prime(ZZ, &(*p as i64), 8)).filter(move |p| *p >= ZZ.power_of_two(min_bits))
}

pub fn binomial(n: usize, mut k: usize) -> u128 {
    if k > n {
        0
    } else {
        k = min(k, n - k);
        ((n - k + 1)..=n).map(|n| n as u128).product::<u128>() / (1..=k).map(|n| n as u128).product::<u128>()
    }
}

pub fn binomial_big(n: usize, mut k: usize) -> El<BigIntRing> {
    if k > n {
        ZZbig.zero()
    } else {
        k = min(k, n - k);
        ZZbig.checked_div(&ZZbig.prod(((n - k + 1)..=n).map(|n| int_cast(n as i64, ZZbig, ZZ))), &ZZbig.prod((1..=k).map(|n| int_cast(n as i64, ZZbig, ZZ)))).unwrap()
    }
}

pub fn joint_reduction_factors<'a, I>(poly_params: &[PolynomialStats], input_size: usize, use_factors: I) -> impl 'a + Iterator<Item = i64>
    where I: 'a + Iterator<Item = i64>
{
    // size bound for the result of the polynomial evaluation
    let bound = poly_params.iter().map(|param| param.poly_eval_bound_log2(input_size as i64)).max_by(f64::total_cmp).unwrap();
    // + 1 to take care of +/-, and + 1 to take care of the slack required by our shortest lift implementation
    let required_bound = bound + 2.;
    assert!(required_bound > 0.);
    
    let result = use_factors.chain(std::iter::once(i64::MIN)).scan(0., move |current: &mut f64, n| {
        if *current <= required_bound {
            assert!(n != i64::MIN, "Not enough primes provided.");
            *current += (n as f64).log2();
            Some(n)
        } else {
            None
        }
    });
    return result;
}

pub fn reduction_factors<'a, I>(poly_params: PolynomialStats, input_size: usize, use_factors: I) -> impl 'a + Iterator<Item = i64>
    where I: 'a + Iterator<Item = i64>
{
    return joint_reduction_factors(&[poly_params], input_size, use_factors);
}

#[derive(Debug, Clone, Copy)]
pub struct ModulusInfo {
    pub modulus: InputStats,
    pub max_index_polynomial_part: usize,
    pub queries: usize
}

impl ModulusInfo {

    pub fn storage_size_file(&self) -> usize {
        self.storage_elements() * size_of::<EvaluationsUInt>()
    }

    pub fn storage_elements(&self) -> usize {
        StaticRing::<i64>::RING.pow(2 * self.modulus.input_size_bound as i64 + 1, reduced_m) as usize * (self.max_index_polynomial_part + 1) * component_count
    }
}

impl Display for ModulusInfo {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ceil_div = |a: usize, b: usize| if a == 0 { 0 } else { (a - 1) / b + 1 };
        match self.max_index_polynomial_part {
            0 => write!(f, "{}: table for f0; accessed {} Mi times; {} Mi stored elements", self.modulus.characteristic, ceil_div(self.queries, 1 << 20), ceil_div(self.storage_elements(), 1 << 20)),
            1 => write!(f, "{}: table for f0, f1; accessed {} Mi times; {} Mi stored elements", self.modulus.characteristic, ceil_div(self.queries, 1 << 20), ceil_div(self.storage_elements(), 1 << 20)),
            c => write!(f, "{}: table for f0, ..., f{}; accessed {} Mi times; {} Mi stored elements", self.modulus.characteristic, c, ceil_div(self.queries, 1 << 20), ceil_div(self.storage_elements(), 1 << 20))
        }
    }
}

pub fn crt_reduction_step<I>(input_params: &[(InputStats, usize)], poly_params: &[PolynomialStats], use_primes: I) -> HashMap<i64, ModulusInfo>
    where I: Clone + Iterator<Item = i64>
{
    let mut lower_level_primes: HashMap<i64, ModulusInfo> = HashMap::new();
    for input in input_params {
        let input_size = input.0.input_size_bound;
        for (i, poly_param) in poly_params.iter().enumerate() {
            for level0_p in reduction_factors(*poly_param, input_size, use_primes.clone()) {
                match lower_level_primes.entry(level0_p) {
                    Entry::Occupied(mut e) => {
                        if i == 0 {
                            e.get_mut().queries += input.1;
                        }
                        e.get_mut().max_index_polynomial_part = max(e.get().max_index_polynomial_part, i);
                    },
                    Entry::Vacant(e) => _ = e.insert(ModulusInfo {
                        modulus: InputStats { characteristic: level0_p, input_size_bound: (level0_p - 1) as usize / 2 + 1 },
                        queries: if i == 0 { input.1 } else { 0 },
                        max_index_polynomial_part: i
                    })
                }
            }
        }
    }
    return lower_level_primes;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PolynomialStats {
    pub m: usize,
    pub D: usize,
    pub monomials: usize,
    pub log_coefficient_size: usize
}

impl PolynomialStats {

    pub fn poly_eval_bound_log2(&self, input_bound: i64) -> f64 {
        // let input_bound = int_cast((char - 1) / 2 + 1, &ZZbig, &ZZ);
        let result = ZZbig.abs_log2_ceil(
            &ZZbig.sum((0..=self.D).map(|k: usize| {
                let degree_k_part = ZZbig.prod([
                    int_cast(binomial(k + self.m - 1, k) as i64, &ZZbig, &ZZ),
                    ZZbig.pow(int_cast(input_bound, &ZZbig, &ZZ), k)
                ].into_iter());
                return degree_k_part;
            }))
        ).unwrap() as f64 + self.log_coefficient_size as f64;
        return result;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct InputStats {
    pub characteristic: i64,
    pub input_size_bound: usize
}

#[cfg(test)]
use std::collections::HashSet;

#[test]
fn test_crt_reduction_step_constant() {
    let result = crt_reduction_step(&[(InputStats { characteristic: 65537, input_size_bound: 32769 }, 1)], &[PolynomialStats { D: 0, m: 4, monomials: 1, log_coefficient_size: 0 }], [2, 3, 5].into_iter());
    assert_eq!([2, 3].into_iter().collect::<HashSet<_>>(), result.keys().copied().collect());
}