use std::mem::size_of;
use std::sync::atomic::AtomicU64;

use feanor_math::integer::int_cast;
use feanor_math::primitive_int::*;
use feanor_math::ring::{El, RingStore};
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::homomorphism::*;

use super::{TestPolynomialTracker, ZnEvaluator, ZnEvaluatorCallback};
use crate::polynomial::io::{read_file, EvaluationsUInt};
use crate::strategy::ModulusInfo;

pub static PERFORMED_RAM_QUERIES: AtomicU64 = AtomicU64::new(0);

pub struct ReadRamEvaluator<Zn: ZnRingStore, const m: usize>
    where Zn::Type: ZnRing
{
    data: Vec<EvaluationsUInt>,
    R: Zn,
    p: i64,
    poly_count: usize
}

impl<Zn: ZnRingStore, const m: usize> ReadRamEvaluator<Zn, m>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    ///
    /// We expect the file to contain the "compressed" evaluations of all polynomials `f1, ..., fr` and
    /// all points. The order is first defined by the lexicographic ordering of the points (where a comparison
    /// in `Fp` is done by comparing the shortest positive lifts), and in case of equality by the order of the
    /// polynomials. In other words, the file should contain
    /// ```text
    /// f1(0, 0, 0), ..., fr(0, 0, 0), f1(1, 0, 0), ..., fr(1, 0, 0), f1(2, 0, 0), ..., fr(2, 0, 0), ...
    /// f1(0, 1, 0), ..., fr(0, 1, 0), f1(1, 1, 0), ..., fr(1, 1, 0), ...
    /// ```
    /// 
    pub fn initialize_from_file(R: Zn, p: i64, poly_count: usize, filename: &str) -> Self {
        let result = Self {
            data: read_file::<_, EvaluationsUInt>(&R, filename).map(|x| x.map(|x| int_cast(R.smallest_positive_lift(x), StaticRing::<i32>::RING, R.integer_ring()) as u16)).collect::<Result<Vec<EvaluationsUInt>, _>>().unwrap(),
            R: R,
            p: p,
            poly_count: poly_count,
        };
        assert_eq!(poly_count * StaticRing::<i64>::RING.pow(p, m) as usize, result.data.len());
        return result;
    }

    pub fn actual_size_in_bytes(&self) -> usize {
        self.data.len() * size_of::<EvaluationsUInt>()
    }

    pub fn expected_size_in_bytes(info: ModulusInfo) -> usize {
        size_of::<EvaluationsUInt>() * info.storage_elements()
    }
    
    fn compute_index(&self, x: [El<Zn>; m]) -> usize {
        let mut result = 0;
        for i in (0..m).rev() {
            debug_assert!(int_cast(self.R.smallest_lift(self.R.clone_el(&x[i])), &StaticRing::<i64>::RING, self.R.integer_ring()).abs() <= self.p / 2);
            result = result * self.p as usize + (int_cast(self.R.smallest_lift(self.R.clone_el(&x[i])), &StaticRing::<i64>::RING, self.R.integer_ring()) + self.p / 2) as usize;
        }
        return result * self.poly_count;
    }

    #[allow(unused)]
    fn prefetch(&self, index: usize) {
        unsafe {
            std::intrinsics::prefetch_read_data(self.data.as_ptr().offset(index as isize), 0);
        }
    }

    fn lookup_index(&self, index: usize) -> El<Zn> {
        self.R.int_hom().map(self.data[index] as i32)
    }
}

// enabling this does not seem to significantly affect performance
const PREFETCH_COUNT: usize = 0;

impl<Zn: ZnRingStore, const m: usize> ZnEvaluator<Zn, m> for ReadRamEvaluator<Zn, m>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    #[inline(never)]
    fn evaluate_many<Index, Points, Callback>(&self, entries: Points, callback: Callback, _test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<Zn>; m])>,
            Callback: ZnEvaluatorCallback<Zn, Index>
    {
        PERFORMED_RAM_QUERIES.fetch_add(entries.len() as u64, std::sync::atomic::Ordering::Relaxed);
        if PREFETCH_COUNT == 0 {
            for i in 0..entries.len() {
                callback.call(entries.at(i).0, (0..self.poly_count).map(|j| self.lookup_index(self.compute_index(entries.at(i).1) + j)), self.ring());
            }
        } else if entries.len() < PREFETCH_COUNT {
            let mut prefetched_data: [Vec<El<Zn>>; PREFETCH_COUNT] = std::array::from_fn(|_| Vec::new());
            for i in 0..entries.len() {
                prefetched_data[i].extend((0..self.poly_count).map(|j| self.lookup_index(self.compute_index(entries.at(i).1) + j)));
            }
            let mut callback_i = 0;
            while callback_i < entries.len() {
                callback.call(entries.at(callback_i).0, prefetched_data[callback_i].drain(..), self.ring());
                callback_i += 1;
            }
        } else {
            let mut prefetched_data: [Vec<El<Zn>>; PREFETCH_COUNT + PREFETCH_COUNT] = std::array::from_fn(|_| Vec::new());
            for i in 0..PREFETCH_COUNT {
                prefetched_data[i].extend((0..self.poly_count).map(|j| self.lookup_index(self.compute_index(entries.at(i).1) + j)));
            }
            let mut callback_i = 0;
            let mut retrieve_i = PREFETCH_COUNT;
            while retrieve_i < entries.len() {
                callback.call(entries.at(callback_i).0, prefetched_data[callback_i % (2 * PREFETCH_COUNT)].drain(..), self.ring());
                prefetched_data[retrieve_i % (2 * PREFETCH_COUNT)].extend((0..self.poly_count).map(|j| self.lookup_index(self.compute_index(entries.at(retrieve_i).1) + j)));
                callback_i += 1;
                retrieve_i += 1
            }
            while callback_i < entries.len() {
                callback.call(entries.at(callback_i).0, prefetched_data[callback_i % (2 * PREFETCH_COUNT)].drain(..), self.ring());
                callback_i += 1;
            }
        }
    }
    
    fn ring(&self) -> &Zn {
        &self.R
    }

    fn poly_count(&self) -> usize {
        self.poly_count
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use super::test::create_storage_evaluator_test_data;
#[cfg(test)]
use feanor_math::rings::multivariate::MultivariatePolyRingStore;

#[test]
fn test_compute_index() {
    let R = zn_64::Zn::new(11);
    let ram_evaluator = ReadRamEvaluator {
        R,
        poly_count: 1,
        p: 11,
        data: Vec::new()
    };
    let i = R.int_hom();
    assert_eq!(1, ram_evaluator.compute_index([i.map(-4), i.map(-5), i.map(-5), i.map(-5)]));
    assert_eq!(2, ram_evaluator.compute_index([i.map(-3), i.map(-5), i.map(-5), i.map(-5)]));
    assert_eq!(3, ram_evaluator.compute_index([i.map(-2), i.map(-5), i.map(-5), i.map(-5)]));
    assert_eq!(4, ram_evaluator.compute_index([i.map(-1), i.map(-5), i.map(-5), i.map(-5)]));
    assert_eq!(15, ram_evaluator.compute_index([i.map(-1), i.map(-4), i.map(-5), i.map(-5)]));
}

#[test]
fn test_evaluate() {
    let (poly_ring, polys, R, data) = create_storage_evaluator_test_data();
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);
    let f1 = poly_ring.clone_el(&polys[0]);
    let f2 = poly_ring.clone_el(&polys[1]);
    let modulo = R.can_hom(&StaticRing::<i64>::RING).unwrap();
    
    let read_ram_evaluator = ReadRamEvaluator {
        R: R,
        poly_count: 2,
        p: 7,
        data: data.into_iter().map(|x| R.smallest_positive_lift(x) as u16).collect()
    };
    for x in R.elements() {
        for y in R.elements() {
            assert_el_eq!(&R, &poly_ring.evaluate(&f1, [R.zero(), R.zero(), x, y], &modulo), &read_ram_evaluator.evaluate_sync([R.zero(), R.zero(), x, y], &test_polynomial_tracker)[0]);
            assert_el_eq!(&R, &poly_ring.evaluate(&f2, [R.zero(), R.zero(), x, y], &modulo), &read_ram_evaluator.evaluate_sync([R.zero(), R.zero(), x, y], &test_polynomial_tracker)[1]);
        }
    }
}
