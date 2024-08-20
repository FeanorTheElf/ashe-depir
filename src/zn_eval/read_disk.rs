use std::fs;
use std::mem::size_of;
use std::sync::atomic::AtomicU64;

use feanor_math::homomorphism::CanHomFrom;
use feanor_math::integer::int_cast;
use feanor_math::primitive_int::*;
use feanor_math::ring::{El, RingStore};
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::vector::vec_fn::VectorFn;

use super::async_read::{perform_reads_async, ReadPosition};
use super::{TestPolynomialTracker, ZnEvaluator, ZnEvaluatorCallback};
use crate::polynomial::io::*;
use crate::strategy::ModulusInfo;
use crate::DATASTRUCTURE_PATH;

pub static PERFORMED_DISK_QUERIES: AtomicU64 = AtomicU64::new(0);

pub struct ReadDiskEvaluator<Zn: ZnRingStore, const m: usize>
    where Zn::Type: ZnRing
{
    filename: String,
    R: Zn,
    p: i64,
    poly_count: usize
}


impl<Zn: ZnRingStore, const m: usize> ReadDiskEvaluator<Zn, m>
    where Zn::Type: ZnRing
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
    pub fn initialize_for_file(R: Zn, p: i64, poly_count: usize, filename: &str) -> Self {
        let filepath = format!("{}{}", DATASTRUCTURE_PATH, filename);
        let result = Self {
            filename: filepath.clone(),
            R: R,
            p: p,
            poly_count: poly_count,
        };
        assert!(fs::metadata(filepath.as_str()).is_ok(), "Error locating file {}", filepath);
        let expected_size = size_of::<EvaluationsUInt>() as u64 * poly_count as u64 * StaticRing::<i64>::RING.pow(p, m) as u64;
        let actual_size = fs::metadata(filepath.as_str()).unwrap().len();
        assert_eq!(expected_size, actual_size, "Evaluations file {}{} has wrong length, {} instead of {}", DATASTRUCTURE_PATH, filename, actual_size, expected_size);
        return result;
    }
    
    pub fn expected_size_in_bytes(info: ModulusInfo) -> usize {
        size_of::<EvaluationsUInt>() * info.storage_elements()
    }

    pub fn actual_size_in_bytes(&self) -> usize {
        size_of::<EvaluationsUInt>() * self.poly_count * StaticRing::<i64>::RING.pow(self.p, m) as usize
    }

    fn compute_index(&self, x: [El<Zn>; m]) -> usize {
        let mut result = 0;
        for i in (0..m).rev() {
            debug_assert!(int_cast(self.R.smallest_lift(self.R.clone_el(&x[i])), &StaticRing::<i64>::RING, self.R.integer_ring()).abs() <= self.p / 2);
            result = result * self.p as usize + (int_cast(self.R.smallest_lift(self.R.clone_el(&x[i])), &StaticRing::<i64>::RING, self.R.integer_ring()) + self.p / 2) as usize;
        }
        return result * self.poly_count;
    }
}

impl<Zn: ZnRingStore, const m: usize> ZnEvaluator<Zn, m> for ReadDiskEvaluator<Zn, m>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    fn evaluate_many<Index, Points, Callback>(&self, entries: Points, callback: Callback, _test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<Zn>; m])>,
            Index: Copy,
            Callback: ZnEvaluatorCallback<Zn, Index>
    {
        perform_reads_async(|context| {
            let mut file = context.open_file(self.filename.as_str(), |read_data, request: ReadPosition<Index>| {
                callback.call(request.read_request_index, (0..self.poly_count).map(|i| decode_zn_el::<_, EvaluationsUInt>(self.ring(), &read_data[(i * size_of::<EvaluationsUInt>())..((i + 1) * size_of::<EvaluationsUInt>())])), self.ring());
            });
            for i in 0..entries.len() {
                let (idx, point) = entries.at(i);
                let read_index = self.compute_index(point);
                file.submit(idx, read_index * size_of::<EvaluationsUInt>(), size_of::<EvaluationsUInt>() * self.poly_count);
            }
            PERFORMED_DISK_QUERIES.fetch_add(entries.len() as u64, std::sync::atomic::Ordering::Relaxed);
        });
    }
    
    fn ring(&self) -> &Zn {
        &self.R
    }

    fn poly_count(&self) -> usize {
        self.poly_count
    }
}

#[cfg(test)]
use std::panic::UnwindSafe;
#[cfg(test)]
use std::panic::catch_unwind;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;
#[cfg(test)]
use feanor_math::rings::multivariate::*;
#[cfg(test)]
use super::test::create_storage_evaluator_test_data;
#[cfg(test)]
use std::cell::Cell;
#[cfg(test)]
use feanor_math::homomorphism::Homomorphism;
#[cfg(test)]
use feanor_math::vector::vec_fn::*;
#[cfg(test)]
use feanor_math::iters::cartesian_product;
#[cfg(test)]
use feanor_math::rings::zn::zn_64;

#[cfg(test)]
fn test_with_testfile<F>(testfile: &str, testfile_data: &[u8], base: F)
    where F: FnOnce() + UnwindSafe
{
    fs::write(testfile, testfile_data).unwrap();
    let result = catch_unwind(base);
    fs::remove_file(testfile).unwrap();
    result.unwrap();
}

#[test]
fn test_evaluate() {
    let (poly_ring, polys, R, data) = create_storage_evaluator_test_data();
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);
    let f1 = poly_ring.clone_el(&polys[0]);
    let f2 = poly_ring.clone_el(&polys[1]);
    let modulo = R.can_hom(&StaticRing::<i64>::RING).unwrap();

    test_with_testfile("testfile_test_evaluate", &data.into_iter().flat_map(|x| encode_zn_el::<_, EvaluationsUInt>(R, x)).collect::<Vec<_>>(), || {

        let evaluator = ReadDiskEvaluator {
            R: R,
            p: 7,
            poly_count: 2,
            filename: "testfile_test_evaluate".to_owned()
        };
        for x in R.elements() {
            for y in R.elements() {
                assert_el_eq!(&R, &poly_ring.evaluate(&f1, [R.zero(), R.zero(), x, y], &modulo), &evaluator.evaluate_sync([R.zero(), R.zero(), x, y], &test_polynomial_tracker)[0]);
                assert_el_eq!(&R, &poly_ring.evaluate(&f2, [R.zero(), R.zero(), x, y], &modulo), &evaluator.evaluate_sync([R.zero(), R.zero(), x, y], &test_polynomial_tracker)[1]);
            }
        }

    });
}

#[test]
fn test_evaluate_multiple() {
    let (poly_ring, polys, R, data) = create_storage_evaluator_test_data();
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);
    
    test_with_testfile("testfile_test_evaluate_multiple", &data.into_iter().flat_map(|x| encode_zn_el::<_, EvaluationsUInt>(R, x)).collect::<Vec<_>>(), || {

        let evaluator = ReadDiskEvaluator {
            R: R,
            p: 7,
            poly_count: 2,
            filename: "testfile_test_evaluate_multiple".to_owned()
        };
        let points = cartesian_product(R.elements(), R.elements()).map(|(x, y)| ((R.smallest_positive_lift(x), R.smallest_positive_lift(y)), [R.zero(), R.zero(), x, y]));
        let count = Cell::new(0);

        struct TestCallback<'a>(&'a Cell<usize>);

        impl<'a> ZnEvaluatorCallback<zn_64::Zn, (i64, i64)> for TestCallback<'a> {

            fn call<I>(&self, (x, y): (i64, i64), mut ys: I, ring: &zn_64::Zn)
                where I: ExactSizeIterator<Item = El<zn_64::Zn>>
            {
                assert_eq!(2, ys.len());

                self.0.update(|x| x + 1);
                assert_el_eq!(ring, &ring.int_hom().map((1 - x + 2 * y + x * y * y) as i32), &ys.next().unwrap());
                assert_el_eq!(ring, &ring.int_hom().map((1 + 2 * x * x - y) as i32), &ys.next().unwrap());
            }
        }

        evaluator.evaluate_many(points.collect::<Vec<_>>().into_fn(), TestCallback(&count), &test_polynomial_tracker);

        assert_eq!(R.size(&StaticRing::<i64>::RING).unwrap() * R.size(&StaticRing::<i64>::RING).unwrap(), count.get() as i64);
    });
}
