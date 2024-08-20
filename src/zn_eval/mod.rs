use std::cell::RefCell;
use std::hash::Hash;

use feanor_math::homomorphism::*;
use feanor_math::integer::*;
use feanor_math::mempool::DefaultMemoryProvider;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::rings::multivariate::ordered::MultivariatePolyRingImpl;
use feanor_math::rings::multivariate::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::vector::vec_fn::{IntoVectorFn, VectorFn};

mod prime_components;
pub mod async_read;
pub mod read_ram;
pub mod read_disk;
pub mod crt_evaluator;
pub mod zp2_evaluator;

pub trait ZnEvaluatorCallback<Zn: ZnRingStore, Index>
    where Zn::Type: ZnRing
{
    fn call<I>(&self, idx: Index, ys: I, ring: &Zn)
        where I: ExactSizeIterator<Item = El<Zn>>;   
}

pub trait ZnEvaluator<Zn: ZnRingStore, const m: usize>
    where Zn::Type: ZnRing
{
    fn evaluate_many<Index, Points, Callback>(&self, entries: Points, callback: Callback, test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<Zn>; m])>,
            Index: 'static + std::fmt::Debug + Hash + Eq + Ord + Copy,
            Callback: ZnEvaluatorCallback<Zn, Index>;

    fn ring(&self) -> &Zn;

    fn poly_count(&self) -> usize;

    ///
    /// A synchronous equivalent to [`evaluate_many()`]. Use for testing, but not in
    /// performance-criticial situations.
    /// 
    fn evaluate_sync(&self, values: [El<Zn>; m], test_polynomial_tracker: &TestPolynomialTracker<m>) -> Vec<El<Zn>>
        where El<Zn>: Clone
    {
        let result = RefCell::new(None);

        struct Callback<'a, Zn>(&'a RefCell<Option<Vec<El<Zn>>>>) where Zn: ZnRingStore, Zn::Type: ZnRing;

        impl<'a, Zn, Index> ZnEvaluatorCallback<Zn, Index> for Callback<'a, Zn> where Zn: ZnRingStore, Zn::Type: ZnRing {
            fn call<I>(&self, _idx: Index, ys: I, _: &Zn)
                where I: ExactSizeIterator<Item = El<Zn>>
            {
                *self.0.borrow_mut() = Some(ys.collect());
            }
        }

        self.evaluate_many([((), values)].into_fn(), Callback(&result), test_polynomial_tracker);
        return result.into_inner().unwrap();
    }
}

impl<Zn: ZnRingStore, const m: usize> ZnEvaluator<Zn, m> for !
    where Zn::Type: ZnRing
{
    fn ring(&self) -> &Zn { *self }

    fn evaluate_many<Index, Points, Callback>(&self, _entries: Points, _callback: Callback, _test_polynomial_tracker: &TestPolynomialTracker<m>)
            where Points: VectorFn<(Index, [El<Zn>; m])>,
                Index: std::fmt::Debug + Hash + Eq + Ord + Copy,
                Callback: ZnEvaluatorCallback<Zn, Index> { *self }

    fn poly_count(&self) -> usize { *self }
}

pub struct TestEvaluator<'a, Zn: ZnRingStore, const m: usize>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    R: Zn,
    poly_ring: &'a MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>,
    polys: &'a [El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>]
}

impl<'a, Zn: ZnRingStore, const m: usize> TestEvaluator<'a, Zn, m>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    pub fn new(R: Zn, poly_ring: &'a MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>, polys: &'a [El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>]) -> Self {
        Self { R, poly_ring, polys }
    }
}

impl<'a, Zn: ZnRingStore, const m: usize> ZnEvaluator<Zn, m> for TestEvaluator<'a, Zn, m>
    where Zn::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>
{
    fn ring(&self) -> &Zn {
        &self.R
    }

    fn evaluate_many<Index, Points, Callback>(&self, entries: Points, callback: Callback, _test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<Zn>; m])>,
            Callback: ZnEvaluatorCallback<Zn, Index>
    {
        let hom = self.R.can_hom(&StaticRing::<i64>::RING).unwrap();
        for i in 0..entries.len() {
            let (idx, x) = entries.at(i);
            // println!("Access to {}", self.ring().integer_ring().format(self.ring().modulus()));
            callback.call(idx, self.polys.iter().map(|f| self.poly_ring.evaluate(f, &x, &hom)), self.ring());
        }
    }

    fn poly_count(&self) -> usize {
        self.polys.len()
    }
}

#[cfg(feature = "evaluation_runtime_checks")]
pub struct TestPolynomialTracker<const m: usize> {
    poly_ring: MultivariatePolyRingImpl<StaticRing<i32>, DegRevLex, DefaultMemoryProvider, m>,
    polys: Vec<El<MultivariatePolyRingImpl<StaticRing<i32>, DegRevLex, DefaultMemoryProvider, m>>>
}

#[cfg(feature = "evaluation_runtime_checks")]
impl<const m: usize> TestPolynomialTracker<m> {

    fn check_evaluation<R: RingStore>(&self, point: &[El<R>], ring: R, result: &El<R>, i: usize) {
        use feanor_math::assert_el_eq;

        assert_el_eq!(&ring, &self.poly_ring.evaluate(&self.polys[i], point, &ring.int_hom()), result);
    }

    pub fn create_test<R: IntegerRingStore>(poly_ring: &MultivariatePolyRingImpl<R, DegRevLex, DefaultMemoryProvider, m>, polys: &[El<MultivariatePolyRingImpl<R, DegRevLex, DefaultMemoryProvider, m>>]) -> Self
        where R::Type: IntegerRing
    {
        use feanor_math::default_memory_provider;

        let poly_ring_new = MultivariatePolyRingImpl::new(StaticRing::<i32>::RING, DegRevLex, default_memory_provider!());
        Self {
            polys: polys.iter().map(|f| poly_ring_new.from_terms(poly_ring.terms(f).map(|(c, mon)| (int_cast(poly_ring.base_ring().clone_el(c), &StaticRing::<i32>::RING, poly_ring.base_ring()), *mon)))).collect(),
            poly_ring: poly_ring_new,
        }
    }

    pub fn in_production() -> Self {
        panic!("Production code uses TestPolynomialTracker")
    }

    pub fn derived(&self, var_index: usize) -> Self {
        use feanor_math::default_memory_provider;

        let poly_ring_new = MultivariatePolyRingImpl::new(StaticRing::<i32>::RING, DegRevLex, default_memory_provider!());
        Self {
            polys: self.polys.iter().map(|f| poly_ring_new.from_terms(self.poly_ring.terms(f).map(|(c, mon)| (*c * mon[var_index] as i32, Monomial::new(std::array::from_fn(|i| if i == var_index { mon[var_index].saturating_sub(1) } else { mon[i] })))))).collect(),
            poly_ring: poly_ring_new
        }
    }
}

#[cfg(not(feature = "evaluation_runtime_checks"))]
pub struct TestPolynomialTracker<const m: usize> {
    _vars: [(); m]
}

#[cfg(not(feature = "evaluation_runtime_checks"))]
impl<const m: usize> TestPolynomialTracker<m> {

    fn check_evaluation<R: RingStore>(&self, _: &[El<R>], _: R, _: &El<R>, _: usize) {}
    
    pub fn create_test<R: IntegerRingStore>(_: &MultivariatePolyRingImpl<R, DegRevLex, DefaultMemoryProvider, m>, _: &[El<MultivariatePolyRingImpl<R, DegRevLex, DefaultMemoryProvider, m>>]) -> Self
        where R::Type: IntegerRing
    {
        Self::in_production()
    }

    pub fn in_production() -> Self {
        Self {
            _vars: [(); m]
        }
    }

    pub fn derived(&self, _: usize) -> Self {
        Self::in_production()
    }
}

#[cfg(test)]
mod test {

    use feanor_math::{assert_el_eq, default_memory_provider};
    use feanor_math::rings::zn::zn_64;
    use feanor_math::iters::multi_cartesian_product;
    use super::*;

    pub const TEST_M: usize = 4;
    
    ///
    /// Create the two polynomials `1 - x + 2y + xy^2` and `1 + 2x^2 - y`, and the corresponding list of all evaluations
    /// 
    #[cfg(test)]
    pub fn create_storage_evaluator_test_data() -> (MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, TEST_M>, Vec<El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, TEST_M>>>, zn_64::Zn, Vec<El<zn_64::Zn>>) {
    
        let R = zn_64::Zn::new(7);
        let modulo = R.can_hom(&StaticRing::<i64>::RING).unwrap();
        let poly_ring = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
        // 1 - x + 2y + xy^2
        let f1 = poly_ring.from_terms([(1, Monomial::new([0, 0, 0, 0])), (2, Monomial::new([0, 0, 0, 1])), (-1, Monomial::new([0, 0, 1, 0])), (1, Monomial::new([0, 0, 1, 2]))].into_iter());
        // 1 + 2x^2 - y
        let f2 = poly_ring.from_terms([(1, Monomial::new([0, 0, 0, 0])), (2, Monomial::new([0, 0, 2, 0])), (-1, Monomial::new([0, 0, 0, 1]))].into_iter());
        
        let elements = (-3..=3).map(|x| R.int_hom().map(x));
        let data = multi_cartesian_product([elements.clone(), elements.clone(), elements.clone(), elements].into_iter(), |slice| std::array::from_fn::<_, TEST_M, _>(|i| slice[TEST_M - 1 - i]), |_, x| R.clone_el(x))
            .flat_map(|point| [poly_ring.evaluate(&f1, point, &modulo), poly_ring.evaluate(&f2, point, &modulo)].into_iter())
            .collect::<Vec<_>>();
        
        return (poly_ring, vec![f1, f2], R, data)
    }
    
    #[test]
    fn test_test_evaluator() {
        let F = zn_64::Zn::new(60013);
        let P = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
        let f = P.from_terms([(1, Monomial::new([0, 0, 0, 0])), (1, Monomial::new([0, 0, 0, 1]))].into_iter());
        let polys = [f];
        let evaluator = TestEvaluator::new(F, &P, &polys);
        let test_polynomial_tracker = TestPolynomialTracker::create_test(&P, &polys);
        assert_el_eq!(&F, &F.int_hom().map(1), &evaluator.evaluate_sync([F.zero(), F.zero(), F.zero(), F.zero()], &test_polynomial_tracker)[0]);
        assert_el_eq!(&F, &F.int_hom().map(2), &evaluator.evaluate_sync([F.zero(), F.zero(), F.zero(), F.one()], &test_polynomial_tracker)[0]);
    }
}