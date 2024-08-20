use feanor_math::ring::*;
use feanor_math::rings::zn::*;
use feanor_math::integer::*;
use feanor_math::divisibility::*;
use feanor_math::homomorphism::*;
use feanor_math::assert_el_eq;
use feanor_math::vector::vec_fn::IntoVectorFn;
use feanor_math::ordered::*;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::algorithms;

use super::*;

use std::cell::RefCell;
use std::marker::PhantomData;

///
/// An evaluator that splits an evaluation in `Z/p^2Z` into the "upper" and "lower" components.
/// 
/// In other words, when evaluating a polynomial `f(x1, ..., xm)`, we first write each
/// `xi = xi' + p * xi''` with `xi', xi''` at most `p/2`. Then we evaluate suitable polynomials
/// (derived from `f`) at the points `(x1', ..., xm', xi'')`. This way, the downstream-evaluators
/// can work with smaller inputs (e.g. decreasing the precomputed table size), at the cost of an
/// additional variable (and increase in query count).
/// 
pub struct Zp2Evaluator<Zn, ZnBase, BaseEvalLow, BaseEvalHigh, const m: usize>
    where Zn: ZnRingStore,
        ZnBase: ZnRingStore,
        Zn::Type: ZnRing,
        ZnBase::Type: ZnRing,
        BaseEvalLow: ZnEvaluator<ZnBase, m>,
        BaseEvalHigh: ZnEvaluator<ZnBase, m>
{
    ring: Zn,
    zn_base: PhantomData<ZnBase>,
    base_eval_low: BaseEvalLow,
    base_evals_high: [BaseEvalHigh; m]
}

impl<Zn, ZnBase, BaseEvalLow, BaseEvalHigh, const m: usize> Zp2Evaluator<Zn, ZnBase, BaseEvalLow, BaseEvalHigh, m>
    where Zn: ZnRingStore,
        ZnBase: ZnRingStore,
        Zn::Type: ZnRing,
        ZnBase::Type: ZnRing,
        BaseEvalLow: ZnEvaluator<ZnBase, m>,
        BaseEvalHigh: ZnEvaluator<ZnBase, m>
{
    pub fn init(ring: Zn, poly_count: usize, base_eval_low: BaseEvalLow, base_evals_high: [BaseEvalHigh; m]) -> Self {
        let ZZ = ring.integer_ring();
        let (p, e) = algorithms::int_factor::is_prime_power(ZZ, ring.modulus()).unwrap();
        assert_eq!(2, e);
        assert_el_eq!(ZZ, ring.modulus(), &int_cast(base_eval_low.ring().integer_ring().clone_el(base_eval_low.ring().modulus()), ZZ, base_eval_low.ring().integer_ring()));
        assert_eq!(poly_count, base_eval_low.poly_count());
        for i in 0..m {
            assert!(base_evals_high[i].ring().get_ring() == base_evals_high[0].ring().get_ring());
            assert_el_eq!(ZZ, &p, &int_cast(base_evals_high[i].ring().integer_ring().clone_el(base_evals_high[i].ring().modulus()), ZZ, base_evals_high[i].ring().integer_ring()));
            assert_eq!(poly_count, base_evals_high[i].poly_count());
        }
        Self {
            ring: ring,
            base_eval_low: base_eval_low,
            base_evals_high: base_evals_high,
            zn_base: PhantomData
        }
    }

    pub fn low_eval(&self) -> &BaseEvalLow {
        &self.base_eval_low
    }

    pub fn high_evals(&self) -> &[BaseEvalHigh] {
        &self.base_evals_high
    }
}

impl<Zn, ZnBase, BaseEvalLow, BaseEvalHigh, const m: usize> ZnEvaluator<Zn, m> for Zp2Evaluator<Zn, ZnBase, BaseEvalLow, BaseEvalHigh, m>
    where Zn: ZnRingStore,
        ZnBase: ZnRingStore,
        Zn::Type: ZnRing + PreparedDivisibilityRing,
        ZnBase::Type: ZnRing,
        BaseEvalLow: ZnEvaluator<ZnBase, m>,
        BaseEvalHigh: ZnEvaluator<ZnBase, m>
{
    fn poly_count(&self) -> usize {
        self.base_eval_low.poly_count()
    }

    fn ring(&self) -> &Zn {
        &self.ring
    }

    fn evaluate_many<Index, Points, Callback>(&self, points: Points, callback: Callback, test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<Zn>; m])>,
            Index: std::fmt::Debug + Hash + Eq + Ord + Copy,
            Callback: ZnEvaluatorCallback<Zn, Index>
    {
        let result = RefCell::new((0..(points.len() * self.poly_count())).map(|_| self.ring().zero()).collect::<Vec<_>>());

        let ZZ = self.ring().integer_ring();
        let Zp = self.base_evals_high[0].ring();
        let reductions: (ReductionMap<&Zn, &ZnBase>, ReductionMap<&Zn, &ZnBase>) = (
            ReductionMap::new(self.ring(), self.base_eval_low.ring()).unwrap(),
            ReductionMap::new(self.ring(), Zp).unwrap()
        );

        struct EnterResultCallback<'a, Zn, BaseZn, V> 
            where Zn: ZnRingStore,
                Zn::Type: ZnRing,
                BaseZn: ZnRingStore,
                BaseZn::Type: ZnRing,
                V: VectorFn<El<Zn>>
        {
            poly_count: usize,
            result: &'a RefCell<Vec<El<Zn>>>,
            red: &'a ReductionMap<&'a Zn, &'a BaseZn>,
            factors: V
        }

        impl<'a, Zn, BaseZn, V> ZnEvaluatorCallback<BaseZn, usize> for EnterResultCallback<'a, Zn, BaseZn, V>
            where Zn: ZnRingStore,
                Zn::Type: ZnRing,
                BaseZn: ZnRingStore,
                BaseZn::Type: ZnRing,
                V: VectorFn<El<Zn>>
        {
            #[inline(never)]
            fn call<I>(&self, j: usize, ys: I, _ring: &BaseZn)
                where I: ExactSizeIterator<Item = El<BaseZn>>
            {
                let factor = self.factors.at(j);
                let mut result = self.result.borrow_mut();
                for (i, y) in ys.enumerate() {
                    self.red.domain().add_assign(
                        &mut result[j * self.poly_count + i], 
                        self.red.domain().mul_ref_snd(self.red.smallest_lift(y), &factor)
                    );
                }
            }
        }
        
        let modulus_quo = ZZ.checked_div(
            self.ring().modulus(),
            &int_cast(
                self.base_evals_high[0].ring().integer_ring().clone_el(self.base_evals_high[0].ring().modulus()), 
                ZZ, 
                self.base_evals_high[0].ring().integer_ring()
            )
        ).unwrap();
        let mut modulus_quo_half = ZZ.clone_el(&modulus_quo);
        ZZ.euclidean_div_pow_2(&mut modulus_quo_half, 1);

        let split_high_low = |x: &El<Zn>| {
            let Zp_rem = reductions.1.map_ref(x);
            let Zp2_rem = reductions.1.smallest_lift(Zp_rem);
            assert!(ZZ.is_leq(&ZZ.abs(self.ring().smallest_lift(self.ring().clone_el(&Zp2_rem))), &modulus_quo_half));
            (self.ring().sub_ref_snd(self.ring().clone_el(x), &Zp2_rem), Zp2_rem)
        };

        let points: Vec<(Index, [(El<Zn>, El<Zn>); m])> = (0..points.len()).map(|i| {
            let (idx, point) = points.at(i);
            (idx, std::array::from_fn(|j| split_high_low(&point[j])))
        }).collect::<Vec<_>>();

        for k in 0..m {
            self.base_evals_high[k].evaluate_many::<usize, _, _>(
                (0..points.len()).into_fn().map(|j| (j, std::array::from_fn::<_, m, _>(|i| 
                    reductions.1.map(self.ring().clone_el(&points[j].1[i].1))
                ))),
                EnterResultCallback {
                    poly_count: self.poly_count(),
                    result: &result,
                    red: &reductions.1,
                    factors: (0..points.len()).into_fn()
                        .map(|i| self.ring().clone_el(&points[i].1[k].0))
                },
                &test_polynomial_tracker.derived(k)
            )
        }

        self.base_eval_low.evaluate_many::<usize, _, _>(
            (0..points.len()).into_fn().map(|j| (j, std::array::from_fn::<_, m, _>(|i| 
                reductions.0.map(self.ring().clone_el(&points[j].1[i].1))
            ))),
            EnterResultCallback {
                poly_count: self.poly_count(),
                result: &result,
                red: &reductions.0,
                factors: (0..points.len()).into_fn().map(|_| self.ring().one())
            },
            test_polynomial_tracker
        );

        let mut result = result.into_inner();
        for i in 0..points.len() {
            callback.call(points[i].0, result.drain(0..self.poly_count()), self.ring());
        }
    }
}

#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use crate::strategy::PolynomialStats;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::Zn;
#[cfg(test)]
use feanor_math::iters::multi_cartesian_product;
#[cfg(test)]
use feanor_math::rings::finite::FiniteRingStore;

#[cfg(test)]
const TEST_M: usize = 4;

#[cfg(test)]
fn test_poly() -> (MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, TEST_M>, El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, TEST_M>>, PolynomialStats) {
    let ZZX = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    // 1 + 2x^2 + y + xy + x^2y + 2y^2 + 2y^3
    let f = ZZX.from_terms([
        (1, Monomial::new([0, 0, 0, 0])),
        (2, Monomial::new([0, 0, 2, 0])),
        (1, Monomial::new([0, 0, 0, 1])),
        (1, Monomial::new([0, 0, 1, 1])),
        (1, Monomial::new([0, 0, 2, 1])),
        (2, Monomial::new([0, 0, 0, 2])),
        (2, Monomial::new([0, 0, 0, 3]))
    ].into_iter());
    let poly_params = PolynomialStats {
        D: 3,
        m: TEST_M,
        log_coefficient_size: 1,
        monomials: 7
    };
    return (ZZX, f, poly_params);
}
#[test]
fn test_zp2_evaluator() {
    let (ZZX, f, _poly_params) = test_poly();
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&ZZX, &[ZZX.clone_el(&f)]);
    let Zp2 = Zn::new(17 * 17);
    let Zp = Zn::new(17);
    
    let part_polys_high: [_; TEST_M] = std::array::from_fn(|i| [
        ZZX.from_terms(ZZX.terms(&f).map(|(c, mon)| (
            *c * mon[i] as i64, 
            Monomial::new(std::array::from_fn(|k| if k == i { mon[k].saturating_sub(1) } else { mon[k] }))
        )))
    ]);
    let part_poly_low = [ZZX.clone_el(&f)];
    let actual = Zp2Evaluator::init(
        &Zp2, 
        1,
        TestEvaluator::new(&Zp2, &ZZX, &part_poly_low),
        std::array::from_fn(|i| TestEvaluator::new(&Zp, &ZZX, &part_polys_high[i]))
    );
    let expected = TestEvaluator::new(&Zp2, &ZZX, &part_poly_low);

    for (x, y) in multi_cartesian_product((0..2).map(|_| Zp2.elements()), |slice| (slice[0], slice[1]), |_, x| *x) {
        assert_el_eq!(&Zp2, &expected.evaluate_sync([Zp2.zero(), Zp2.zero(), x, y], &test_polynomial_tracker)[0], &actual.evaluate_sync([Zp2.zero(), Zp2.zero(), x, y], &test_polynomial_tracker)[0]);
    }
}