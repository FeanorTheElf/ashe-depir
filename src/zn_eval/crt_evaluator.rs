use std::cell::RefCell;
use std::hash::Hash;
use std::marker::PhantomData;
use std::cmp::{min, max};
use std::sync::OnceLock;

use feanor_math::homomorphism::CanHomFrom;
use feanor_math::integer::{int_cast, BigIntRingBase};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::ring::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};
use feanor_math::assert_el_eq;

use crate::strategy::{reduction_factors, InputStats, PolynomialStats};

use super::prime_components::*;
use super::*;

pub struct CRTEvaluator<'evaluator_table, F, T1, T2, T3, E1, E2, E3, const m: usize>
    where F: Clone + ZnRingStore,
        T1: ZnRingStore,
        T2: ZnRingStore,
        T3: ZnRingStore,
        T1::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T2::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T3::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        E1: ZnEvaluator<T1, m>,
        E2: ZnEvaluator<T2, m>,
        E3: ZnEvaluator<T3, m>
{
    reduction_ring0: PhantomData<T1>,
    reduction_ring1: PhantomData<T2>,
    reduction_ring2: PhantomData<T3>,
    evaluators0: OnceLock<&'evaluator_table [E1]>,
    evaluators1: OnceLock<&'evaluator_table [E2]>,
    evaluators2: OnceLock<&'evaluator_table [E3]>,
    prime_decompositions: OnceLock<Vec<PrimeDecomposition<'evaluator_table, F, T1, T2, T3, StaticRing<i64>>>>,
    main_prime_decomposition: OnceLock<PrimeDecomposition<'evaluator_table, F, T1, T2, T3, StaticRing<i64>>>,
    poly_count: OnceLock<usize>,

    #[cfg(test)] pub polys: OnceLock<Vec<El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>>>,
    #[cfg(test)] pub poly_ring: OnceLock<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>
}

impl<'evaluator_table, F, T1, T2, T3, E1, E2, E3, const m: usize> CRTEvaluator<'evaluator_table, F, T1, T2, T3, E1, E2, E3, m>
    where F: Clone + ZnRingStore,
        T1: ZnRingStore,
        T2: ZnRingStore,
        T3: ZnRingStore,
        T1::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T2::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T3::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        E1: ZnEvaluator<T1, m>,
        E2: ZnEvaluator<T2, m>,
        E3: ZnEvaluator<T3, m>
{
    pub fn uninitialized() -> Self {
        Self {
            reduction_ring0: PhantomData,
            reduction_ring1: PhantomData,
            reduction_ring2: PhantomData,
            evaluators0: OnceLock::new(),
            evaluators1: OnceLock::new(),
            evaluators2: OnceLock::new(),
            prime_decompositions: OnceLock::new(),
            main_prime_decomposition: OnceLock::new(),
            poly_count: OnceLock::new(),
            #[cfg(test)] polys: OnceLock::new(),
            #[cfg(test)] poly_ring: OnceLock::new()
        }
    }

    pub fn init(&self, R: F, poly_params: &[PolynomialStats], input_params: InputStats, primes: (&'evaluator_table [T1], &'evaluator_table [T2], &'evaluator_table [T3]), evaluators: (&'evaluator_table [E1], &'evaluator_table [E2], &'evaluator_table [E3])) {
        assert_eq!(primes.0.len(), evaluators.0.len());
        assert_eq!(primes.1.len(), evaluators.1.len());
        assert_eq!(primes.2.len(), evaluators.2.len());
        for i in 0..primes.0.len() {
            assert!(primes.0[i].get_ring() == evaluators.0[i].ring().get_ring());
        }
        for i in 0..primes.1.len() {
            assert!(primes.1[i].get_ring() == evaluators.1[i].ring().get_ring());
        }
        for i in 0..primes.2.len() {
            assert!(primes.2[i].get_ring() == evaluators.2[i].ring().get_ring());
        }

        let reduction_primes = evaluators.0.iter().map(
            |base_eval| int_cast(base_eval.ring().integer_ring().clone_el(base_eval.ring().modulus()), &StaticRing::<i64>::RING, base_eval.ring().integer_ring())
        ).chain(
            evaluators.1.iter().map(|base_eval| int_cast(base_eval.ring().integer_ring().clone_el(base_eval.ring().modulus()), &StaticRing::<i64>::RING, base_eval.ring().integer_ring()))
        ).chain(
            evaluators.2.iter().map(|base_eval| int_cast(base_eval.ring().integer_ring().clone_el(base_eval.ring().modulus()), &StaticRing::<i64>::RING, base_eval.ring().integer_ring()))
        );
        let prime_decompositions = poly_params.iter().enumerate().map(|(i, poly_param)| {
            let decomposition_primes = reduction_factors(
                *poly_param, 
                input_params.input_size_bound, 
                reduction_primes.clone()
            );
            let mut max_prime = 0;
            let mut decomposition_prime_count = 0;
            for p in decomposition_primes {
                max_prime = max(p, max_prime);
                decomposition_prime_count += 1;
            }
            let decomposition_prime_count0 = min(evaluators.0.len(), decomposition_prime_count);
            let decomposition_prime_count1 = min(evaluators.1.len(), decomposition_prime_count - decomposition_prime_count0);
            let decomposition_prime_count2 = min(evaluators.2.len(), decomposition_prime_count - decomposition_prime_count0 - decomposition_prime_count1);
            let gamma = decomposition_prime_count as i32 * max_prime as i32;

            let l0_evaluator_count = evaluators.0.iter().filter(|eval| eval.poly_count() > i).count();
            assert!(l0_evaluator_count >= decomposition_prime_count0, "Found only {} evaluators that provide results for polynomial {}, expected {} (when reducing {} to level 0)", l0_evaluator_count, i, decomposition_prime_count0, R.integer_ring().format(R.modulus()));
            
            let l1_evaluator_count = evaluators.1.iter().filter(|eval| eval.poly_count() > i).count();
            assert!(l1_evaluator_count >= decomposition_prime_count1, "Found only {} evaluators that provide results for polynomial {}, expected {} (when reducing {} to level 1)", l1_evaluator_count, i, decomposition_prime_count1, R.integer_ring().format(R.modulus()));
            
            let l2_evaluator_count = evaluators.2.iter().filter(|eval| eval.poly_count() > i).count();
            assert!(l2_evaluator_count >= decomposition_prime_count2, "Found only {} evaluators that provide results for polynomial {}, expected {} (when reducing {} to level 2)", l2_evaluator_count, i, decomposition_prime_count2, R.integer_ring().format(R.modulus()));

            return PrimeDecomposition::new(
                R.clone(), 
                (&primes.0[..decomposition_prime_count0], &primes.1[..decomposition_prime_count1], &primes.2[..decomposition_prime_count2]), 
                StaticRing::<i64>::RING, 
                gamma as i64
            );
        }).collect::<Vec<_>>();
        
        let main_prime_decomposition = prime_decompositions.iter().max_by_key(|decomp| decomp.total_len()).unwrap();
        for i in 1..prime_decompositions.len() {
            assert!(prime_decompositions[i].len0() <= main_prime_decomposition.len0());
            assert!(prime_decompositions[i].len1() <= main_prime_decomposition.len1());
            assert!(prime_decompositions[i].len2() <= main_prime_decomposition.len2());
        }

        self.poly_count.set(poly_params.len()).unwrap();
        self.evaluators0.set(evaluators.0).ok().unwrap();
        self.evaluators1.set(evaluators.1).ok().unwrap();
        self.evaluators2.set(evaluators.2).ok().unwrap();
        self.main_prime_decomposition.set(main_prime_decomposition.clone()).ok().unwrap();
        self.prime_decompositions.set(prime_decompositions).ok().unwrap();
    }

    fn run_for_base_evaluators<Points, Index, T, E, F1, F2>(entries: Points, evaluators: &[E], result_indices: &RefCell<Vec<Option<Index>>>, decompose: F1, compose: F2, _from: &F, test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<F>; m])>,
            Index: 'static + std::fmt::Debug + Hash + Eq + Ord + Copy,
            T: ZnRingStore,
            T::Type: ZnRing + CanHomFrom<BigIntRingBase>,
            E: ZnEvaluator<T, m>,
            F1: Fn(usize, El<F>, usize) -> El<T>,
            F2: Fn(usize, usize, El<T>, usize)
    {
        struct Callback<'a, F, Index, T, F2>
            where F: ZnRingStore,
                F::Type: ZnRing,
                Index: Hash + Eq + Ord + Copy,
                T: ZnRingStore,
                T::Type: ZnRing + CanHomFrom<BigIntRingBase>,
                F2: Fn(usize, usize, El<T>, usize)
        {
            from: PhantomData<&'a F>,
            to: PhantomData<&'a T>,
            prime_component_index: usize,
            result_indices: &'a RefCell<Vec<Option<Index>>>,
            compose: &'a F2
        }

        impl<'a, F, Index, T, F2> ZnEvaluatorCallback<T, (usize, Index)> for Callback<'a, F, Index, T, F2>
            where F: ZnRingStore,
                F::Type: ZnRing,
                Index: std::fmt::Debug + Hash + Eq + Ord + Copy,
                T: ZnRingStore,
                T::Type: ZnRing + CanHomFrom<BigIntRingBase>,
                F2: Fn(usize, usize, El<T>, usize)
        {
            #[inline(never)]
            fn call<I>(&self, (j, idx): (usize, Index), ys: I, _: &T)
                where I: ExactSizeIterator<Item = El<T>>
            {
                self.result_indices.borrow_mut()[j] = Some(idx);
                for (k, y) in ys.enumerate() {
                    (self.compose)(j, k, y, self.prime_component_index);
                }
            }
        }

        // let mut random_indices = (0..evaluators.len()).collect::<Vec<_>>();
        // random_indices.shuffle(&mut thread_rng());

        for i in 0..evaluators.len() {
            evaluators[i].evaluate_many::<(usize, Index), _, _>(
                (0..entries.len()).into_fn().map(|j| {
                    let mut point_coordinates = entries.at(j).1.into_iter();
                    ((j, entries.at(j).0), std::array::from_fn::<_, m, _>(|_| decompose(j, point_coordinates.next().unwrap(), i)))
                }),
                Callback {
                    compose: &compose,
                    from: PhantomData::<&F>,
                    to: PhantomData,
                    prime_component_index: i,
                    result_indices: result_indices
                },
                test_polynomial_tracker
            );
        }
    }

    pub fn used_base_evaluators(&self) -> (&[E1], &[E2], &[E3]) {
        (
            &self.evaluators0.get().unwrap()[..self.prime_decompositions.get().unwrap()[0].len0()],
            &self.evaluators1.get().unwrap()[..self.prime_decompositions.get().unwrap()[0].len1()],
            &self.evaluators2.get().unwrap()[..self.prime_decompositions.get().unwrap()[0].len2()]
        )
    }
}

impl<'evaluator_table, F, T1, T2, T3, E1, E2, E3, const m: usize> ZnEvaluator<F, m> for CRTEvaluator<'evaluator_table, F, T1, T2, T3, E1, E2, E3, m>
    where F: Clone + ZnRingStore,
        T1: ZnRingStore,
        T2: ZnRingStore,
        T3: ZnRingStore,
        T1::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T2::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        T3::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<BigIntRingBase> + CanHomFrom<StaticRingBase<i64>>,
        E1: ZnEvaluator<T1, m>,
        E2: ZnEvaluator<T2, m>,
        E3: ZnEvaluator<T3, m>
{
    fn ring(&self) -> &F {
        self.prime_decompositions.get().unwrap()[0].ring_from()
    }

    fn evaluate_many<Index, Points, Callback>(&self, entries: Points, callback: Callback, test_polynomial_tracker: &TestPolynomialTracker<m>)
        where Points: VectorFn<(Index, [El<F>; m])>,
            Index: 'static + std::fmt::Debug + Hash + Eq + Ord + Copy,
            Callback: ZnEvaluatorCallback<F, Index>
    {
        let poly_count = *self.poly_count.get().unwrap();
        let prime_decompositions = self.prime_decompositions.get().unwrap();
        // we rely on the fact that `decompose()` works the same for every prime decomposition, as it is
        // just `smallest_lift(x) mod p`
        let main_prime_decomposition = self.main_prime_decomposition.get().unwrap();
        let evaluators0 = self.evaluators0.get().unwrap();

        let result_indices: RefCell<Vec<Option<Index>>> = RefCell::new((0..entries.len()).map(|_| None).collect());
        let result_values: RefCell<Vec<PrimeComposer<_, _, _, _, _>>> = RefCell::new((0..entries.len()).flat_map(|_| (0..poly_count).map(|i| prime_decompositions[i].start_compose())).collect::<Vec<_>>());

        #[cfg(debug_assertions)] 
        let check_decompose_main_matches = |x: &El<F>, i: usize| {
            for k in 0..prime_decompositions.len() {
                if i < prime_decompositions[k].len0() {
                    assert_el_eq!(evaluators0[i].ring(), &main_prime_decomposition.direct_decompose0(self.ring().clone_el(&x), i), &prime_decompositions[k].direct_decompose0(self.ring().clone_el(&x), i));
                }
            }
        };
        #[cfg(not(debug_assertions))]
        let check_decompose_main_matches = |x: &El<F>, i: usize| {};
        
        Self::run_for_base_evaluators(
            &entries, &self.evaluators0.get().unwrap()[..main_prime_decomposition.len0()], 
            &result_indices, 
            |_j, x, i| {
                check_decompose_main_matches(&x, i);
                main_prime_decomposition.direct_decompose0(x, i)
            }, 
            |j, k, y, i| if k < poly_count { _ = result_values.borrow_mut()[j * poly_count + k].try_supply0(y, i) },
            self.ring(),
            test_polynomial_tracker
        );
        Self::run_for_base_evaluators(
            &entries, 
            &self.evaluators1.get().unwrap()[..main_prime_decomposition.len1()], 
            &result_indices, 
            |_j, x, i| main_prime_decomposition.direct_decompose1(x, i), 
            |j, k, y, i| if k < poly_count { _ = result_values.borrow_mut()[j * poly_count + k].try_supply1(y, i) },
            self.ring(),
            test_polynomial_tracker
        );
        Self::run_for_base_evaluators(
            &entries, 
            &self.evaluators2.get().unwrap()[..main_prime_decomposition.len2()], 
            &result_indices, 
            |_j, x, i| main_prime_decomposition.direct_decompose2(x, i), 
            |j, k, y, i| if k < poly_count { _ = result_values.borrow_mut()[j * poly_count + k].try_supply2(y, i) },
            self.ring(),
            test_polynomial_tracker
        );

        let mut result_values = result_values.into_inner();
        let mut result_indices = result_indices.into_inner();
        for (i, idx) in result_indices.drain(..).enumerate() {
            callback.call(idx.unwrap(), result_values.drain(0..poly_count).enumerate().map(|(j, y)| {
                let result = y.finish();
                test_polynomial_tracker.check_evaluation(&entries.at(i).1, self.ring(), &result, j);
                return result;
        }), self.ring());
        }
    }

    fn poly_count(&self) -> usize {
        *self.poly_count.get().unwrap()
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64;
#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::rings::multivariate::ordered::MultivariatePolyRingImpl;
#[cfg(test)]
use feanor_math::mempool::DefaultMemoryProvider;
#[cfg(test)]
use feanor_math::homomorphism::Homomorphism;
#[cfg(test)]
use super::TestEvaluator;
#[cfg(test)]
use super::test::*;

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
fn test_evaluate_one_level() {
    let GF = |p: i64| zn_64::Zn::new(p as u64);
    let (poly_ring, poly, poly_params) = test_poly();
    let polys = [poly];
    let expected_evaluator = TestEvaluator::new(GF(60013), &poly_ring, &polys);
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);

    let primes = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71].into_iter().map(GF).collect::<Vec<_>>();
    let base_evaluators = primes.iter().map(|Fp| TestEvaluator::new(*Fp, &poly_ring, &polys)).collect::<Vec<_>>();
    let empty_evaluators: Vec<TestEvaluator<zn_64::Zn, TEST_M>> = Vec::new();
    let empty_primes: Vec<zn_64::Zn> = Vec::new();
    let input_params = InputStats { characteristic: 60013, input_size_bound: 30007 };
    let actual_evaluator = CRTEvaluator::uninitialized();
    actual_evaluator.init(GF(60013), &[poly_params], input_params, (&primes[..], &empty_primes[..], &empty_primes[..]), (&base_evaluators[..], &empty_evaluators, &empty_evaluators));

    let i = GF(60013).into_int_hom();
    for (x, y) in [(0, 0), (1, 0), (0, 1), (0, 50000), (1, 50000), (0, 50001), (1, 50001), (50000, 50000), (50001, 50000), (50000, 50001), (50001, 50001)] {
        let point = [i.codomain().zero(), i.codomain().zero(), i.map(x), i.map(y)];
        assert_el_eq!(expected_evaluator.ring(), &expected_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0], &actual_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0]);
    }
}

#[test]
fn test_evaluate_two_levels() {
    let GF = |p: i64| zn_64::Zn::new(p as u64);
    let (poly_ring, poly, poly_params) = test_poly();
    let polys = [poly];
    let expected_evaluator = TestEvaluator::new(GF(1048583), &poly_ring, &polys);
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);

    let primes0 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53].into_iter().map(GF).collect::<Vec<_>>();
    let primes1 = [547, 557, 563].into_iter().map(GF).collect::<Vec<_>>();
    let base_evaluators0 = primes0.iter().map(|Fp| TestEvaluator::new(*Fp, &poly_ring, &polys)).collect::<Vec<_>>();
    let base_evaluators1 = primes1.iter().map(|Fp| TestEvaluator::new(*Fp, &poly_ring, &polys)).collect::<Vec<_>>();
    let empty_primes: Vec<zn_64::Zn> = Vec::new();
    let empty_evaluators: Vec<TestEvaluator<zn_64::Zn, TEST_M>> = Vec::new();
    let input_params = InputStats { characteristic: 1048583, input_size_bound: 524292 };
    let actual_evaluator = CRTEvaluator::uninitialized();
    actual_evaluator.init(GF(1048583), &[poly_params], input_params, (&primes0[..], &primes1[..], &empty_primes[..]), (&base_evaluators0[..], &base_evaluators1[..], &empty_evaluators));

    let i = GF(60013).into_int_hom();
    for (x, y) in [(0, 0), (1, 0), (0, 1), (0, 50000), (1, 500000), (0, 500001), (1, 500001), (500000, 500000), (500001, 500000), (500000, 500001), (500001, 500001)] {
        let point = [i.codomain().zero(), i.codomain().zero(), i.map(x), i.map(y)];
        assert_el_eq!(expected_evaluator.ring(), &expected_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0], &actual_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0]);
    }
}

#[test]
fn test_evaluate_multiple_polynomials() {
    let GF = |p: i64| zn_64::Zn::new(p as u64);
    let (poly_ring, f1, f1_params) = test_poly();
    let f2 = poly_ring.from_terms([(1, Monomial::new([0, 1, 0, 1]))].into_iter());
    let polys = [f1, f2];
    let f2_params = PolynomialStats { D: 2, m: TEST_M, monomials: 1, log_coefficient_size: 0 };
    let expected_evaluator = TestEvaluator::new(GF(65537), &poly_ring, &polys);
    let test_polynomial_tracker = TestPolynomialTracker::create_test(&poly_ring, &polys);

    let primes0 = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61].into_iter().map(GF).collect::<Vec<_>>();
    let primes1 = [97, 257].into_iter().map(GF).collect::<Vec<_>>();
    let base_evaluators0 = primes0.iter().map(|Fp| TestEvaluator::new(*Fp, &poly_ring, &polys)).collect::<Vec<_>>();
    let base_evaluators1 = primes1.iter().map(|Fp| TestEvaluator::new(*Fp, &poly_ring, &polys)).collect::<Vec<_>>();
    let empty_primes: Vec<zn_64::Zn> = Vec::new();
    let empty_evaluators: Vec<TestEvaluator<zn_64::Zn, TEST_M>> = Vec::new();
    let input_params = InputStats { characteristic: 65537, input_size_bound: 32768 };
    let actual_evaluator = CRTEvaluator::uninitialized();
    actual_evaluator.init(GF(65537), &[f1_params, f2_params], input_params, (&primes0[..], &primes1[..], &empty_primes[..]), (&base_evaluators0[..], &base_evaluators1[..], &empty_evaluators));

    let i = GF(65537).into_int_hom();
    for (z, x, y) in [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (32768, 1, 1), (1, 32768, 1), (32768, 0, 2)] {
        let point = [i.codomain().zero(), i.map(z), i.map(x), i.map(y)];
        assert_el_eq!(expected_evaluator.ring(), &expected_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0], &actual_evaluator.evaluate_sync(point, &test_polynomial_tracker)[0]);
        assert_el_eq!(expected_evaluator.ring(), &expected_evaluator.evaluate_sync(point, &test_polynomial_tracker)[1], &actual_evaluator.evaluate_sync(point, &test_polynomial_tracker)[1]);
        assert_eq!(2, expected_evaluator.evaluate_sync(point, &test_polynomial_tracker).len());
        assert_eq!(2, actual_evaluator.evaluate_sync(point, &test_polynomial_tracker).len());
    }
}