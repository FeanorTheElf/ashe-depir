use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::*;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::vector::vec_fn::IntoVectorFn;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::vector::VectorView;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::homomorphism::*;
use feanor_math::divisibility::DivisibilityRingStore;
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use selfref::opaque;
use selfref::Holder;

use crate::ashe::CiphertextRing;
use crate::{m, reduced_m, component_count};
use crate::strategy::sample_primes_arithmetic_progression;
use crate::strategy::InputStats;
use crate::strategy::PolynomialStats;
use crate::strategy::ModulusInfo;
use crate::zn_eval::*;
use zp2_evaluator::Zp2Evaluator;

use std::collections::HashMap;
use std::hash::Hash;
use std::io::Write;
use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::atomic::{AtomicI64, AtomicU64};

// pub mod l1;
// pub mod l2;
pub mod l2_zp2;

pub type RamPrime = Zn;
pub type DiskPrime = Zn;
pub type Level2Prime = Zn;
pub type TopLevelPrime = Zn;

pub type Level2Evaluator<'this, E0, E1> = crt_evaluator::CRTEvaluator<'this, Level2Prime, RamPrime, DiskPrime, Zn, E0, E1, !, reduced_m>;
pub type TopLevelEvaluator<'this, E0, E1, E2> = crt_evaluator::CRTEvaluator<'this, TopLevelPrime, RamPrime, DiskPrime, Level2Prime, E0, E1, E2, reduced_m>;

struct RNSComponentEvaluator<'this, E0, E1>
    where E0: ZnEvaluator<RamPrime, reduced_m>,
        E1: ZnEvaluator<DiskPrime, reduced_m>
{
    primes0: Vec<RamPrime>,
    primes1: Vec<DiskPrime>,
    primes2: Vec<Level2Prime>,
    primes3: Vec<TopLevelPrime>,
    level0: Vec<E0>,
    level1: Vec<E1>,
    level2: Vec<Level2Evaluator<'this, E0, E1>>,
    level3: Vec<TopLevelEvaluator<'this, E0, E1, Level2Evaluator<'this, E0, E1>>>
}

impl<'this, E0, E1> RNSComponentEvaluator<'this, E0, E1>
    where E0: ZnEvaluator<RamPrime, reduced_m>,
        E1: ZnEvaluator<DiskPrime, reduced_m>
{
    fn init_level_2<'a>(&'this self, poly_params: &'a [PolynomialStats], level2_plan: &HashMap<i64, ModulusInfo>) {
        for (eval, Fp) in self.level2.iter().zip(self.primes2.iter()) {
            let input_params = InputStats { characteristic: *Fp.modulus() as i64, input_size_bound: ((Fp.modulus() - 1) / 2 + 1) as usize };
            eval.init(*Fp, &poly_params[..=level2_plan.get(&(*Fp.modulus() as i64)).unwrap().max_index_polynomial_part], input_params, (&self.primes0, &self.primes1, &[]), (&self.level0, &self.level1, &[]));
        }
    }

    fn init_level_3<'a>(&'this self, poly_params: &'a [PolynomialStats]) {
        for (eval, Fp) in self.level3.iter().zip(self.primes3.iter()) {
            let input_params = InputStats { characteristic: *Fp.modulus(), input_size_bound: ((Fp.modulus() - 1) / 2 + 1) as usize };
            eval.init(*Fp, poly_params, input_params, (&self.primes0, &self.primes1, &self.primes2), (&self.level0, &self.level1, &self.level2));
        }
    }

    fn evaluate_many<Index, Points, Callback>(&self, rns_component: usize, entries: Points, callback: Callback, test_polynomial_tracker: &TestPolynomialTracker<reduced_m>)
        where Points: VectorFn<(Index, [El<Zn>; reduced_m])>,
            Index: 'static + std::fmt::Debug + Hash + Eq + Ord + Copy,
            Callback: ZnEvaluatorCallback<Zn, Index>
    {
        self.level3[rns_component].evaluate_many::<Index, _, _>(entries, callback, test_polynomial_tracker)
    }
}

struct CiphertextRingEvaluatorStructKey<E0, E1>
    where E0: ZnEvaluator<RamPrime, reduced_m>,
        E1: ZnEvaluator<DiskPrime, reduced_m>
{
    e0: PhantomData<E0>,
    e1: PhantomData<E1>
}

opaque! {
    impl[E0, E1] Opaque for CiphertextRingEvaluatorStructKey<E0, E1> {
        type Kind<'this> = RNSComponentEvaluator<'this, E0, E1>;
    } where E0: ZnEvaluator<RamPrime, reduced_m>,
        E1: ZnEvaluator<DiskPrime, reduced_m>
}

pub struct ASHEEvaluator<'a, E0, E1>
    where E0: 'a + ZnEvaluator<RamPrime, reduced_m>,
        E1: 'a  + ZnEvaluator<DiskPrime, reduced_m>
{
    d: usize,
    evaluators: Vec<Pin<Box<Holder<'a, CiphertextRingEvaluatorStructKey<E0, E1>>>>>,
    test_polynomial_tracker: Vec<TestPolynomialTracker<reduced_m>>
}

fn available_primes() -> impl Clone + Iterator<Item = i64> {
    sample_primes_arithmetic_progression(0, 1, 2)
}

pub type RamDiskASHEEvaluator<'a> = ASHEEvaluator<'a, read_ram::ReadRamEvaluator<RamPrime, reduced_m>, read_disk::ReadDiskEvaluator<DiskPrime, reduced_m>>;

pub type RamDiskZp2ASHEEvaluator<'a> = ASHEEvaluator<'a, 
    Zp2Evaluator<RamPrime, RamPrime, read_ram::ReadRamEvaluator<RamPrime, reduced_m>, read_ram::ReadRamEvaluator<RamPrime, reduced_m>, reduced_m>, 
    Zp2Evaluator<DiskPrime, DiskPrime, read_disk::ReadDiskEvaluator<DiskPrime, reduced_m>, read_disk::ReadDiskEvaluator<DiskPrime, reduced_m>, reduced_m>
>;

impl<'a, E0, E1> ASHEEvaluator<'a, E0, E1> 
    where E0: 'a + Sync + ZnEvaluator<RamPrime, reduced_m>,
        E1: 'a + Sync + ZnEvaluator<DiskPrime, reduced_m>
{
    ///
    /// Computes `f(b1 + a1 Y, ..., bm + am Y)`.
    /// 
    /// Here the `bi` must have `{0, 1}`-CRT components, and are given as an
    /// iterator over these components, in the order corresponding to 
    /// [`he_ring::doublerns::double_rns_ring::DoubleRNSRingBase::fourier_coefficient()`]
    /// when `(i, j)` runs through all possible tuples in lex order, i.e. 
    /// `(0, 0), (0, 1), (0, 2), ..., (1, 0), ...`.
    /// 
    /// Furthermore, we require that `am` is invertible.
    /// 
    pub fn evaluate<I>(&self, C: &CiphertextRing, a_els: [&El<CiphertextRing>; m], mut b_components_its: [I; m]) -> Vec<El<CiphertextRing>> 
        where I: Iterator<Item = u8>
    {
        let rns_component_it = C.get_ring().rns_base().iter().enumerate().flat_map(|(i, Zp)| (0..C.rank()).map(move |j| (i, j, Zp)));

        let a_els_ref = &a_els;
        let a_points: Vec<(_, [_; reduced_m])> = rns_component_it.clone().map(|(i, j, Fp)| {
            let a_last = *C.get_ring().fourier_coefficient(i, j, a_els_ref.at(m - 1));
            (
                a_last,
                std::array::from_fn(|k| Fp.checked_div(C.get_ring().fourier_coefficient(i, j, a_els_ref.at(k)), &a_last)
                    .expect("Last input ciphertext does not have an invertible a."))
            )
        }).collect::<Vec<_>>();
        let a_points_ref = &a_points;

        let b_points: Vec<usize> = rns_component_it.clone().map(|(_i, _j, _)| (0..m).map(|k| {
            let next_component = b_components_its[k].next().unwrap() as usize;
            assert!(next_component == 0 || next_component == 1, "Input ciphertext does not have a {{0, 1}}-CRT element for b.");
            next_component * (1 << k)
        }).sum::<usize>()).collect::<Vec<_>>();
        let b_points_ref = &b_points;

        let result_rns_components = (0..(self.d + 1)).map(|_| rns_component_it.clone().map(|_| AtomicI64::new(0)).collect::<Vec<_>>()).collect::<Vec<_>>();
        let result_rns_components_ref = &result_rns_components;

        struct EnterResultCallback<'a>(&'a [Vec<AtomicI64>]);

        impl<'a> ZnEvaluatorCallback<TopLevelPrime, usize> for EnterResultCallback<'a> {

            fn call<I>(&self, idx: usize, ys: I, ring: &TopLevelPrime)
                where I: ExactSizeIterator<Item = El<TopLevelPrime>> 
            {
                for (i, y) in ys.enumerate() {
                    debug_assert!(self.0[i][idx].load(std::sync::atomic::Ordering::Acquire) == 0);
                    self.0[i][idx].store(ring.smallest_positive_lift(y), std::sync::atomic::Ordering::Release);
                }
            }
        }

        // now group the RNS components by same modulus and same a;
        // we pass on every group to the suitable asynchronous base evaluator

        #[cfg(not(feature = "disable_parallel"))]
        let task_iterator = (0..component_count).into_par_iter().flat_map(
            |b_index| (0..C.get_ring().rns_base().len()).into_par_iter().map(move |rns_component_i| (b_index, rns_component_i))
        );
        #[cfg(feature = "disable_parallel")]
        let task_iterator = (0..component_count).into_iter().flat_map(
            |b_index| (0..C.get_ring().rns_base().len()).into_iter().map(move |rns_component_i| (b_index, rns_component_i))
        );

        let progress = AtomicU64::new(0);
        
        print!("Evaluation complete 0%");
        std::io::stdout().flush().unwrap();
        task_iterator.for_each(|(b_index, rns_component_i)| {
            let indices = (0..C.rank()).map(|j| rns_component_i * C.rank() + j).filter(|idx| b_points_ref[*idx] == b_index).rev().collect::<Vec<_>>();
            let indices_len = indices.len();
            self.evaluators[b_index].as_ref().operate_in(|evals| evals.evaluate_many::<usize, _, _>(
                rns_component_i,
                indices.into_fn().map(|idx| (idx, a_points_ref[idx].1)),
                EnterResultCallback(&result_rns_components_ref),
                &self.test_polynomial_tracker[b_index]
            ));
            let current = progress.fetch_add(indices_len as u64, std::sync::atomic::Ordering::Relaxed);
            print!("\rEvaluation complete {}%  ", (1000. * current as f64 / C.rank() as f64 / C.get_ring().rns_base().len() as f64).round() / 10.);
            std::io::stdout().flush().unwrap();
        });
        println!();
        std::io::stdout().flush().unwrap();

        // put the result back into ring elements
        let mut result = (0..(self.d + 1)).map(|_| C.zero()).collect::<Vec<_>>();
        for (i, Zp) in C.get_ring().rns_base().iter().enumerate() {
            let mod_p = Zp.can_hom(&StaticRing::<i64>::RING).unwrap();
            for j in 0..C.rank() {
                for k in 0..=self.d {
                    *C.get_ring().fourier_coefficient_mut(i, j, &mut result[k]) = mod_p.map(result_rns_components[k][i * C.rank() + j].load(std::sync::atomic::Ordering::Acquire));
                    Zp.mul_assign(C.get_ring().fourier_coefficient_mut(i, j, &mut result[k]), Zp.pow(a_points_ref[i * C.rank() + j].0, self.d - k));
                }
            }
        }

        // we did the polynomials in reverse order, as this means we have degrees d, d - 1, ..., 1, 0
        result.reverse();
        return result;
    }
}

#[cfg(test)]
pub type TestASHEEvaluator<'a> = ASHEEvaluator<'a, TestEvaluator<'a, RamPrime, m>, !>;

#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::mempool::DefaultMemoryProvider;
#[cfg(test)]
use feanor_math::rings::multivariate::ordered::MultivariatePolyRingImpl;
#[cfg(test)]
use feanor_math::rings::multivariate::*;
#[cfg(test)]
use feanor_math::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use feanor_math::rings::poly::PolyRingStore;

#[cfg(test)]
fn test_poly() -> (
    usize,
    MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>, 
    El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>, 
    Vec<Vec<El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, m>>>>
) {

    let P = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    let PY = DensePolyRing::new(MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!()), "Y");

    let d = 3;
    // z^2w - 8y^2 + xz + 2yw + x - w
    let original_poly = P.from_terms([
        (1, Monomial::new([1, 0, 0, 0])),
        (-1, Monomial::new([0, 0, 0, 1])),
        (1, Monomial::new([1, 0, 1, 0])),
        (2, Monomial::new([0, 1, 0, 1])),
        (-8, Monomial::new([0, 2, 0, 0])),
        (1, Monomial::new([0, 0, 2, 1])),
    ].into_iter());

    let processed_polys = (0..component_count).map(|b_index| {
        let b = std::array::from_fn::<_, m, _>(|j| ((b_index >> j) & 1) as i32);
        let eval_point = std::array::from_fn::<_, m, _>(|j| PY.from_terms([
            (P.int_hom().map(b[j]), 0),
            (P.indeterminate(j), 1)
        ].into_iter()));
        let mut result = P.evaluate(&original_poly, eval_point, &PY.inclusion().compose(P.inclusion()));
        assert!(result.len() <= d + 1 || result[(d + 1)..].iter().all(|f| P.is_zero(f)));
        result.truncate(d + 1);
        result.extend((0..(d + 1 - result.len())).map(|_| P.zero()));
        
        // to have decreasing degrees, we reverse the polynomials here
        result.reverse();
        return result;
    }).collect::<Vec<_>>();

    return (d, P, original_poly, processed_polys);
}
