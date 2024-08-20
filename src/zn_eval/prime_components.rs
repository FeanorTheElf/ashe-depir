use std::marker::PhantomData;

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::*;
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::integer::*;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::zn::*;
use feanor_math::pid::EuclideanRingStore;
use feanor_math::vector::VectorView;
use feanor_math::ring::*;
use feanor_math::primitive_int::{StaticRing, StaticRingBase};

#[allow(non_upper_case_globals)]
const ZZbig: BigIntRing = BigIntRing::RING;

pub trait UsableZnRing<I: IntegerRingStore>: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>
    where I::Type: IntegerRing, Self::IntegerRingBase: CanIsoFromTo<I::Type> 
{}

impl<I: IntegerRingStore, T: ?Sized + ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>> UsableZnRing<I> for T
    where I::Type: IntegerRing, T::IntegerRingBase: CanIsoFromTo<I::Type> 
{}

///
/// A fast, random access implementation of the map
/// ```text
/// Z/qZ -> Z/NZ ~ Z/p1Z x ... x Z/pnZ,  a -> shortest_lift(a) mod N
/// ```
/// and the converse
/// ```text
/// Z/p1Z x ... x Z/pnZ ~ Z/NZ -> Z/qZ,  a -> shortest_lift(a) mod N
/// ```
///
/// # Definitions
/// 
/// Let original modulus be `q` and the target moduli `pi <= B`, with product `p` where `i < l`.
/// Further consider a correction size parameter `γ`.
/// 
/// # Requirements
/// 
/// To make the algorithm from [`PrimeDecomposition::compose()`] work, it is required that
/// the integer ring `I` can store values up to `B/2 * l * γ`.
/// 
pub struct PrimeDecomposition<'a, F: ZnRingStore, T1: ZnRingStore, T2: ZnRingStore, T3: ZnRingStore, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T1::Type: UsableZnRing<I>, <T1::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T2::Type: UsableZnRing<I>, <T2::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T3::Type: UsableZnRing<I>, <T3::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    from: F,
    to: (&'a [T1], &'a [T2], &'a [T3]),
    inv_unit_vectors: (Vec<El<T1>>, Vec<El<T2>>, Vec<El<T3>>),
    unit_vectors: (Vec<El<F>>, Vec<El<F>>, Vec<El<F>>),
    correction_unit_vectors: (Vec<El<I>>, Vec<El<I>>, Vec<El<I>>),
    
    #[allow(unused)]
    to_ring_type: (PhantomData<T1>, PhantomData<T2>, PhantomData<T3>),

    int_ring: I,
    gamma: El<I>,
    P_mod: El<F>
}

pub struct PrimeDecomposeVec<'a, F: ZnRingStore, T: ZnRingStore, I, const m: usize>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    lift: [El<I>; m],
    from: PhantomData<&'a F>,
    to: &'a [T],
    int_ring: &'a I,
    to_ring_type: PhantomData<T>,
}

impl<'a, F: ZnRingStore, T: ZnRingStore, I, const m: usize> Copy for PrimeDecomposeVec<'a, F, T, I, m>
    where I: IntegerRingStore,
        El<I>: Copy,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{}

impl<'a, F: ZnRingStore, T: ZnRingStore, I, const m: usize> Clone for PrimeDecomposeVec<'a, F, T, I, m>
    where I: IntegerRingStore,
        El<I>: Copy,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, F: ZnRingStore, T: ZnRingStore, I, const m: usize> VectorFn<[El<T>; m]> for PrimeDecomposeVec<'a, F, T, I, m>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    fn len(&self) -> usize {
        self.to.len()
    }

    fn at(&self, index: usize) -> [El<T>; m] {
        core::array::from_fn(|i| self.to.at(index).coerce(&self.int_ring, self.int_ring.clone_el(&self.lift[i])))
    }
}

impl<'a, F: Clone + ZnRingStore, T1: ZnRingStore, T2: ZnRingStore, T3: ZnRingStore, I> PrimeDecomposition<'a, F, T1, T2, T3, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T1::Type: UsableZnRing<I>, <T1::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T2::Type: UsableZnRing<I>, <T2::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T3::Type: UsableZnRing<I>, <T3::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    pub fn new(from: F, to: (&'a [T1], &'a [T2], &'a [T3]), int_ring: I, gamma: El<I>) -> Self {

        const ZZ: StaticRing::<i64> = StaticRing::<i64>::RING;
        let moduli = to.0.iter().map(|Fp| int_cast(Fp.integer_ring().clone_el(Fp.modulus()), ZZbig, Fp.integer_ring()))
            .chain(to.1.iter().map(|Fp| int_cast(Fp.integer_ring().clone_el(Fp.modulus()), ZZbig, Fp.integer_ring())))
            .chain(to.2.iter().map(|Fp| int_cast(Fp.integer_ring().clone_el(Fp.modulus()), ZZbig, Fp.integer_ring())));
        let l = int_cast(moduli.clone().count() as i64, ZZbig, ZZ);
        let B = moduli.clone().max_by(|a, b| ZZbig.cmp(a, b)).unwrap();

        // we need to be able to sum `l` values of size at most `γB/2`
        assert!(ZZbig.is_lt(&ZZbig.prod([
            ZZbig.clone_el(&l),
            ZZbig.clone_el(&B),
            int_cast(int_ring.clone_el(&gamma), ZZbig, &int_ring)
        ].into_iter()), &ZZbig.power_of_two(int_ring.get_ring().representable_bits().unwrap())));

        // the error made by the approximation is at most:
        // `l` times the error in `lift(x pi / Q) round(gamma/pi)`, which is `pi/2 * 1/2`;
        // thus the total error is at most `lB/4` and must be bounded by `γ/4`, so `γ >= lB`
        assert!(ZZbig.is_geq(&int_cast(int_ring.clone_el(&gamma), ZZbig, &int_ring), &ZZbig.prod([l, B].into_iter())));

        let mut result = PrimeDecomposition {
            P_mod: from.zero(),
            from: from.clone(),
            to: to,
            unit_vectors: (Vec::new(), Vec::new(), Vec::new()),
            inv_unit_vectors: (Vec::new(), Vec::new(), Vec::new()),
            correction_unit_vectors: (Vec::new(), Vec::new(), Vec::new()),
            to_ring_type: (PhantomData, PhantomData, PhantomData),
            int_ring: int_ring,
            gamma: gamma
        };

        let P = result.P();
        result.P_mod = from.coerce(&ZZbig, ZZbig.clone_el(&P));
        Self::initialize::<T1>(&result.from, result.to.0, &result.int_ring, &P, &result.gamma, &mut result.unit_vectors.0, &mut result.inv_unit_vectors.0, &mut result.correction_unit_vectors.0);
        Self::initialize::<T2>(&result.from, result.to.1, &result.int_ring, &P, &result.gamma, &mut result.unit_vectors.1, &mut result.inv_unit_vectors.1, &mut result.correction_unit_vectors.1);
        Self::initialize::<T3>(&result.from, result.to.2, &result.int_ring, &P, &result.gamma, &mut result.unit_vectors.2, &mut result.inv_unit_vectors.2, &mut result.correction_unit_vectors.2);
        return result;
    }

    fn P(&self) -> El<BigIntRing> {
        ZZbig.prod(
            Iterator::map(0..self.to.0.len(), |i| (0, i)).chain(
                    Iterator::map(0..self.to.1.len(), |i| (1, i))).chain(
                    Iterator::map(0..self.to.2.len(), |i| (2, i)))
                .map(|(group, i)| int_cast(self.p(group, i), &ZZbig, &self.int_ring))
        )
    }

    fn initialize<T: ZnRingStore>(from: &F, to: &[T], int_ring: &I, P: &El<BigIntRing>, gamma: &El<I>, unit_vectors: &mut Vec<El<F>>, inv_unit_vectors: &mut Vec<El<T>>, correction_unit_vectors: &mut Vec<El<I>>)
        where T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
    {
        for i in 0..to.len() {
            let p = int_cast(to.at(i).integer_ring().clone_el(to.at(i).modulus()), &int_ring, to.at(i).integer_ring());
            let P_over_pi = ZZbig.checked_div(&P, &int_cast(int_ring.clone_el(&p), &ZZbig, int_ring)).unwrap();

            unit_vectors.push(from.coerce(&ZZbig, ZZbig.clone_el(&P_over_pi)));
            inv_unit_vectors.push(to.at(i).invert(&to.at(i).coerce(&ZZbig, P_over_pi)).unwrap());
            correction_unit_vectors.push(int_ring.rounded_div(int_ring.clone_el(&gamma), &p));
        }
    }

    pub fn p(&self, group: usize, i: usize) -> El<I> {
        match group {
            0 => int_cast(self.to.0.at(i).integer_ring().clone_el(self.to.0.at(i).modulus()), &self.int_ring, self.to.0.at(i).integer_ring()),
            1 => int_cast(self.to.1.at(i).integer_ring().clone_el(self.to.1.at(i).modulus()), &self.int_ring, self.to.1.at(i).integer_ring()),
            2 => int_cast(self.to.2.at(i).integer_ring().clone_el(self.to.2.at(i).modulus()), &self.int_ring, self.to.2.at(i).integer_ring()),
            _ => panic!()
        }
    }

    fn B(&self) -> El<I> {
        let mut result = self.p(0, 0);
        for i in 1..self.to.0.len() {
            if self.int_ring.is_gt(&self.p(0, i), &result) {
                result = self.p(0, i);
            }
        }
        for i in 0..self.to.1.len() {
            if self.int_ring.is_gt(&self.p(1, i), &result) {
                result = self.p(1, i); 
           }
        }
        for i in 0..self.to.2.len() {
            if self.int_ring.is_gt(&self.p(2, i), &result) {
                result = self.p(2, i);
            }
        }
        return result;
    }

    pub fn len0(&self) -> usize {
        self.to.0.len()
    }
    
    pub fn len1(&self) -> usize {
        self.to.1.len()
    }

    pub fn len2(&self) -> usize {
        self.to.2.len()
    }

    pub fn total_len(&self) -> usize {
        self.to.0.len() + self.to.1.len() + self.to.2.len()
    }

    fn l(&self) -> El<I> {
        int_cast(self.total_len() as i64, &self.int_ring, &StaticRing::<i64>::RING)
    }

    #[allow(unused)]
    fn assumed_slack_factor(&self) -> f64 {
        self.int_ring.to_float_approx(&self.int_ring.mul(self.B(), self.l())) / 2. / self.int_ring.to_float_approx(&self.gamma)
    }

    fn direct_decompose<T: ZnRingStore>(&self, el: El<F>, i: usize, rings: &[T]) -> El<T>
        where T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
    {
        rings.at(i).coerce(&self.int_ring, int_cast(self.from.smallest_lift(el), &self.int_ring, self.from.integer_ring()))
    }

    pub fn direct_decompose0(&self, el: El<F>, i: usize) -> El<T1> {
        self.direct_decompose(el, i, &self.to.0)
    }

    pub fn direct_decompose1(&self, el: El<F>, i: usize) -> El<T2> {
        self.direct_decompose(el, i, &self.to.1)
    }

    pub fn direct_decompose2(&self, el: El<F>, i: usize) -> El<T3> {
        self.direct_decompose(el, i, &self.to.2)
    }

    pub fn start_compose<'b>(&'b self) -> PrimeComposer<'b, F, T1, T2, T3, I> {
        PrimeComposer {
            parent: self,
            current_correction: self.int_ring.zero(),
            current: self.from.zero(),
            supplied: 0
        }
    }

    pub fn ring_from(&self) -> &F {
        &self.from
    }
}

impl<'a, F: Clone + ZnRingStore, T1: ZnRingStore, T2: ZnRingStore, T3: ZnRingStore, I> Clone for PrimeDecomposition<'a, F, T1, T2, T3, I>
    where I: IntegerRingStore + Clone,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T1::Type: UsableZnRing<I>, <T1::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T2::Type: UsableZnRing<I>, <T2::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T3::Type: UsableZnRing<I>, <T3::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    fn clone(&self) -> Self {
        Self {
            from: self.from.clone(),
            to: self.to,
            inv_unit_vectors: (
                self.to.0.iter().zip(self.inv_unit_vectors.0.iter()).map(|(R, x)| R.clone_el(x)).collect(),
                self.to.1.iter().zip(self.inv_unit_vectors.1.iter()).map(|(R, x)| R.clone_el(x)).collect(),
                self.to.2.iter().zip(self.inv_unit_vectors.2.iter()).map(|(R, x)| R.clone_el(x)).collect(),
            ),
            unit_vectors: (
                self.unit_vectors.0.iter().map(|x| self.from.clone_el(x)).collect(),
                self.unit_vectors.1.iter().map(|x| self.from.clone_el(x)).collect(),
                self.unit_vectors.2.iter().map(|x| self.from.clone_el(x)).collect(),
            ),
            correction_unit_vectors: (
                self.correction_unit_vectors.0.iter().map(|x| self.int_ring.clone_el(x)).collect(),
                self.correction_unit_vectors.1.iter().map(|x| self.int_ring.clone_el(x)).collect(),
                self.correction_unit_vectors.2.iter().map(|x| self.int_ring.clone_el(x)).collect(),
            ),
            P_mod: self.from.clone_el(&self.P_mod),
            gamma: self.int_ring.clone_el(&self.gamma),
            int_ring: self.int_ring.clone(),
            to_ring_type: (PhantomData, PhantomData, PhantomData)
        }
    }
}

pub struct PrimeComposer<'a, F: ZnRingStore, T1: ZnRingStore, T2: ZnRingStore, T3: ZnRingStore, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T1::Type: UsableZnRing<I>, <T1::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T2::Type: UsableZnRing<I>, <T2::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T3::Type: UsableZnRing<I>, <T3::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
        
{
    parent: &'a PrimeDecomposition<'a, F, T1, T2, T3, I>,
    current_correction: El<I>,
    current: El<F>,
    supplied: usize
}

impl<'a, F: Clone + ZnRingStore, T1: ZnRingStore, T2: ZnRingStore, T3: ZnRingStore, I> PrimeComposer<'a, F, T1, T2, T3, I>
    where I: IntegerRingStore,
        I::Type: IntegerRing + CanIsoFromTo<StaticRingBase<i64>>,
        F::Type: ZnRing + CanHomFrom<I::Type> + CanHomFrom<BigIntRingBase>,
        <F::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T1::Type: UsableZnRing<I>, <T1::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T2::Type: UsableZnRing<I>, <T2::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>,
        T3::Type: UsableZnRing<I>, <T3::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
{
    #[inline(always)]
    fn supply_impl<T: ZnRingStore>(&mut self, rings: &[T], congruence: El<T>, index: usize, inv_unit_vectors: &[El<T>], correction_unit_vectors: &[El<I>], unit_vectors: &[El<F>]) 
        where T::Type: UsableZnRing<I>, <T::Type as ZnRing>::IntegerRingBase: CanIsoFromTo<I::Type>
    {
        debug_assert_eq!(inv_unit_vectors.len(), correction_unit_vectors.len());
        debug_assert_eq!(inv_unit_vectors.len(), unit_vectors.len());
        let to = rings.at(index);
        let lift = int_cast(
            to.smallest_lift(to.mul_ref_snd(congruence, &inv_unit_vectors.at(index))),
            &self.parent.int_ring,
            to.integer_ring()
        );
        self.parent.int_ring.add_assign(&mut self.current_correction, self.parent.int_ring.mul_ref(&lift, correction_unit_vectors.at(index)));
        self.parent.from.add_assign(&mut self.current, self.parent.from.mul_ref_snd(self.parent.from.coerce(&self.parent.int_ring, lift), unit_vectors.at(index)));
        self.supplied += 1;
    }

    ///
    /// # Algorithm
    /// 
    /// The basic idea is to use the formula `sum_i shortest_lift(yi) shortest_lift(ei) mod q` 
    /// where `ei` is the `i`-th "unit vector" modulo `P`. However, this introduces a really large error.
    /// 
    /// Instead, we can compute `sum_i shortest_lift(yi / (P/pi)) P/pi mod q`. Since `| shortest_lift(yi / (P/pi)) P/pi |`
    /// is at most `P`, we see that the result of this computation is `shortest_lift(y) + f P mod q`, where `|f| <= l`.
    /// 
    /// To correct it, we now also compute `C := sum_i shortest_lift(yi / (P/pi)) round(γ/pi)`.
    /// This value is now at most `lB / 4` away from `γ/P sum_i shortest_lift(yi / (P/pi)) P/pi`.
    /// Hence, we can estimate the error `f` from above as `C / γ`.
    /// 
    /// In particular, if the desired value `shortest_lift(y)` is bounded as `|shortest_lift(y)| + PlB/γ/4 <= P/2`,
    /// the result is correct. However, we remark that this method does not work for all `y`.
    /// 
    #[inline(always)]
    pub fn supply0(&mut self, congruence: El<T1>, index: usize) {
        self.supply_impl(self.parent.to.0, congruence, index, &self.parent.inv_unit_vectors.0, &self.parent.correction_unit_vectors.0, &self.parent.unit_vectors.0)
    }

    #[inline(always)]
    pub fn supply1(&mut self, congruence: El<T2>, index: usize) {
        self.supply_impl(self.parent.to.1, congruence, index, &self.parent.inv_unit_vectors.1, &self.parent.correction_unit_vectors.1, &self.parent.unit_vectors.1)
    }

    #[inline(always)]
    pub fn supply2(&mut self, congruence: El<T3>, index: usize) {
        self.supply_impl(self.parent.to.2, congruence, index, &self.parent.inv_unit_vectors.2, &self.parent.correction_unit_vectors.2, &self.parent.unit_vectors.2)
    }

    #[inline(always)]
    pub fn try_supply0(&mut self, congruence: El<T1>, index: usize) -> Result<(), ()> {
        if index < self.parent.len0() {
            self.supply0(congruence, index);
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline(always)]
    pub fn try_supply1(&mut self, congruence: El<T2>, index: usize) -> Result<(), ()> {
        if index < self.parent.len1() {
            self.supply1(congruence, index);
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline(always)]
    pub fn try_supply2(&mut self, congruence: El<T3>, index: usize) -> Result<(), ()> {
        if index < self.parent.len2() {
            self.supply2(congruence, index);
            Ok(())
        } else {
            Err(())
        }
    }

    #[inline(never)]
    pub fn finish(mut self) -> El<F> {
        // this does not ensure that exactly one element per prime has been provided, but
        // is better than no checks at all
        assert_eq!(self.supplied, self.parent.total_len());

        #[cfg(debug_assertions)] 
        {
            let ZZ = &self.parent.int_ring;
            let mut remainder = ZZ.euclidean_rem(ZZ.clone_el(&self.current_correction), &self.parent.gamma);
            if ZZ.is_neg(&remainder) {
                ZZ.add_assign_ref(&mut remainder, &self.parent.gamma);
            }
            if ZZ.is_gt(&remainder, &ZZ.euclidean_div(ZZ.clone_el(&self.parent.gamma), &ZZ.int_hom().map(2))) {
                ZZ.sub_assign_ref(&mut remainder, &self.parent.gamma);
            }
            assert!(ZZ.is_leq(&ZZ.abs(remainder), &ZZ.rounded_div(ZZ.clone_el(&self.parent.gamma), &ZZ.int_hom().map(4))), "input not bounded as expected");
        }

        let delta = self.parent.int_ring.rounded_div(self.current_correction, &self.parent.gamma);
        self.parent.from.sub_assign(&mut self.current, self.parent.from.mul_ref_snd(self.parent.from.coerce(&self.parent.int_ring, delta), &self.parent.P_mod));
        return self.current;
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_64::*;
#[cfg(test)]
use feanor_math::rings::finite::*;
#[cfg(test)]
use feanor_math::assert_el_eq;

#[test]
fn test_compose_decompose_1() {
    let from = Zn::new(257);
    let to = [Zn::new(2), Zn::new(3), Zn::new(5), Zn::new(7), Zn::new(11)];
    let empty: [Zn; 0] = [];
    let conv = PrimeDecomposition::new(from, (&to[..], &empty[..], &empty[..]), StaticRing::<i64>::RING, 55);
    assert_eq!(0.5, conv.assumed_slack_factor());

    for x in from.elements() {
        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(conv.direct_decompose0(x, i), i);
        }
        assert_el_eq!(&from, &x, &composer.finish());
    }

    for x in 0..(2 * 3 * 5 * 7 * 11 / 4) {
        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(to[i].int_hom().map(x), i);
        }
        assert_el_eq!(&from, &from.int_hom().map(x), &composer.finish());

        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(to[i].int_hom().map(-x), i);
        }
        assert_el_eq!(&from, &from.int_hom().map(-x), &composer.finish());
    }
}

#[test]
fn test_compose_decompose_3() {
    let from = Zn::new(257);
    let to = [Zn::new(2), Zn::new(3), Zn::new(5), Zn::new(7), Zn::new(11)];
    let conv = PrimeDecomposition::new(from, (&to[..2], &to[2..4], &to[4..]), StaticRing::<i64>::RING, 55);
    assert_eq!(0.5, conv.assumed_slack_factor());

    for x in from.elements() {
        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(conv.direct_decompose0(x, i), i);
        }
        for i in 0..conv.len1() {
            composer.supply1(conv.direct_decompose1(x, i), i);
        }
        for i in 0..conv.len2() {
            composer.supply2(conv.direct_decompose2(x, i), i);
        }
        assert_el_eq!(&from, &x, &composer.finish());
    }

    for x in 0..(2 * 3 * 5 * 7 * 11 / 4) {
        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(to[i].int_hom().map(x), i);
        }
        for i in 0..conv.len1() {
            composer.supply1(to[2 + i].int_hom().map(x), i);
        }
        for i in 0..conv.len2() {
            composer.supply2(to[4 + i].int_hom().map(x), i);
        }
        assert_el_eq!(&from, &from.int_hom().map(x), &composer.finish());

        let mut composer = conv.start_compose();
        for i in 0..conv.len0() {
            composer.supply0(to[i].int_hom().map(-x), i);
        }
        for i in 0..conv.len1() {
            composer.supply1(to[2 + i].int_hom().map(-x), i);
        }
        for i in 0..conv.len2() {
            composer.supply2(to[4 + i].int_hom().map(-x), i);
        }
        assert_el_eq!(&from, &from.int_hom().map(-x), &composer.finish());
    }
}

#[test]
#[should_panic(expected = "input not bounded as expected")]
fn test_check_compose_bound() {
    let from = Zn::new(257);
    let to = [Zn::new(2), Zn::new(3), Zn::new(5), Zn::new(7), Zn::new(11)];
    let conv = PrimeDecomposition::new(from, (&to[..2], &to[2..4], &to[4..]), StaticRing::<i64>::RING, 55);
    assert_eq!(0.5, conv.assumed_slack_factor());

    for x in 0..(3 * 5 * 7 * 11) {
        let mut composer = conv.start_compose();
        composer.supply0(to[0].int_hom().map(x), 0);
        composer.supply0(to[1].int_hom().map(x), 1);
        composer.supply1(to[2].int_hom().map(x), 0);
        composer.supply1(to[3].int_hom().map(x), 1);
        composer.supply2(to[4].int_hom().map(x), 0);
        assert_el_eq!(&from, &from.int_hom().map(x), &composer.finish());
    }
}