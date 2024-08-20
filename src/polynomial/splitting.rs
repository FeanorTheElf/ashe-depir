use feanor_math::homomorphism::Homomorphism;
use feanor_math::ring::*;

use crate::polynomial::{index_of_monomial, monomials_iter};
use crate::strategy::binomial;

fn split_one_b_XY<R: RingStore>(data: &[El<R>], ring: R, m: usize, d: usize, var_index: usize, b: &El<R>) -> Vec<Vec<El<R>>> {
    assert_eq!(usize::try_from(binomial(d + m, m)).unwrap(), data.len());
    let mut output = (0..=d).map(|_| (0..binomial(d + m, m)).map(|_| ring.zero()).collect::<Vec<_>>()).collect::<Vec<_>>();

    monomials_iter(m, d).enumerate().for_each(|(j, mut monomial)| {
        debug_assert_eq!(j, index_of_monomial(m, d, &monomial));
        assert_eq!(j, index_of_monomial(m, d, &monomial));
        // println!("{:?}", monomial);
        let coeff = &data[j];
        let power_of_vari = monomial[var_index];
        assert!(power_of_vari <= d);
        for i in 0..=power_of_vari {
            monomial[var_index] = i;
            let factor = ring.mul(ring.int_hom().map(binomial(power_of_vari, i).try_into().unwrap()), ring.pow(ring.clone_el(b), power_of_vari - i));
            ring.add_assign(&mut output[i][index_of_monomial(m, d, &monomial)], ring.mul_ref_fst(coeff, factor));
        }
    });

    return output;
} 

pub fn split_b_XY<R: RingStore>(data: &[El<R>], ring: R, m: usize, d: usize, bs: &[El<R>]) -> Vec<Vec<El<R>>> {
    assert_eq!(usize::try_from(binomial(d + m, m)).unwrap(), data.len());
    let mut output = split_one_b_XY(data, &ring, m, d, 0, &bs[0]);
    for i in 1..m {
        let mut new = split_one_b_XY(&output[0], &ring, m, d, i, &bs[i]);
        for k in 1..output.len() {
            for (j, poly) in split_one_b_XY(&output[k], &ring, m, d, i, &bs[i]).into_iter().enumerate() {
                if k + j <= d {
                    add_assign_mul(&poly, &mut new[k + j], &ring.one(), &ring);
                } else {
                    assert!(poly.iter().all(|x| ring.is_zero(x)));
                }
            }
        }
        output = new;
    }
    // check that the polynomials are indeed homogeneous
    for (i, poly) in output.iter().enumerate() {
        monomials_iter(m, d).enumerate().for_each(|(j, mon)| assert!(ring.is_zero(&poly[j]) || mon.iter().copied().sum::<usize>() == i));
    }
    return output;
}

pub fn decrease_degree<R: RingStore>(data: &[El<R>], ring: R, m: usize, d: usize, new_d: usize) -> Vec<El<R>> {
    let mut output = (0..binomial(new_d + m, m)).map(|_| ring.zero()).collect::<Vec<_>>();
    monomials_iter(m, d).enumerate().for_each(|(j, mon)| {
        debug_assert_eq!(j, index_of_monomial(m, d, &mon));
        if mon.iter().copied().sum::<usize>() <= new_d {
            output[index_of_monomial(m, new_d, &mon)] = ring.clone_el(&data[j]);
        } else {
            assert!(ring.is_zero(&data[j]));
        }
    });
    return output;
}

pub fn diff<R: RingStore>(data: &[El<R>], ring: R, m: usize, d: usize, var_index: usize) -> Vec<El<R>> {
    assert_eq!(usize::try_from(binomial(d + m, m)).unwrap(), data.len());
    let mut output = (0..binomial(d + m - 1, m)).map(|_| ring.zero()).collect::<Vec<_>>();
    monomials_iter(m, d).enumerate().for_each(|(j, mut monomial)| {
        debug_assert_eq!(j, index_of_monomial(m, d, &monomial));
        let coeff = &data[j];
        let current_power = monomial[var_index];
        if monomial[var_index] > 0 {
            monomial[var_index] -= 1;
            ring.add_assign(&mut output[index_of_monomial(m, d - 1, &monomial)], ring.mul_ref_fst(coeff, ring.int_hom().map(current_power as i32)));
        }
    });
    return output;
}

fn add_assign_mul<R: RingStore>(src: &[El<R>], dst: &mut [El<R>], factor: &El<R>, ring: R) {
    assert_eq!(src.len(), dst.len());
    for i in 0..src.len() {
        ring.add_assign(&mut dst[i], ring.mul_ref(&src[i], factor));
    }
}

pub fn specialize_last<R: RingStore>(poly: &[El<R>], ring: R, m: usize, d: usize, value: &El<R>) -> Vec<El<R>> {
    assert_eq!(usize::try_from(binomial(m + d, m)).unwrap(), poly.len());
    let mut output = (0..binomial(m + d - 1, m - 1)).map(|_| ring.zero()).collect::<Vec<_>>();
    let var_index = m - 1;
    monomials_iter(m, d).enumerate().for_each(|(j, monomial)| {
        debug_assert_eq!(j, index_of_monomial(m, d, &monomial));
        let coeff = &poly[j];
        let current_power = monomial[var_index];
        ring.add_assign(&mut output[index_of_monomial(m - 1, d, &monomial[..(m - 1)])], ring.mul_ref_fst(coeff, ring.pow(ring.clone_el(value), current_power)));
    });
    return output;
}

#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::primitive_int::StaticRing;
#[cfg(test)]
use feanor_math::rings::multivariate::{ordered::MultivariatePolyRingImpl, DegRevLex};
#[cfg(test)]
use feanor_math::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use super::poly_as_ring_el;
#[cfg(test)]
use feanor_math::rings::multivariate::MultivariatePolyRingStore;
#[cfg(test)]
use feanor_math::rings::poly::PolyRingStore;

#[test]
fn test_split_one_b_XY() {
    let d = 2;
    let m = 3;
    // 2 + X + 4Y + 2XY + YZ + Z^2
    let poly = [2, 1, 0, 4, 2, 0, 0, 0, 1, 1];

    let P: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    let poly_P = poly_as_ring_el::<_, _, _, 3>(&poly, d, &P, &P.base_ring().identity());

    let PY = DensePolyRing::new(&P, "Y");

    // specialize at `2 + Y * X0`
    let expected = P.evaluate(&poly_P, [
        PY.add(PY.int_hom().map(2), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(0)))),
        PY.inclusion().map(P.indeterminate(1)),
        PY.inclusion().map(P.indeterminate(2)),
    ], &PY.inclusion().compose(PY.base_ring().inclusion()));

    let actual = split_one_b_XY(&poly, &StaticRing::<i64>::RING, m, d, 0, &2);

    assert_eq!(3, actual.len());
    for i in 0..3 {
        assert_el_eq!(&P, PY.coefficient_at(&expected, i), &poly_as_ring_el::<_, _, _, 3>(&actual[i], d, &P, &P.base_ring().identity()));
    }
}

#[test]
fn test_split_b_XY() {
    let d = 2;
    let m = 3;
    // 2 + X + 4Y + 2XY + YZ + Z^2
    let poly = [2, 1, 0, 4, 2, 0, 0, 0, 1, 1];

    let P: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    let poly_P = poly_as_ring_el::<_, _, _, 3>(&poly, d, &P, &P.base_ring().identity());

    let PY = DensePolyRing::new(&P, "Y");

    // specialize at `Y * X0`, `Y * X1`, `Y * X2`
    let expected = P.evaluate(&poly_P, [
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(0))),
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(1))),
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(2))),
    ], &PY.inclusion().compose(PY.base_ring().inclusion()));

    let actual = split_b_XY(&poly, &StaticRing::<i64>::RING, m, d, &[0, 0, 0]);

    assert_eq!(3, actual.len());
    for i in 0..3 {
        assert_el_eq!(&P, PY.coefficient_at(&expected, i), &poly_as_ring_el::<_, _, _, 3>(&actual[i], d, &P, &P.base_ring().identity()));
    }

    // specialize at `2 + Y * X0`, `-1 + Y * X1`, `4 + Y * X2`
    let expected = P.evaluate(&poly_P, [
        PY.add(PY.int_hom().map(2), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(0)))),
        PY.add(PY.int_hom().map(-1), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(1)))),
        PY.add(PY.int_hom().map(4), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(2)))),
    ], &PY.inclusion().compose(PY.base_ring().inclusion()));

    let actual = split_b_XY(&poly, &StaticRing::<i64>::RING, m, d, &[2, -1, 4]);

    assert_eq!(3, actual.len());
    for i in 0..3 {
        assert_el_eq!(&P, PY.coefficient_at(&expected, i), &poly_as_ring_el::<_, _, _, 3>(&actual[i], d, &P, &P.base_ring().identity()));
    }
}

#[test]
fn test_split_homog_b_XY() {
    let d = 2;
    let m = 3;
    // X^2 + 2XY + 3XZ - Z^2
    let poly = [0, 0, 1, 0, 2, 0, 0, 3, 0, -1];

    let P: MultivariatePolyRingImpl<_, _, _, 3> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    let poly_P = poly_as_ring_el::<_, _, _, 3>(&poly, d, &P, &P.base_ring().identity());

    let PY = DensePolyRing::new(&P, "Y");

    // specialize at `Y * X0`, `Y * X1`, `Y * X2`
    let expected = P.evaluate(&poly_P, [
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(0))),
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(1))),
        PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(2))),
    ], &PY.inclusion().compose(PY.base_ring().inclusion()));

    let actual = split_b_XY(&poly, &StaticRing::<i64>::RING, m, d, &[0, 0, 0]);

    assert_eq!(3, actual.len());
    for i in 0..3 {
        assert_el_eq!(&P, PY.coefficient_at(&expected, i), &poly_as_ring_el::<_, _, _, 3>(&actual[i], d, &P, &P.base_ring().identity()));
    }

    // specialize at `2 + Y * X0`, `-1 + Y * X1`, `4 + Y * X2`
    let expected = P.evaluate(&poly_P, [
        PY.add(PY.int_hom().map(2), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(0)))),
        PY.add(PY.int_hom().map(-1), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(1)))),
        PY.add(PY.int_hom().map(4), PY.mul(PY.indeterminate(), PY.inclusion().map(P.indeterminate(2)))),
    ], &PY.inclusion().compose(PY.base_ring().inclusion()));

    let actual = split_b_XY(&poly, &StaticRing::<i64>::RING, m, d, &[2, -1, 4]);

    assert_eq!(3, actual.len());
    for i in 0..3 {
        assert_el_eq!(&P, PY.coefficient_at(&expected, i), &poly_as_ring_el::<_, _, _, 3>(&actual[i], d, &P, &P.base_ring().identity()));
    }
}

#[test]
fn test_diff() {
    let m = 4;
    let d = 3;
    
    // x^3 - xz + 2y^2w - 2w + 1 
    let poly_lex: [i32; 35] = [
        1, 0, 0, 1, 0, 0, 0, 0, 0, 0, // z^0 w^0
        0, -1, 0, 0, 0, 0, // z^1 w^0
        0, 0, 0, // z^2 w^0
        0, // z^3
        -2, 0, 0, 0, 0, 2, // z^0 w^1
        0, 0, 0, // z^1 w^1
        0, // z^2 w^1
        0, 0, 0, // z^0 w^2
        0, // z^1 w^2
        0, // w^3
    ];

    // 4yw
    let result_lex: [i32; 15] = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, // w^0
        0, 0, 4, 0, // w^1
        0 // w^2
    ];

    assert_eq!(&result_lex, &diff(&poly_lex, &StaticRing::<i32>::RING, m, d, 1)[..]);
}