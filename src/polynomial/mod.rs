use std::convert::identity;

use feanor_math::iters::multi_cartesian_product;
use feanor_math::ring::*;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::rings::multivariate::*;

use crate::strategy::binomial;

pub mod interpolate;
pub mod splitting;
pub mod io;

pub fn index_of_monomial(m: usize, d: usize, monomial: &[usize]) -> usize {
    if m == 0 {
        return 0;
    }
    assert_eq!(m, monomial.len());
    assert!(monomial[m - 1] <= d);
    index_of_monomial(m - 1, d - monomial[m - 1], &monomial[..(m - 1)]) + (0..monomial[m - 1]).map(|k| usize::try_from(binomial(d + m - k - 1, m - 1)).unwrap()).sum::<usize>()
}

pub fn monomials_iter(m: usize, d: usize) -> impl Clone + Iterator<Item = Vec<usize>> {
    multi_cartesian_product((0..m).map(|_| (0..=d)), move |slice| if slice.iter().copied().sum::<usize>() <= d { Some(slice.iter().rev().copied().collect()) } else { None }, |_, x| *x).filter_map(identity)
}

///
/// Reorders the coefficients of the given polynomial from lex order (used in all functions here) to
/// deglex order (used in the GPU accelerated evaluation table computation).
/// 
pub fn convert_lex_to_deglex<I, T>(m: usize, d: usize, poly_in: I, poly_out: &mut [T])
    where I: ExactSizeIterator + Iterator<Item = T>
{
    assert_eq!(usize::try_from(binomial(m + d, m)).unwrap(), poly_in.len());
    assert_eq!(usize::try_from(binomial(m + d, m)).unwrap(), poly_out.len());

    let mut js = {
        let mut result = (0..=d).map(|k| binomial(k + m - 2, m - 1) as usize).collect::<Vec<_>>();
        for i in 0..d {
            result[i + 1] += result[i];
        }
        result
    };

    for (degree, value) in multi_cartesian_product((0..m).map(|_| (0..=d)), |degrees| degrees.iter().copied().sum(), |_, x| *x).filter(|degree: &usize| *degree <= d).zip(poly_in) {
        poly_out[js[degree]] = value;
        js[degree] += 1;
    }
}

#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::primitive_int::StaticRing;
#[cfg(test)]
use feanor_math::rings::multivariate::ordered::*;

pub fn poly_as_ring_el<R, P, H, const m: usize>(poly: &[R::Element], d: usize, poly_ring: &P, hom: &H) -> El<P>
    where R: RingBase,
        P: MultivariatePolyRingStore,
        P::Type: MultivariatePolyRing,
        H: Homomorphism<R, <<P::Type as RingExtension>::BaseRing as RingStore>::Type> 
{
    assert!(poly_ring.base_ring().get_ring() == hom.codomain().get_ring());
    poly_ring.from_terms(
        multi_cartesian_product((0..m).map(|_| (0..=d)), |indices| if indices.iter().copied().sum::<usize>() > d { None } else { Some(std::array::from_fn::<_, m, _>(|k| indices[m - 1 - k])) }, |_, x| *x)
            .filter_map(|indices| indices)
            .zip(poly.iter())
            .map(|(indices, c)| (hom.map_ref(c), poly_ring.get_ring().create_monomial(indices.into_iter().map(|i| i as u16))))
    )
}

#[test]
fn test_convert_lex_to_deglex() {
    let poly_lex = [1, 0, 1, -1, 1, 0, 0, 2, 3, 1];
    let poly_deglex = [1, 0, -1, 0, 1, 1, 0, 2, 3, 1];

    let mut out = [0; 10];
    convert_lex_to_deglex(3, 2, poly_lex.iter().copied(), &mut out);

    assert_eq!(poly_deglex, out);
}

#[test]
fn test_as_ring_el() {
    {
        const m: usize = 3;

        let P: MultivariatePolyRingImpl<_, _, _, m> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
        
        // 1 + x^2 - y + yx + 2zx + 3zy + z^2
        let poly_lex = [1, 0, 1, -1, 1, 0, 0, 2, 3, 1];
        let poly = poly_as_ring_el::<_, _, _, m>(&poly_lex, 2, &P, &P.base_ring().identity());

        assert_eq!(1, *P.coefficient_at(&poly, &Monomial::new([0, 0, 0])));
        assert_eq!(0, *P.coefficient_at(&poly, &Monomial::new([1, 0, 0])));
        assert_eq!(1, *P.coefficient_at(&poly, &Monomial::new([2, 0, 0])));
        assert_eq!(-1, *P.coefficient_at(&poly, &Monomial::new([0, 1, 0])));
        assert_eq!(1, *P.coefficient_at(&poly, &Monomial::new([1, 1, 0])));
        assert_eq!(0, *P.coefficient_at(&poly, &Monomial::new([0, 2, 0])));
        assert_eq!(0, *P.coefficient_at(&poly, &Monomial::new([0, 0, 1])));
        assert_eq!(2, *P.coefficient_at(&poly, &Monomial::new([1, 0, 1])));
        assert_eq!(3, *P.coefficient_at(&poly, &Monomial::new([0, 1, 1])));
        assert_eq!(1, *P.coefficient_at(&poly, &Monomial::new([0, 0, 2])));
    }
    {
        const m: usize = 4;

        let P: MultivariatePolyRingImpl<_, _, _, m> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
        
        // x^3 - xz + 2y^2w - 2w + 1 
        let poly_lex = [
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
        let actual = poly_as_ring_el::<_, _, _, m>(&poly_lex, 3, &P, &P.base_ring().identity());
        let expected = P.from_terms([
            (1, Monomial::new([3, 0, 0, 0])),
            (-1, Monomial::new([1, 0, 1, 0])),
            (2, Monomial::new([0, 2, 0, 1])),
            (-2, Monomial::new([0, 0, 0, 1])),
            (1, Monomial::new([0, 0, 0, 0]))
        ].into_iter());

        assert_el_eq!(&P, &expected, &actual);
    }
}
