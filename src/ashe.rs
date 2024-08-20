use feanor_math::algorithms::fft::cooley_tuckey;
use feanor_math::default_memory_provider;
use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::homomorphism::Homomorphism;
use feanor_math::integer::{int_cast, BigIntRing};
use feanor_math::mempool::DefaultMemoryProvider;
use feanor_math::primitive_int::StaticRing;
use feanor_math::ring::{El, RingExtensionStore, RingStore};
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::rings::field::AsField;
use feanor_math::rings::float_complex::Complex64;
use feanor_math::rings::poly::dense_poly::DensePolyRing;
use feanor_math::rings::zn::zn_64::{Zn, ZnFastmul};
use feanor_math::integer::IntegerRingStore;
use feanor_math::ordered::OrderedRingStore;
use feanor_math::rings::zn::{zn_rns, ZnRingStore};
use feanor_math::vector::vec_fn::VectorFn;
use feanor_math::vector::VectorView;
use he_ring::complexfft::complex_fft_ring::ComplexFFTBasedRing;
use he_ring::doublerns::double_rns_ring::DoubleRNSRing;
use he_ring::*;
use rand::rngs::StdRng;
use rand::{CryptoRng, Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha12Rng;
use rand_distr::StandardNormal;

use crate::eval::TopLevelPrime;
use crate::strategy::sample_primes_arithmetic_progression;
use crate::SIGMA;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ASHEParams {
    pub log2_ring_degree: usize,
    pub t: i64,
    pub q_moduli_count: usize
}

pub type CiphertextRing = DoubleRNSRing<Zn, doublerns::pow2_cyclotomic::Pow2CyclotomicFFT<cooley_tuckey::FFTTableCooleyTuckey<ZnFastmul>>, DefaultMemoryProvider>;
pub type PlaintextRing = ComplexFFTBasedRing<complexfft::pow2_cyclotomic::Pow2CyclotomicFFT<AsField<Zn>, cooley_tuckey::FFTTableCooleyTuckey<Complex64>>, DefaultMemoryProvider, DefaultMemoryProvider>;

pub struct NewCiphertext {
    seed_of_b: [u8; 32],
    a: El<CiphertextRing>
}

impl ASHEParams {

    pub fn estimate_is_128bit_secure(&self) -> bool {
        if self.log2_ring_degree < 13 {
            false
        } else {
            self.q_bits() <= (220 << (self.log2_ring_degree - 13))
        }
    }

    pub fn available_toplevel_primes(&self) -> impl Iterator<Item = i64> {
        let t = self.t;
        sample_primes_arithmetic_progression(1, 2 << self.log2_ring_degree, 30).inspect(move |p| assert!(*p != t))
    }

    pub fn q_bits(&self) -> usize {
        self.available_toplevel_primes().map(|p| p as u64).take(self.q_moduli_count).map(|p| (p as f64).log2()).sum::<f64>() as usize
    }

    pub fn create_rns_base(&self) -> zn_rns::Zn<TopLevelPrime, BigIntRing> {
        let primes = self.available_toplevel_primes().map(|p| p as u64);
        zn_rns::Zn::new(primes.take(self.q_moduli_count).map(Zn::new).collect(), BigIntRing::RING, default_memory_provider!())
    }

    pub fn create_ciphertext_ring(&self) -> CiphertextRing {
        return <CiphertextRing as RingStore>::Type::new(self.create_rns_base(), self.create_rns_base().get_ring().iter().copied().map(ZnFastmul::new).collect(), self.log2_ring_degree, default_memory_provider!());
    }

    pub fn create_plaintext_ring(&self) -> PlaintextRing {
        return <PlaintextRing as RingStore>::Type::new(Zn::new(self.t as u64).as_field().ok().unwrap(), self.log2_ring_degree, default_memory_provider!(), default_memory_provider!());
    }

    ///
    /// Expands an element in the ciphertext ring from the given seed. This element will have
    /// RNS-components all in `{0, 1}`.
    /// 
    fn expand_01_crt(&self, C: &CiphertextRing, seed: [u8; 32]) -> El<CiphertextRing> {
        let mut result = C.zero();
        let mut components = self.components_01_crt(C, seed);
        for (i, Zp) in C.get_ring().rns_base().iter().enumerate() {
            let modulo_p = Zp.can_hom(&StaticRing::<i64>::RING).unwrap();
            for j in 0..C.rank() {
                *C.get_ring().fourier_coefficient_mut(i, j, &mut result) = modulo_p.map(components.next().unwrap() as i64);
            }
        }
        assert!(components.next().is_none());
        return result;
    }

    fn components_01_crt<'a>(&'a self, C: &'a CiphertextRing, seed: [u8; 32]) -> impl 'a + ExactSizeIterator<Item = u8> {
        let mut rng = ChaCha12Rng::from_seed(seed);
        (0..(C.get_ring().rns_base().len() * C.rank())).map(move |_| (rng.next_u32() % 2) as u8)
    }

    fn sample_ternary<G: Rng + CryptoRng>(&self, C: &CiphertextRing, rng: &mut G) -> El<CiphertextRing> {
        C.get_ring().sample_from_coefficient_distribution(|| (rng.next_u32() % 3) as i32 - 1)
    }

    fn sample_gaussian<G: Rng + CryptoRng>(&self, C: &CiphertextRing, rng: &mut G) -> El<CiphertextRing> {
        C.get_ring().sample_from_coefficient_distribution(|| {
            (rng.sample::<f64, _>(StandardNormal) * SIGMA).round() as i32
        })
    }

    pub fn ciphertext_a<'a>(&'a self, _C: &'a CiphertextRing, ct: &'a NewCiphertext) -> &'a El<CiphertextRing> {
        &ct.a
    }

    pub fn ciphertext_b_expanded(&self, C: &CiphertextRing, ct: &NewCiphertext) -> El<CiphertextRing> {
        self.expand_01_crt(C, ct.seed_of_b)
    }

    pub fn ciphertext_b_components<'a>(&'a self, C: &'a CiphertextRing, ct: &'a NewCiphertext) -> impl 'a + ExactSizeIterator<Item = u8> {
        self.components_01_crt(&C, ct.seed_of_b)
    }

    pub fn sample_sk<G: Rng + CryptoRng>(&self, C: &CiphertextRing, rng: &mut G) -> El<CiphertextRing> {
        let mut result = self.sample_ternary(C, rng);
        while !C.is_unit(&result) {
            result = self.sample_ternary(C, rng);
            println!("SK not a unit, resample");
        }
        return result;
    }

    pub fn encrypt<G: Rng + CryptoRng>(&self, C: &CiphertextRing, P: &PlaintextRing, rng: &mut G, m: &El<PlaintextRing>, sk: &El<CiphertextRing>) -> NewCiphertext {
        let m_coeffs = P.wrt_canonical_basis(m);
        let modulo_q = C.base_ring().can_hom(P.base_ring().integer_ring()).unwrap();
        let m_in_C = C.from_canonical_basis((0..P.rank()).map(|i| modulo_q.map(P.base_ring().smallest_lift(m_coeffs.at(i)))));

        let b_seed = std::array::from_fn(|_| (rng.next_u32() & u8::MAX as u32) as u8);
        let e = C.mul(self.sample_gaussian(C, rng), C.int_hom().map(self.t as i32));
        let e_m = C.add(m_in_C, e);

        let b = self.expand_01_crt(C, b_seed);
        let a = C.checked_div(&C.sub_ref_snd(e_m, &b), sk).unwrap();

        let CC = Complex64::RING;
        let poly_ring = DensePolyRing::new(C.base_ring(), "X");
        let critical_quantity = C.add_ref_fst(&b, C.mul_ref(sk, &a));
        let critical_quantity = C.wrt_canonical_basis(&critical_quantity);
        let in_CC = (0..(1 << self.log2_ring_degree)).map(|i| CC.from_f64(int_cast(C.base_ring().smallest_lift(critical_quantity.at(i)), StaticRing::<i64>::RING, C.base_ring().integer_ring()) as f64)).collect::<Vec<_>>();

        let mut max_error_bits = 0.;
        for i in (1..(2 << self.log2_ring_degree)).step_by(2) {
            let minkowski_embedding_vector_entry = CC.sum((0..(1 << self.log2_ring_degree)).map(|j| CC.mul(in_CC[j], CC.root_of_unity((i * j) as i64, 2 << self.log2_ring_degree))));
            let log2_abs = CC.abs(minkowski_embedding_vector_entry).log2();
            if log2_abs > max_error_bits {
                max_error_bits = log2_abs;
            }
        }
        println!("Critical quantity of encryption: {} bits", max_error_bits);

        return NewCiphertext {
            a: a,
            seed_of_b: b_seed
        };
    }

    pub fn decrypt<I>(&self, C: &CiphertextRing, P: &PlaintextRing, mut ct: I, sk: &El<CiphertextRing>) -> El<PlaintextRing>
        where I: Iterator<Item = El<CiphertextRing>>
    {
        let mut noisy_m = ct.next().unwrap();
        for cti in ct {
            C.mul_assign_ref(&mut noisy_m, sk);
            C.add_assign(&mut noisy_m, cti);
        }
        let noisy_m = C.wrt_canonical_basis(&noisy_m);
        let modulo_t = P.base_ring().can_hom(C.base_ring().integer_ring()).unwrap();
        let ZZbig = C.base_ring().integer_ring();
        let bound = ZZbig.rounded_div(ZZbig.clone_el(C.base_ring().modulus()), &ZZbig.int_hom().map(4));
        return P.from_canonical_basis((0..P.rank()).map(|i| {
            let lift = C.base_ring().smallest_lift(noisy_m.at(i));
            let result = modulo_t.map_ref(&lift);
            let ZZbig = C.base_ring().integer_ring();
            // assert!(ZZbig.is_lt(&ZZbig.abs(lift), &bound));
            return result;
        }));
    }
}

#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use rand::thread_rng;

#[test]
fn test_ashe_enc_dec() {
    let scheme = ASHEParams {
        log2_ring_degree: 5,
        q_moduli_count: 2,
        t: 5
    };
    let P = scheme.create_plaintext_ring();
    let C = scheme.create_ciphertext_ring();
    let m = P.from_canonical_basis((0..P.rank()).map(|i| P.base_ring().int_hom().map(i as i32)));
    let sk = scheme.sample_sk(&C, &mut thread_rng());
    let ct = scheme.encrypt(&C, &P, &mut thread_rng(), &m, &sk);
    let result = scheme.decrypt(&C, &P, [C.clone_el(scheme.ciphertext_a(&C, &ct)), scheme.ciphertext_b_expanded(&C, &ct)].into_iter(), &sk);
    assert_el_eq!(&P, &m, &result);
}

#[test]
fn test_ashe_dec_hommul() {
    let scheme = ASHEParams {
        log2_ring_degree: 5,
        q_moduli_count: 2,
        t: 5
    };
    let P = scheme.create_plaintext_ring();
    let C = scheme.create_ciphertext_ring();
    let m = P.from_canonical_basis((0..P.rank()).map(|i| P.base_ring().int_hom().map(i as i32)));
    let sk = scheme.sample_sk(&C, &mut thread_rng());
    let ct = scheme.encrypt(&C, &P, &mut thread_rng(), &m, &sk);
    let mut expanded = [C.clone_el(scheme.ciphertext_a(&C, &ct)), scheme.ciphertext_b_expanded(&C, &ct)];
    C.mul_assign(&mut expanded[0], C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap()).map(scheme.t * (1 << 29) + 1));
    C.mul_assign(&mut expanded[1], C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap()).map(scheme.t * (1 << 29) + 1));
    let result = scheme.decrypt(&C, &P, expanded.into_iter(), &sk);
    assert_el_eq!(&P, &m, &result);
}

#[test]
fn test_ashe_decryption_failure() {
    let scheme = ASHEParams {
        log2_ring_degree: 5,
        q_moduli_count: 1,
        t: 5
    };
    let P = scheme.create_plaintext_ring();
    let C = scheme.create_ciphertext_ring();
    let m = P.from_canonical_basis((0..P.rank()).map(|i| P.base_ring().int_hom().map(i as i32)));

    let mut rng = StdRng::from_seed([0; 32]);
    let sk = scheme.sample_sk(&C, &mut rng);
    let ct = scheme.encrypt(&C, &P, &mut rng, &m, &sk);
    let mut expanded = [C.clone_el(scheme.ciphertext_a(&C, &ct)), scheme.ciphertext_b_expanded(&C, &ct)];
    C.mul_assign(&mut expanded[0], C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap()).map(scheme.t * (1 << 29) + 1));
    C.mul_assign(&mut expanded[1], C.inclusion().compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap()).map(scheme.t * (1 << 29) + 1));
    let result = scheme.decrypt(&C, &P, expanded.into_iter(), &sk);
    assert!(!P.eq_el(&m, &result));
}