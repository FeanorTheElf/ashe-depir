#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![feature(iter_advance_by)]
#![feature(cell_update)]
#![feature(fn_traits)]
#![feature(unboxed_closures)]
#![feature(never_type)]
#![feature(thread_id_value)]
#![feature(generic_const_exprs)]
#![feature(core_intrinsics)]

use std::time::{Instant, SystemTime};

use feanor_math::homomorphism::*;
use feanor_math::integer::{int_cast, BigIntRing, IntegerRingStore};
use feanor_math::primitive_int::{StaticRing, StaticRingBase};
use feanor_math::rings::multivariate::ordered::*;
use feanor_math::default_memory_provider;
use feanor_math::ring::*;
use feanor_math::rings::field::AsField;
use feanor_math::rings::finite::FiniteRingStore;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::vector::VectorView;
use feanor_math::rings::multivariate::*;
use feanor_math::vector::vec_fn::IntoVectorFn;
use feanor_math::rings::extension::FreeAlgebraStore;
use feanor_math::divisibility::DivisibilityRingStore;

use rand::*;

use rngs::StdRng;
use statrs::function::erf::erf_inv;
use strategy::binomial_big;

use crate::polynomial::poly_as_ring_el;
use crate::polynomial::interpolate::InterpolationMatrix;
use crate::strategy::{binomial, PolynomialStats};
use crate::zn_eval::read_disk::PERFORMED_DISK_QUERIES;
use crate::zn_eval::read_ram::PERFORMED_RAM_QUERIES;
use crate::polynomial::io::poly_hash;
use crate::polynomial::splitting::{decrease_degree, specialize_last, split_b_XY};
use crate::zn_eval::async_read::print_disk_read_stats;
use crate::ashe::{ASHEParams, CiphertextRing, NewCiphertext};
use crate::eval::*;
use crate::eval::l2_zp2::{L2Zp2Plan, StrategyL2Zp2};

extern crate feanor_math;
extern crate he_ring;
extern crate rand;
extern crate rand_chacha;
extern crate rand_distr;
extern crate rayon;
extern crate selfref;
extern crate windows_sys;

pub mod strategy;
pub mod zn_eval;
pub mod ashe;
pub mod polynomial;
pub mod eval;

///
/// The polynomial that interpolates the database will have that many variables.
/// 
pub const m: usize = 4;
///
/// Path to the executable that will create the preprocessing database.
/// 
pub const PREPROCESSOR_PATH: &str = "F:\\Users\\Simon\\Documents\\Projekte\\poly-batch-eval\\x64\\Release\\poly-batch-eval";
///
/// Path to the directory where the preprocessing database is/will be stored.
/// 
pub const DATASTRUCTURE_PATH: &'static str = "E:\\";

pub const SIGMA: f64 = 3.2;
pub const reduced_m: usize = m - 1;
pub const component_count: usize = 1 << m; 

#[derive(Clone, Copy, Debug)]
struct DEPIRParams {
    t: i64,
    d: usize,
    q_moduli_count: usize,
    log2_ring_degree: usize,
    use_ram_bytes: usize
}


impl DEPIRParams {

    fn ashe_params(&self) -> ASHEParams {
        ASHEParams {
            log2_ring_degree: self.log2_ring_degree,
            t: self.t,
            q_moduli_count: self.q_moduli_count
        }
    }

    fn poly_params<'a>(&'a self) -> impl 'a + Iterator<Item = PolynomialStats> {
        let max_log2_coefficient_size = (0..=self.d).map(|i| StaticRing::<i64>::RING.abs_log2_ceil(&self.t).unwrap() + 
            BigIntRing::RING.abs_log2_ceil(&binomial_big(self.d, i)).unwrap() +  
            BigIntRing::RING.abs_log2_ceil(&binomial_big(i + m, i)).unwrap()).max().unwrap();
        (0..=self.d).map(move |i| PolynomialStats {
            m: m,
            D: self.d - i,
            monomials: usize::try_from(binomial(self.d + reduced_m - i as usize, reduced_m)).unwrap(),
            // `fi` is the sum of at most `binomial(i + m, i)` polynomials that result from `f` by taking `i` derivatives (w.r.t. arbitrary variables), divided by `i!`;
            // taking these derivates results in coeff size of `binomial(d, i)`
            log_coefficient_size: StaticRing::<i64>::RING.abs_log2_ceil(&self.t).unwrap() + 
                BigIntRing::RING.abs_log2_ceil(&binomial_big(self.d, i)).unwrap() +  
                BigIntRing::RING.abs_log2_ceil(&binomial_big(i + m, i)).unwrap()
        })
    }

    fn top_level_primes(&self) -> Vec<TopLevelPrime> {
        self.ashe_params().create_rns_base().get_ring().iter().copied().collect::<Vec<_>>()
    }

    fn strategy(&self) -> StrategyL2Zp2 {
        StrategyL2Zp2
    }

    fn N(&self) -> usize {
        binomial(self.d + m, m).try_into().unwrap()
    }

    fn Zt(&self) -> AsField<Zn> {
        Zn::new(self.t as u64).as_field().ok().unwrap()
    }

    fn interpolation_grid(&self) -> InterpolationMatrix<AsField<Zn>, Vec<El<AsField<Zn>>>> {
        let Zt = self.Zt();
        InterpolationMatrix::new(
            (0..m).map(|_| Zt.elements().take(self.d + 1)
                .collect::<Vec<_>>()), 
                Zt.clone()
        )
    }

    fn boring_database(&self) -> Vec<El<AsField<Zn>>> {
        let Zt = self.Zt();
        let mod_t = Zt.can_hom(&StaticRing::<i64>::RING).unwrap();
        (0..self.N()).map(|i| mod_t.map(i.trailing_ones() as i64)).collect::<Vec<_>>()

        // this database turns out to have very low degree polynomial
        // (0..self.N()).map(|i| mod_t.map(i as i64)).collect::<Vec<_>>()
    }

    fn compute_all_polynomials<'a>(&'a self, mut database: Vec<El<AsField<Zn>>>) -> (String, Vec<i64>, impl 'a + ExactSizeIterator<Item = Vec<Vec<i64>>>) {
        assert_eq!(self.N(), database.len());
        let ZZ = StaticRing::<i64>::RING;
        let Zt = self.Zt();
        let interpolation_grid = self.interpolation_grid();
        interpolation_grid.solve_inplace(&mut database, self.d, &Zt.identity());
        println!("Solved interpolation problem");
        let polynomial = database.into_iter().map(|x| Zt.smallest_lift(x)).collect::<Vec<_>>();
        return (
            poly_hash(&polynomial, &ZZ), 
            polynomial.clone(),
            (0..(1 << m)).map(move |b_index| {
                let result = split_b_XY(&polynomial, ZZ, m, self.d, &(0..m).map(|j| ((b_index >> j) & 1) as i64).collect::<Vec<_>>())
                    .into_iter()
                    .enumerate()
                    .map(|(i, poly)| (i, decrease_degree(&poly, ZZ, m, self.d, i)))
                    .map(|(i, poly)| specialize_last(&poly, ZZ, m, i, &1))
                    // reverse the order so that the degrees are decreasing d, d - 1, ..., 1, 0
                    .rev()
                    .collect::<Vec<_>>();

                assert!(result.len() == self.poly_params().count());
                for (poly_params, poly) in self.poly_params().zip(result.iter()) {
                    assert!(poly.len() == usize::try_from(binomial(poly_params.D + reduced_m, reduced_m)).unwrap());
                    assert!(poly.iter().all(|c| c.abs() < (1 << poly_params.log_coefficient_size)));
                }
                return result;
            })
        );
    }

    fn prepare_preprocessing<I>(&self, poly_hash: &str, all_polynomials: I)
        where I: ExactSizeIterator + Iterator<Item = Vec<Vec<i64>>>
    {
        self.strategy().prepare_preprocessing(self.top_level_primes(), &self.poly_params().collect::<Vec<_>>(), poly_hash, all_polynomials, false)
    }

    fn evaluation_plan(&self) -> L2Zp2Plan {
        let ashe_params = self.ashe_params();
        StrategyL2Zp2.plan(&self.top_level_primes(), &self.poly_params().collect::<Vec<_>>(), 1 << ashe_params.log2_ring_degree, self.use_ram_bytes)
    }

    #[allow(unused)]
    fn call_preprocessor(&self, poly_hash: &str) {
        self.strategy().call_preprocessor(self.top_level_primes(), &self.poly_params().collect::<Vec<_>>(), poly_hash, false);
    }

    fn create_evaluator(&self, poly_hash: &str) -> RamDiskZp2ASHEEvaluator<'static> {
        self.strategy().create(self.top_level_primes(), &self.poly_params().collect::<Vec<_>>(), poly_hash, 1 << self.ashe_params().log2_ring_degree, self.use_ram_bytes)
    }

    fn evaluate(&self, ciphertext_ring: &CiphertextRing, evaluator: &RamDiskZp2ASHEEvaluator, ciphertexts: &[NewCiphertext]) -> Vec<El<CiphertextRing>> {
        assert_eq!(ciphertexts.len(), m);
        let ashe_params = self.ashe_params();
        evaluator.evaluate(
            ciphertext_ring, 
            std::array::from_fn(|k| ashe_params.ciphertext_a(ciphertext_ring, &ciphertexts[k])), 
            std::array::from_fn(|k| ashe_params.ciphertext_b_components(ciphertext_ring, &ciphertexts[k]))
        )
    }
}

#[allow(unused)]
const TEST_PARAMS: DEPIRParams = DEPIRParams {
    d: 2,
    t: 65537,
    q_moduli_count: 11,
    log2_ring_degree: 4,
    use_ram_bytes: 1 << 30
};

#[allow(unused)]
const TEST_FORMAT_PARAMS: DEPIRParams = DEPIRParams {
    d: 10,
    t: 65537,
    q_moduli_count: 20,
    log2_ring_degree: 14,
    use_ram_bytes: 1 << 30
};

#[allow(unused)]
const SMALLBENCH_PARAMS: DEPIRParams = DEPIRParams {
    d: 14,
    t: 65537,
    q_moduli_count: 20,
    log2_ring_degree: 15,
    use_ram_bytes: 8 << 30
};

#[allow(unused)]
const BENCH_PARAMS: DEPIRParams = DEPIRParams {
    d: 18,
    t: 65537,
    q_moduli_count: 20,
    log2_ring_degree: 15,
    use_ram_bytes: 16 << 30
};

#[allow(unused)]
const SSD_PARAMS: DEPIRParams = DEPIRParams {
    d: 30,
    t: 65537,
    q_moduli_count: 29,
    log2_ring_degree: 15,
    use_ram_bytes: 28 << 30
};

#[allow(unused)]
const EXP_PARAMS: DEPIRParams = DEPIRParams {
    d: 68,
    t: 65537,
    q_moduli_count: 60,
    log2_ring_degree: 16,
    use_ram_bytes: 16 << 30
};

fn estimate_error_bits(log2_n: usize, t: i64, d: usize, N: usize, sigma: f64) -> f64 {
    // actual success probability will be significantly higher than that
    const SUCCESS_PROBABILITY_LOWER_BOUND: f64 = 0.5;
    // we assume can-norm coefficients of critical quantity of BV is distributed as `Xij ~ sqrt(N) (t sqrt(n) sigma)`;
    // then we solve Pr[all |Xij| <= q^(1/d)/(tN)] = erf(q^(1/d)/(t sqrt(n) sigma))^(nm)/(t sqrt(N))
    (erf_inv(SUCCESS_PROBABILITY_LOWER_BOUND.powf(1. / (2f64.powi(log2_n as i32) * m as f64))).log2() + (t as f64).log2() + (log2_n as f64) / 2. + sigma.log2()) * d as f64 + (t as f64).log2() + (N as f64).log2() / 2.
}

#[allow(unused)]
fn estimate_parameters_for_N(N: usize) -> DEPIRParams {
    let mut d = 10;
    while usize::try_from(binomial(m + d, m)).unwrap() < N {
        d += 1;
    }
    let mut log2_n = 13;
    let q_bits = |log2_n: usize| (220 << (log2_n - 13)) as f64;
    let expected_error_bits = |log2_n: usize| estimate_error_bits(log2_n, 65537, d, N, SIGMA);
    while q_bits(log2_n) < expected_error_bits(log2_n) {
        log2_n += 1;
    }
    let mut result = DEPIRParams { t: 65537, d: d, q_moduli_count: 0, log2_ring_degree: log2_n, use_ram_bytes: 24 << 30 };
    let mut current = 0f64;
    let mut toplevel_primes = result.ashe_params().available_toplevel_primes();
    while current < expected_error_bits(log2_n) {
        result.q_moduli_count += 1;
        current += (toplevel_primes.next().unwrap() as f64).log2();
    }
    return result;
}

fn main() {
    let mut rng = StdRng::from_seed([0; 32]);
    let mut depir = estimate_parameters_for_N(1 << 17);
    depir.use_ram_bytes = 24 << 30;
    let pir_index = 5;

    println!("N = {}", depir.N());
    println!("d = {}", depir.d);
    println!("n = 2^{}", depir.ashe_params().log2_ring_degree);
    println!("t = {}", depir.top_level_primes().iter().map(|Fp| (*Fp.modulus() as f64).log2()).max_by(f64::total_cmp).unwrap());
    println!("r = {}", depir.q_moduli_count);
    println!("log2(q) = {}", BigIntRing::RING.abs_log2_ceil(&BigIntRing::RING.prod(depir.top_level_primes().iter().map(|Fp| int_cast(*Fp.modulus(), BigIntRing::RING, StaticRing::<i64>::RING)))).unwrap());
    println!("log2(est_error) = {}", estimate_error_bits(depir.log2_ring_degree, depir.t, depir.d, depir.N(), SIGMA));
    let plan = depir.evaluation_plan();
    plan.print();

    let database = depir.boring_database();

    println!("Found database");

    let (poly_hash, global_poly, all_polynomials) = depir.compute_all_polynomials(database);
    
    // println!("Found polys");

    // let poly_ring: MultivariatePolyRingImpl<_, _, _, m> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    // println!("Actual interpolated polynomial degree: {}", poly_ring.terms(&poly_as_ring_el::<_, _, _, m>(&global_poly, depir.d, &poly_ring, &StaticRing::<i64>::RING.identity())).map(|(_, mon)| mon.deg()).max().unwrap());
    // println!("Database code: {}", poly_hash);

    // depir.prepare_preprocessing(poly_hash.as_str(), all_polynomials);

    // println!("Prepared preprocessing");

    // depir.call_preprocessor(&poly_hash);

    let evaluator = depir.create_evaluator(poly_hash.as_str());

    let ashe = depir.ashe_params();
    let interpolation_grid = depir.interpolation_grid();
    let C = ashe.create_ciphertext_ring();
    let P = ashe.create_plaintext_ring();
    let Zt_to_P = P.inclusion().compose(P.base_ring().into_can_hom(depir.Zt()).ok().unwrap());
    let sk = ashe.sample_sk(&C, &mut rng);
    let encoded_index = interpolation_grid.point_at_index(depir.d, pir_index);
    let encrypted_encoded_index: [_; m] = std::array::from_fn(|k| ashe.encrypt(&C, &P, &mut rng, &Zt_to_P.map(encoded_index[k]), &sk));

    println!("Started PIR evaluation at {:?}", SystemTime::now());
    let start = Instant::now();
    let encrypted_result = depir.evaluate(&C, &evaluator, &encrypted_encoded_index);
    let end = Instant::now();
    println!("Replied to PIR query in {} ms", (end - start).as_millis());
    println!("Performed {} RAM queries", PERFORMED_RAM_QUERIES.load(std::sync::atomic::Ordering::SeqCst));
    println!("Performed {} Disk queries", PERFORMED_DISK_QUERIES.load(std::sync::atomic::Ordering::SeqCst));

    let result = ashe.decrypt(&C, &P, encrypted_result.into_iter().rev(), &sk);
    P.println(&result);

    print_disk_read_stats()
}

#[cfg(test)]
use polynomial::splitting::diff;
#[cfg(test)]
use zn_eval::TestPolynomialTracker;
#[cfg(test)]
use crate::zn_eval::read_ram::*;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::multivariate::ordered::MultivariatePolyRingImpl;
#[cfg(test)]
use feanor_math::rings::multivariate::{DegRevLex, MultivariatePolyRingStore};
#[cfg(test)]
use feanor_math::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use feanor_math::rings::poly::PolyRingStore;
#[cfg(test)]
use feanor_math::vector::vec_fn::VectorFn;
#[cfg(test)]
use zn_eval::ZnEvaluator;
#[cfg(test)]
use polynomial::interpolate::evaluate_poly;

#[ignore]
#[test]
fn test_scheme() {
    let mut rng = StdRng::from_seed([0; 32]);
    let depir = TEST_PARAMS;
    depir.evaluation_plan().print();

    let pir_index = 3;
    // prepare datbase & preprocessing data
    let Zt = depir.Zt();
    let mod_t = Zt.can_hom(&StaticRing::<i64>::RING).unwrap();
    let database = depir.boring_database();
    let (poly_identifier, interpolation_poly, all_polynomials) = depir.compute_all_polynomials(database);
    // assert_eq!("GJVxgA", poly_identifier);
    let all_polynomials = all_polynomials.collect::<Vec<_>>();
    depir.prepare_preprocessing(&poly_identifier, all_polynomials.iter().cloned());
    depir.call_preprocessor(poly_identifier.as_str());
    let evaluator = depir.create_evaluator(&poly_identifier);

    // prepare query
    let interpolation_grid = depir.interpolation_grid();
    let ashe = depir.ashe_params();
    let C = ashe.create_ciphertext_ring();
    let P = ashe.create_plaintext_ring();
    let Zt_to_P = P.inclusion().compose(P.base_ring().into_can_hom(depir.Zt()).ok().unwrap());
    let sk = ashe.sample_sk(&C, &mut rng);
    let encoded_index = interpolation_grid.point_at_index(depir.d, pir_index);
    let encrypted_encoded_index: [_; m] = std::array::from_fn(|k| ashe.encrypt(&C, &P, &mut rng, &Zt_to_P.map(encoded_index[k]), &sk));
    for i in 0..m {
        assert_el_eq!(&P, &P.inclusion().map(encoded_index[i]), &ashe.decrypt(&C, &P, [C.clone_el(ashe.ciphertext_a(&C, &encrypted_encoded_index[i])), ashe.ciphertext_b_expanded(&C, &encrypted_encoded_index[i])].into_iter(), &sk));
    }
    let encrypted_result = depir.evaluate(&C, &evaluator, &encrypted_encoded_index);

    // check that computation of encrypted_result was correct
    let CY = DensePolyRing::new(&C, "Y");
    let point = std::array::from_fn::<_, m, _>(|k| CY.from_terms([(ashe.ciphertext_b_expanded(&C, &encrypted_encoded_index[k]), 0), (C.clone_el(ashe.ciphertext_a(&C, &encrypted_encoded_index[k])), 1)].into_iter()));
    let expected = evaluate_poly((&interpolation_poly[..]).as_el_fn(StaticRing::<i64>::RING), depir.d, &point[..], &CY.inclusion().compose(C.inclusion()).compose(C.base_ring().can_hom(&StaticRing::<i64>::RING).unwrap()));
    for i in 0..encrypted_result.len() {
        assert_el_eq!(&C, &expected[i], &encrypted_result[i]);
    }

    // check that decryption is correct
    let result = ashe.decrypt(&C, &P, encrypted_result.into_iter().rev(), &sk);
    assert_el_eq!(&P, &P.int_hom().map(2), &result);
}

#[ignore]
#[test]
fn test_direct_computation() {
    
    let mut rng = StdRng::from_seed([0; 32]);
    let mut depir = estimate_parameters_for_N(TEST_PARAMS.N());
    let pir_index = 5;

    let database = depir.boring_database();
    let (poly_hash, global_poly, all_polynomials) = depir.compute_all_polynomials(database);
    let all_polynomials = all_polynomials.collect::<Vec<_>>();

    let ashe = depir.ashe_params();
    let interpolation_grid = depir.interpolation_grid();
    let C = ashe.create_ciphertext_ring();
    let P = ashe.create_plaintext_ring();
    let Zt_to_P = P.inclusion().compose(P.base_ring().into_can_hom(depir.Zt()).ok().unwrap());
    let sk = ashe.sample_sk(&C, &mut rng);
    let encoded_index = interpolation_grid.point_at_index(depir.d, pir_index);
    let encrypted_encoded_index: [_; m] = std::array::from_fn(|k| ashe.encrypt(&C, &P, &mut rng, &Zt_to_P.map(encoded_index[k]), &sk));

    let mut encrypted_result = (0..=depir.d).map(|_| C.zero()).collect::<Vec<_>>();
    let mut b_components: [_; m] = std::array::from_fn(|k| ashe.ciphertext_b_components(&C, &encrypted_encoded_index[k]));
    for (i, Zp) in C.get_ring().rns_base().iter().enumerate() {
        for j in 0..C.rank() {
            let b_index = (0..m).map(|k| {
                let next_component = b_components[k].next().unwrap();
                assert!(next_component == 0 || next_component == 1, "Input ciphertext does not have a {{0, 1}}-CRT element for b.");
                next_component as usize * (1 << k)
            }).sum::<usize>();
            for k in 0..=depir.d {
                let result_value = evaluate_poly(
                    (&all_polynomials[b_index][k][..]).into_fn(), 
                    depir.d - k, 
                    &std::array::from_fn::<_, reduced_m, _>(|l| Zp.checked_div(
                        C.get_ring().fourier_coefficient(i, j, ashe.ciphertext_a(&C, &encrypted_encoded_index[l])),
                        C.get_ring().fourier_coefficient(i, j, ashe.ciphertext_a(&C, &encrypted_encoded_index[m - 1])),
                    ).unwrap())[..], 
                    &Zp.can_hom(&StaticRing::<i64>::RING).unwrap()
                );
                *C.get_ring().fourier_coefficient_mut(i, j, &mut encrypted_result[k]) = result_value;
                Zp.mul_assign(C.get_ring().fourier_coefficient_mut(i, j, &mut encrypted_result[k]), Zp.pow(*C.get_ring().fourier_coefficient(i, j, ashe.ciphertext_a(&C, &encrypted_encoded_index[m - 1])), depir.d - k));
            }
        }
    }
    encrypted_result.reverse();

    let result = ashe.decrypt(&C, &P, encrypted_result.into_iter().rev(), &sk);
    assert_el_eq!(&P, &P.one(), &result);
}

#[ignore]
#[test]
fn test_ram_format_compatible() {
    const test_m: usize = 4;
    let depir = TEST_FORMAT_PARAMS;
    let mut depirk = estimate_parameters_for_N(SSD_PARAMS.N());
    depir.evaluation_plan().print();

    let Zt = depir.Zt();
    let mod_t = Zt.can_hom(&StaticRing::<i64>::RING).unwrap();
    let database = depir.boring_database();
    let (poly_hash, _interpolation_poly, all_polynomials) = depir.compute_all_polynomials(database);

    let all_polynomials = all_polynomials.collect::<Vec<_>>();
    depir.prepare_preprocessing(poly_hash.as_str(), all_polynomials.iter().cloned());
    depir.call_preprocessor(poly_hash.as_str());

    let polynomials = all_polynomials.into_iter().next().unwrap();
    let ring = Zn::new(13 * 13);
    let eval = ReadRamEvaluator::<_, reduced_m>::initialize_from_file(ring, 13, depir.d + 1, format!("evaluations_{}_{}_{}_0_13_const", poly_hash, depir.d, reduced_m).as_str());
    let mut rng = thread_rng();
    
    for _ in 0..100 {
        let mut random_small_element = || ring.int_hom().map((rng.next_u64() % 13) as i32 - 6);
        let point: [_; test_m - 1] = [random_small_element(), random_small_element(), random_small_element()];
        let evaluations = eval.evaluate_sync(point, &TestPolynomialTracker::in_production());
        for (i, e) in evaluations.iter().enumerate() {
            assert_el_eq!(&ring, &evaluate_poly((&polynomials[i][..]).as_el_fn(StaticRing::<i64>::RING).map(|x| ring.int_hom().map(x as i32)), depir.d - i, &point[..], &ring.identity()), e);
        }
    }

    let diff_polynomials = polynomials.iter().enumerate().map(|(i, f)| diff(f, StaticRing::<i64>::RING, reduced_m, depir.d - i, 0)).collect::<Vec<_>>();
    let ring = Zn::new(13);
    let eval = ReadRamEvaluator::<_, reduced_m>::initialize_from_file(ring, 13, depir.d + 1, format!("evaluations_{}_{}_{}_0_13_x0", poly_hash, depir.d, reduced_m).as_str());
    let mut rng = thread_rng();

    for _ in 0..100 {
        let mut random_small_element = || ring.int_hom().map((rng.next_u64() % 13) as i32 - 6);
        let point: [_; test_m - 1] = [random_small_element(), random_small_element(), random_small_element()];
        let evaluations = eval.evaluate_sync(point, &TestPolynomialTracker::in_production());
        for (i, e) in evaluations.iter().enumerate() {
            if i == depir.d {
                assert_el_eq!(&ring, &ring.zero(), e);
            } else {
                assert_el_eq!(&ring, &evaluate_poly((&diff_polynomials[i][..]).as_el_fn(StaticRing::<i64>::RING).map(|x| ring.int_hom().map(x as i32)), depir.d - 1 - i, &point[..], &ring.identity()), e);
            }
        }
    }
}

#[test]
fn test_compute_all_polynomials() {
    let params = TEST_PARAMS;
    let Zt = params.Zt();
    let mod_t = Zt.can_hom(&StaticRing::<i64>::RING).unwrap();
    let database = params.boring_database();
    let (_, interpolation_poly, all_polynomials) = params.compute_all_polynomials(database);
    let all_polynomials = all_polynomials.collect::<Vec<_>>();

    let P_m: MultivariatePolyRingImpl<_, _, _, m> = MultivariatePolyRingImpl::new(StaticRing::<i64>::RING, DegRevLex, default_memory_provider!());
    let all_polynomials = all_polynomials.iter().map(|polys| polys.iter().enumerate().map(|(i, f)| poly_as_ring_el::<_, _, _, m>(f, params.d - i, &P_m, &P_m.base_ring().identity())).collect::<Vec<_>>()).collect::<Vec<_>>();
    let original_poly = poly_as_ring_el::<_, _, _, m>(&interpolation_poly, params.d, &P_m, &P_m.base_ring().identity());
    let PY = DensePolyRing::new(&P_m, "Y");

    for b_index in 0..(1 << m) {
        let b_els: [i64; m] = std::array::from_fn(|i| (b_index >> i) & 1);
        let eval_point: [_; m] = std::array::from_fn(|i| if i + 1 < m { 
            PY.add(PY.int_hom().map(b_els[i] as i32), PY.mul(PY.indeterminate(), PY.inclusion().map(P_m.indeterminate(i)))) 
        } else { 
            PY.add(PY.int_hom().map(b_els[m - 1] as i32), PY.indeterminate())
        });
        let expected = P_m.evaluate(&original_poly, &eval_point, &PY.inclusion().compose(P_m.inclusion()));
        for k in 0..=params.d {
            assert_el_eq!(&P_m, &all_polynomials[b_index as usize][params.d - k], PY.coefficient_at(&expected, k));
        }
    }
}