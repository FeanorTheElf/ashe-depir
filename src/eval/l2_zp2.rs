use std::collections::HashMap;
use std::path::Path;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::cmp::max;

use feanor_math::algorithms::{int_bisect, int_factor};
use feanor_math::primitive_int::StaticRing;
use feanor_math::rings::zn::zn_64::Zn;
use feanor_math::rings::zn::ZnRingStore;
use feanor_math::vector::VectorView;
use selfref::Holder;
use crate::eval::{read_disk, read_ram};
use crate::polynomial::convert_lex_to_deglex;
use crate::polynomial::io::{write_file, PolyCoeffInt};
use crate::polynomial::splitting::diff;
use crate::strategy::{binomial, crt_reduction_step, InputStats, ModulusInfo, PolynomialStats};
use crate::eval::read_ram::ReadRamEvaluator;
use crate::{component_count, reduced_m, DATASTRUCTURE_PATH, ASHEEvaluator, DiskPrime, Level2Prime, RamDiskZp2ASHEEvaluator, RamPrime, TopLevelPrime, PREPROCESSOR_PATH};

use crate::eval::zp2_evaluator::Zp2Evaluator;

use super::crt_evaluator::CRTEvaluator;
use super::read_disk::ReadDiskEvaluator;
use super::{available_primes, CiphertextRingEvaluatorStructKey, RNSComponentEvaluator, TestPolynomialTracker, ZnEvaluator};

pub struct StrategyL2Zp2;

#[derive(Clone)]
pub struct L2Zp2Plan {
    pub level0_moduli: HashMap<i64, (ModulusInfo, [ModulusInfo; reduced_m])>,
    pub level1_moduli: HashMap<i64, (ModulusInfo, [ModulusInfo; reduced_m])>,
    pub level2_primes: HashMap<i64, ModulusInfo>
}

impl L2Zp2Plan {
    
    pub fn total_ram_queries(&self) -> usize {
        let mut result = 0;
        for infos in self.level0_moduli.values() {
            result += infos.0.queries;
            for info in &infos.1 {
                result += info.queries;
            }
        }
        return result;
    }

    pub fn total_disk_queries(&self) -> usize {
        let mut result = 0;
        for infos in self.level1_moduli.values() {
            result += infos.0.queries;
            for info in &infos.1 {
                result += info.queries;
            }
        }
        return result;
    }

    pub fn total_ram_storage(&self) -> usize {
        let mut result = 0;
        for infos in self.level0_moduli.values() {
            result += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(infos.0);
            for info in &infos.1 {
                result += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(*info);
            }
        }
        return result;
    }

    pub fn total_disk_storage(&self) -> usize {
        let mut result = 0;
        for infos in self.level1_moduli.values() {
            result += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(infos.0);
            for info in &infos.1 {
                result += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(*info);
            }
        }
        return result;
    }

    pub fn print(&self) {
        let mut level0 = self.level0_moduli.iter().collect::<Vec<_>>();
        level0.sort_unstable_by_key(|(p, _)| *p);
        let mut level1 = self.level1_moduli.iter().collect::<Vec<_>>();
        level1.sort_unstable_by_key(|(p, _)| *p);
        let mut level2 = self.level2_primes.iter().collect::<Vec<_>>();
        level2.sort_unstable_by_key(|(p, _)| *p);

        let ceil_div = |a: usize, b: usize| if a == 0 { 0 } else { (a - 1) / b + 1 };

        let mut total_size = 0;
        let mut total_queries = 0;
        println!("-------------------------------------------------");
        println!("Stored in RAM");
        for (p, infos) in &level0  {
            println!("{}^2", p);
            println!("  {}", infos.0);
            total_size += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(infos.0);
            total_queries += infos.0.queries;
            for info in &infos.1 {
                println!("  {}", info);
                total_size += ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(*info);
                total_queries += info.queries;
            }
        }
        println!("-------------------------------------------------");
        println!("{} primes, {} Mi accesses, {} MB", level0.len(), ceil_div(total_queries, 1 << 20), ceil_div(total_size, 1 << 20));

        let mut total_size = 0;
        let mut total_queries = 0;
        println!("-------------------------------------------------");
        println!("Stored on Disk");
        for (p, infos) in &level1  {
            println!("{}^2", p);
            println!("  {}", infos.0);
            total_size += read_disk::ReadDiskEvaluator::<DiskPrime, reduced_m>::expected_size_in_bytes(infos.0);
            total_queries += infos.0.queries;
            for info in &infos.1 {
                println!("  {}", info);
                total_size += read_disk::ReadDiskEvaluator::<DiskPrime, reduced_m>::expected_size_in_bytes(*info);
                total_queries += info.queries;
            }
        }
        println!("-------------------------------------------------");
        println!("{} primes, {} Mi accesses, {} MB", level1.len(), ceil_div(total_queries, 1 << 20), ceil_div(total_size, 1 << 20));

        let mut total_queries = 0;
        println!("-------------------------------------------------");
        println!("To reduce");
        for (_p, info) in &level2  {
            println!("{}", info);
            total_queries += info.queries;
        }
        println!("-------------------------------------------------");
        println!("{} primes, {} Mi reduction ops", level2.len(), ceil_div(total_queries, 1 << 20));
    }

    ///
    /// We assume `level2_primes` have correct query information; All the other ones are zero
    /// 
    fn compute_poly_components_and_queries(&mut self, primes: &[TopLevelPrime], query_count: usize, poly_params: &[PolynomialStats]) {
        let mut level01_moduli = self.level0_moduli.values().chain(self.level1_moduli.values()).map(|info| info.0.modulus.characteristic).collect::<Vec<_>>();
        level01_moduli.sort_unstable();
        let mut level2_moduli = self.level2_primes.keys().copied().collect::<Vec<_>>();
        level2_moduli.sort_unstable();

        let level3_reduction_results = crt_reduction_step(
            &primes.iter().map(|Fp| (InputStats { characteristic: *Fp.modulus(), input_size_bound: (*Fp.modulus() as usize - 1) / 2 + 1 }, query_count)).collect::<Vec<_>>(), 
            poly_params, 
            level01_moduli.iter().chain(level2_moduli.iter()).copied()
        );
        let used_level2_primes = level3_reduction_results.iter()
            .filter(|(n, _)| self.level2_primes.contains_key(n))
            .inspect(|(n, _)| assert!(int_factor::is_prime_power(&StaticRing::<i64>::RING, n).unwrap().1 == 1))
            .map(|(_, info)| (info.modulus, info.queries))
            .filter(|(_, queries)| *queries > 0)
            .collect::<Vec<_>>();

        for (modulus, queries) in &used_level2_primes {
            self.level2_primes.get_mut(&modulus.characteristic).unwrap().queries = *queries;
            self.level2_primes.get_mut(&modulus.characteristic).unwrap().max_index_polynomial_part = level3_reduction_results.get(&modulus.characteristic).unwrap().max_index_polynomial_part;
        }

        self.level2_primes.retain(|_, info| info.queries > 0);

        let level2_reduction_results = crt_reduction_step(&used_level2_primes, poly_params, level01_moduli.iter().copied());

        for (p, infos) in self.level0_moduli.iter_mut() {
            assert_eq!(0, infos.0.queries);
            let queries_from_l2 = level2_reduction_results.get(&(p * p)).map(|info| info.queries).unwrap_or(0);
            let max_index_polynomial_part_from_l2 = level2_reduction_results.get(&(p * p)).map(|info| info.max_index_polynomial_part).unwrap_or(0);
            infos.0.queries = queries_from_l2 + level3_reduction_results.get(&(p * p)).unwrap().queries;
            infos.0.max_index_polynomial_part = max(max_index_polynomial_part_from_l2, level3_reduction_results.get(&(p * p)).unwrap().max_index_polynomial_part);
            for i in 0..reduced_m {
                infos.1[i].queries = queries_from_l2 + level3_reduction_results.get(&(p * p)).unwrap().queries;
                infos.1[i].max_index_polynomial_part = max(max_index_polynomial_part_from_l2, level3_reduction_results.get(&(p * p)).unwrap().max_index_polynomial_part);
            }
        }
        for (p, infos) in self.level1_moduli.iter_mut() {
            assert_eq!(0, infos.0.queries);
            let queries_from_l2 = level2_reduction_results.get(&(p * p)).map(|info| info.queries).unwrap_or(0);
            let max_index_polynomial_part_from_l2 = level2_reduction_results.get(&(p * p)).map(|info| info.max_index_polynomial_part).unwrap_or(0);
            infos.0.queries = queries_from_l2 + level3_reduction_results.get(&(p * p)).unwrap().queries;
            infos.0.max_index_polynomial_part = max(max_index_polynomial_part_from_l2, level3_reduction_results.get(&(p * p)).unwrap().max_index_polynomial_part);
            for i in 0..reduced_m {
                infos.1[i].queries = queries_from_l2 + level3_reduction_results.get(&(p * p)).unwrap().queries;
                infos.1[i].max_index_polynomial_part = max(max_index_polynomial_part_from_l2, level3_reduction_results.get(&(p * p)).unwrap().max_index_polynomial_part);
            }
        }
    }
}

impl StrategyL2Zp2 {

    fn check_matches_plan(&self, evaluator: &RamDiskZp2ASHEEvaluator, plan: &L2Zp2Plan, initial_queries: usize) {

        let sqrt = |x: i64| int_bisect::root_floor(StaticRing::<i64>::RING, x, 2);

        for b_evaluator in &evaluator.evaluators {
            b_evaluator.as_ref().operate_in(|b_eval| {

                let compute_actual_query_count_multiplier = |modulus: i64| b_eval.level3.iter().map(|eval| {
                        eval.used_base_evaluators().2.iter().filter(|e2| e2.used_base_evaluators().0.iter().any(|e01| *e01.ring().modulus() == modulus) || e2.used_base_evaluators().1.iter().any(|e01| *e01.ring().modulus() == modulus)).count()
                             + if eval.used_base_evaluators().0.iter().any(|e01| *e01.ring().modulus() == modulus) || eval.used_base_evaluators().1.iter().any(|e01| *e01.ring().modulus() == modulus) { 1 } else { 0 }
                    }).sum::<usize>() * initial_queries;

                assert_eq!(plan.level0_moduli.len(), b_eval.level0.len());
                for level0_eval in &b_eval.level0 {
                    let info = plan.level0_moduli.get(&sqrt(*level0_eval.ring().modulus())).unwrap();
                    assert_eq!(info.0.max_index_polynomial_part + 1, level0_eval.poly_count());

                    assert_eq!(info.0.max_index_polynomial_part + 1, level0_eval.low_eval().poly_count());
                    assert_eq!(info.0.queries, compute_actual_query_count_multiplier(info.0.modulus.characteristic));
                    assert_eq!(ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(info.0), level0_eval.low_eval().actual_size_in_bytes() * component_count);
                    for i in 0..info.1.len() {
                        assert_eq!(info.1[i].max_index_polynomial_part + 1, level0_eval.high_evals()[i].poly_count());
                        assert_eq!(info.1[i].queries, compute_actual_query_count_multiplier(info.0.modulus.characteristic));
                        assert_eq!(ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(info.1[i]), level0_eval.high_evals()[i].actual_size_in_bytes() * component_count);
                    }
                }

                assert_eq!(plan.level1_moduli.len(), b_eval.level1.len());
                for level1_eval in &b_eval.level1 {
                    let info = plan.level1_moduli.get(&sqrt(*level1_eval.ring().modulus())).unwrap();
                    assert_eq!(info.0.max_index_polynomial_part + 1, level1_eval.poly_count());

                    assert_eq!(info.0.max_index_polynomial_part + 1, level1_eval.low_eval().poly_count());
                    assert_eq!(info.0.queries, compute_actual_query_count_multiplier(info.0.modulus.characteristic));
                    assert_eq!(ReadDiskEvaluator::<DiskPrime, reduced_m>::expected_size_in_bytes(info.0), level1_eval.low_eval().actual_size_in_bytes() * component_count);
                    for i in 0..info.1.len() {
                        assert_eq!(info.0.max_index_polynomial_part + 1, level1_eval.high_evals()[i].poly_count());
                        assert_eq!(info.1[i].queries, compute_actual_query_count_multiplier(info.0.modulus.characteristic));
                        assert_eq!(ReadDiskEvaluator::<DiskPrime, reduced_m>::expected_size_in_bytes(info.1[i]), level1_eval.high_evals()[i].actual_size_in_bytes() * component_count);
                    }
                }

                assert_eq!(plan.level2_primes.len(), b_eval.level2.len());
                for level2_eval in &b_eval.level2 {
                    let info = plan.level2_primes.get(level2_eval.ring().modulus()).unwrap();
                    assert_eq!(info.max_index_polynomial_part + 1, level2_eval.poly_count());
                }
            });
        }
    }
    
    fn create_part_evaluator<'a, E0, E1, F0, F1>(
        &self,
        primes3: Vec<TopLevelPrime>, 
        poly_params: &[PolynomialStats], 
        plan: &L2Zp2Plan,
        mut create_ram_eval: F0,
        mut create_disk_eval: F1
    ) -> Pin<Box<Holder<'a, CiphertextRingEvaluatorStructKey<Zp2Evaluator<RamPrime, RamPrime, E0, E0, reduced_m>, Zp2Evaluator<DiskPrime, DiskPrime, E1, E1, reduced_m>>>>>    
        where E0: 'a + ZnEvaluator<RamPrime, reduced_m>,
            E1: 'a + ZnEvaluator<DiskPrime, reduced_m>,
            F0: FnMut(i64, RamPrime, usize, Option<usize>) -> E0,
            F1: FnMut(i64, DiskPrime, usize, Option<usize>) -> E1
    {
        let mut l0_primes = plan.level0_moduli.keys().copied().collect::<Vec<_>>();
        l0_primes.sort_unstable();
        let l0_moduli = l0_primes.iter().map(|p| <RamPrime>::new((p * p).try_into().unwrap())).collect::<Vec<_>>();

        let mut l1_primes = plan.level1_moduli.keys().copied().collect::<Vec<_>>();
        l1_primes.sort_unstable();
        let l1_moduli = l1_primes.iter().map(|p| <DiskPrime>::new((p * p).try_into().unwrap())).collect::<Vec<_>>();

        let mut l2_primes = plan.level2_primes.keys().copied().collect::<Vec<_>>();
        l2_primes.sort_unstable();
        let l2_moduli = l2_primes.iter().map(|p| <Level2Prime>::new((*p).try_into().unwrap())).collect::<Vec<_>>();

        let result = Box::pin(Holder::<'_, CiphertextRingEvaluatorStructKey<Zp2Evaluator<RamPrime, RamPrime, E0, E0, reduced_m>, Zp2Evaluator<DiskPrime, DiskPrime, E1, E1, reduced_m>>>::new_with(
            |result| result.build(RNSComponentEvaluator {
                level0: l0_primes.iter().zip(l0_moduli.iter())
                    .map(|(p, Zp2): (&_, &Zn)| Zp2Evaluator::init(
                        *Zp2, 
                        plan.level0_moduli.get(&p).unwrap().0.max_index_polynomial_part + 1,
                        create_ram_eval(*p, *Zp2, plan.level0_moduli.get(&p).unwrap().0.max_index_polynomial_part + 1, None),
                        std::array::from_fn(|k| create_ram_eval(*p, *Zp2, plan.level0_moduli.get(&p).unwrap().1[k].max_index_polynomial_part + 1, Some(k)))
                    ))
                    .collect(),
                level1: l1_primes.iter().zip(l1_moduli.iter())
                    .map(|(p, Zp2): (&_, &Zn)| Zp2Evaluator::init(
                        *Zp2, 
                        plan.level1_moduli.get(&p).unwrap().0.max_index_polynomial_part + 1,
                        create_disk_eval(*p, *Zp2, plan.level1_moduli.get(&p).unwrap().0.max_index_polynomial_part + 1, None),
                        std::array::from_fn(|k| create_disk_eval(*p, *Zp2, plan.level1_moduli.get(&p).unwrap().1[k].max_index_polynomial_part + 1, Some(k)))
                    ))
                    .collect(),
                level2: l2_moduli.iter().map(|_| CRTEvaluator::uninitialized()).collect(),
                level3: primes3.iter().map(|_| CRTEvaluator::uninitialized()).collect(),
                primes0: l0_moduli,
                primes1: l1_moduli,
                primes2: l2_moduli,
                primes3,
            }
        )));

        result.as_ref().operate_in(|e| e.get_ref().init_level_2(&poly_params, &plan.level2_primes));
        result.as_ref().operate_in(|e| e.get_ref().init_level_3(&poly_params));
        return result;
    }

    pub fn plan(
        &self, 
        primes: &[TopLevelPrime], 
        poly_params: &[PolynomialStats], 
        query_count: usize, 
        ram_size_limit: usize
    ) -> L2Zp2Plan {

        // There are many ways how we get basically the same data, but with tiny differences. In particular, the question 
        // is exactly how many primes we store in RAM/Disk. The approach we use here is to first CRT-reduce the input to
        // primes, and then reduce those primes to prime squares. This gives us all prime squares that we will store.
        //
        // However, we can then use these prime squares already as additional moduli for the first reduction step, thus giving
        // slightly less level 2 primes. We update the level 2 primes, but do not reduce them again (since this might then lead
        // to removing level 0/1 primes that we already used in the first reduction). This seems to give very good results, although
        // it might not lead to the least space necessary. 
        // 
        // We also defer the splitting between RAM/Disk to the very end, as then we have all the required data about the size of
        // the involved tables available.

        let fst_step_primes = crt_reduction_step(
            &primes.iter().map(|Fp| (InputStats { characteristic: *Fp.modulus(), input_size_bound: (*Fp.modulus() - 1) as usize / 2 + 1 }, 0)).collect::<Vec<_>>(), 
            poly_params, 
            available_primes()
        );

        let snd_step_moduli = crt_reduction_step(
            &fst_step_primes.iter().map(|(_p, info)| (info.modulus, 0)).collect::<Vec<_>>(), 
            poly_params, 
            available_primes().map(|p| p * p),
        );

        let level01_moduli = snd_step_moduli.into_iter()
            .map(|(p_sqr, info)| (int_bisect::root_floor(StaticRing::<i64>::RING, p_sqr, 2), info))
            .map(|(p, _)| (
                    ModulusInfo {
                        modulus: InputStats { 
                            characteristic: p * p, 
                            input_size_bound: p as usize / 2
                        },
                        max_index_polynomial_part: 0,
                        queries: 0
                    },
                    std::array::from_fn(|_i| ModulusInfo {
                        modulus: InputStats { 
                            characteristic: p, 
                            input_size_bound: p as usize / 2
                        },
                        max_index_polynomial_part: 0,
                        queries: 0
                    })
                )
            )
            .map(|info| (info.1[0].modulus.characteristic, info))
            .collect::<HashMap<_, _>>();

        let level2_primes = fst_step_primes.iter()
            .filter(|(p, _)| !level01_moduli.contains_key(p))
            .map(|(p, info)| (*p, *info))
            .collect::<HashMap<_, _>>();

        let mut result = L2Zp2Plan { 
            level0_moduli: HashMap::new(), 
            level1_moduli: level01_moduli, 
            level2_primes: level2_primes
        };
        
        result.compute_poly_components_and_queries(primes, query_count, poly_params);

        let ram_size = |info: &(ModulusInfo, [ModulusInfo; reduced_m])| read_ram::ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(info.0)
             + info.1.iter().map(|info| read_ram::ReadRamEvaluator::<RamPrime, reduced_m>::expected_size_in_bytes(*info)).sum::<usize>();

        let mut level01_data = result.level1_moduli.drain().collect::<Vec<_>>();
        level01_data.sort_by_key(|(p, _)| *p);
        let mut level01_data = level01_data.into_iter().peekable();

        let mut current_ram_size = 0;
        while level01_data.peek().is_some() && current_ram_size + ram_size(&level01_data.peek().unwrap().1) <= ram_size_limit {
            current_ram_size = current_ram_size + ram_size(&level01_data.peek().unwrap().1);
            result.level0_moduli.insert(level01_data.peek().unwrap().0, level01_data.next().unwrap().1);
        }
        result.level1_moduli.extend(level01_data);

        return result;
    }

    pub fn create(
        &self, 
        primes: Vec<TopLevelPrime>, 
        poly_params: &[PolynomialStats], 
        poly_hash: &str, 
        query_count: usize, 
        ram_size_limit: usize
    ) -> RamDiskZp2ASHEEvaluator<'static> {
        let d = poly_params[0].D;
        let plan = self.plan(&primes, poly_params, query_count, ram_size_limit);

        // this weird cloning is necessary because of a compiler bug - for some reason, the fact that the output
        // evaluator lives for 'static makes the rust compiler ask that also poly_hash should live for 'static
        let poly_hash = poly_hash.to_owned();

        let plan_copy = plan.clone();
        let create_part_eval = move |a_index: usize| {
            let poly_hash1 = poly_hash.clone();
            let poly_hash2 = poly_hash.clone();
            self.create_part_evaluator(
                primes.clone(), 
                poly_params, 
                &plan,
                move |p: i64, Zp2: RamPrime, poly_count: usize, split_var_index: Option<usize>| if let Some(split_var_index) = split_var_index {
                    ReadRamEvaluator::<_, reduced_m>::initialize_from_file(<RamPrime>::new(p as u64), p, poly_count, format!("evaluations_{}_{}_{}_{}_{}_x{}", poly_hash1.as_str(), d, reduced_m, a_index, p, split_var_index).as_str())
                } else {
                    ReadRamEvaluator::<_, reduced_m>::initialize_from_file(Zp2, p, poly_count, format!("evaluations_{}_{}_{}_{}_{}_const", poly_hash1.as_str(), d, reduced_m, a_index, p).as_str())
                },
                move |p: i64, Zp2: DiskPrime, poly_count: usize, split_var_index: Option<usize>| if let Some(split_var_index) = split_var_index {
                    ReadDiskEvaluator::<_, reduced_m>::initialize_for_file(<DiskPrime>::new(p as u64), p, poly_count, format!("evaluations_{}_{}_{}_{}_{}_x{}", poly_hash2.as_str(), d, reduced_m, a_index, p, split_var_index).as_str())
                } else {
                    ReadDiskEvaluator::<_, reduced_m>::initialize_for_file(Zp2, p, poly_count, format!("evaluations_{}_{}_{}_{}_{}_const", poly_hash2.as_str(), d, reduced_m, a_index, p).as_str())
                }
            )
        };
        
        let result = ASHEEvaluator {
            evaluators: (0..component_count).map(create_part_eval).collect(),
            d: d,
            test_polynomial_tracker: (0..component_count).map(|_| TestPolynomialTracker::in_production()).collect::<Vec<_>>()
        };
        self.check_matches_plan(&result, &plan_copy, query_count);
        return result;
    }

    pub fn prepare_preprocessing<I>(&self,
        _primes: Vec<TopLevelPrime>, 
        poly_params: &[PolynomialStats], 
        poly_hash: &str, 
        polynomials: I,
        recompute_everything: bool
    )
        where I: ExactSizeIterator + Iterator<Item = Vec<Vec<i64>>>
    {
        assert_eq!(component_count, polynomials.len());
        for (b_index, b_polynomials) in polynomials.enumerate() {
            let d = poly_params[0].D;
            assert_eq!(poly_params.len(), b_polynomials.len());
            for (j, poly) in b_polynomials.iter().enumerate() {
                assert_eq!(d - j, poly_params[j].D);
                assert_eq!(usize::try_from(binomial(reduced_m + d - j, reduced_m)).unwrap(), poly.len());
                if j < d {
                    for k in 0..reduced_m {
                        let mut deglex_order = (0..binomial(reduced_m + d - j - 1, reduced_m)).map(|_| 0).collect::<Vec<_>>();
                        convert_lex_to_deglex(reduced_m, d - j - 1, diff(&poly, StaticRing::<i64>::RING, reduced_m, d - j, k).into_iter(), &mut deglex_order);
                        let filename = format!("polynomial_{}_{}_{}_{}_{}_x{}", poly_hash, d, reduced_m, b_index, j, k);
                        write_file::<_, PolyCoeffInt>(deglex_order.into_iter(), &filename, recompute_everything);
                    }
                } else {
                    for k in 0..reduced_m {
                        write_file::<_, PolyCoeffInt>([].into_iter(), format!("polynomial_{}_{}_{}_{}_{}_x{}", poly_hash, d, reduced_m, b_index, j, k).as_str(), recompute_everything);
                    }
                }
                let mut deglex_order = (0..poly.len()).map(|_| 0).collect::<Vec<_>>();
                convert_lex_to_deglex(reduced_m, d - j, poly.iter().copied(), &mut deglex_order);
                write_file::<_, PolyCoeffInt>(deglex_order.into_iter(), format!("polynomial_{}_{}_{}_{}_{}_const", poly_hash, d, reduced_m, b_index, j).as_str(), recompute_everything);
            }
        }
    }

    pub fn call_preprocessor(&self,
        primes: Vec<TopLevelPrime>, 
        poly_params: &[PolynomialStats], 
        poly_hash: &str,
        recompute_everything: bool
    ) {
        let d = poly_params[0].D;
        let plan = self.plan(&primes, poly_params, 1, 0);
        for (p, infos) in plan.level0_moduli.iter().chain(plan.level1_moduli.iter()) {
            if recompute_everything || !Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_const", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, 0, p).as_str()).exists() {
                assert!((0..component_count).all(|i| !Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_const", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, i, p).as_str()).exists()));
                let result = Command::new(PREPROCESSOR_PATH)
                    .arg(poly_hash)
                    .arg(format!("{}", d))
                    .arg(format!("{}", infos.0.max_index_polynomial_part + 1))
                    .arg(format!("{}", *p))
                    .arg("const")
                    .arg(format!("{}", component_count))
                    .arg(format!("{}", *p * *p))
                    .stdout(Stdio::inherit())
                    .stderr(Stdio::inherit())
                    .output()
                    .unwrap();
                assert!(result.status.success());
            } else {
                assert!((0..component_count).all(|i| Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_const", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, i, p).as_str()).exists()));
                // println!("Already found {}evaluations_{}_{}_{}_{}_{}_const", PATH, poly_hash, d, reduced_m, 0, p);
            }
            for (k, info) in infos.1.iter().enumerate() {
                if recompute_everything || !Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_x{}", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, 0, p, k).as_str()).exists() {
                    assert!((0..component_count).all(|i| !Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_x{}", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, i, p, k).as_str()).exists()));
                    let result = Command::new(PREPROCESSOR_PATH)
                        .arg(poly_hash)
                        .arg(format!("{}", d))
                        .arg(format!("{}", info.max_index_polynomial_part + 1))
                        .arg(format!("{}", *p))
                        .arg(format!("{}", k))
                        .arg(format!("{}", component_count))
                        .arg(format!("{}", *p))
                        .stdout(Stdio::inherit())
                        .stderr(Stdio::inherit())
                        .output()
                        .unwrap();
                    assert!(result.status.success());
                } else {
                    assert!((0..component_count).all(|i| Path::new(format!("{}evaluations_{}_{}_{}_{}_{}_x{}", DATASTRUCTURE_PATH, poly_hash, d, reduced_m, i, p, k).as_str()).exists()));
                    // println!("Already found {}evaluations_{}_{}_{}_{}_{}_x{}", PATH, poly_hash, d, reduced_m, 0, p, k);
                }
            }
        }
    }

    #[cfg(test)]
    pub fn create_test<'a>(
        /* should not be necessary that self: 'a, but I think this is again the compiler bug */ &'a self, 
        primes: Vec<TopLevelPrime>, 
        d: usize,
        polys: &'a [(Vec<El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, reduced_m>>>, [Vec<El<MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, reduced_m>>>; reduced_m])], 
        poly_ring: &'a MultivariatePolyRingImpl<StaticRing<i64>, DegRevLex, DefaultMemoryProvider, reduced_m>
    ) -> ASHEEvaluator<'a, 
        Zp2Evaluator<RamPrime, RamPrime, TestEvaluator<RamPrime, reduced_m>, TestEvaluator<RamPrime, reduced_m>, reduced_m>, 
        Zp2Evaluator<DiskPrime, DiskPrime, TestEvaluator<DiskPrime, reduced_m>, TestEvaluator<DiskPrime, reduced_m>, reduced_m>
    > {
        for polys_b in polys {
            assert_eq!(d + 1, polys_b.0.len());
        }
        let max_log2_coeff_size = polys.iter().flat_map(|polys_b| polys_b.0.iter().chain(polys_b.1.iter().flat_map(|fs| fs.iter()))).flat_map(|f| poly_ring.terms(f))
            .map(|(c, _)| StaticRing::<i64>::RING.abs_log2_ceil(c).unwrap())
            .max().unwrap();

        for polys_b in polys {
            for (i, f) in polys_b.0.iter().enumerate() {
                assert!(poly_ring.terms(f).map(|(_, mon)| mon.deg()).max().unwrap_or(0) as usize <= d - i);
            }
            for var in 0..reduced_m {
                for (i, f) in polys_b.1[var].iter().enumerate() {
                    assert!(poly_ring.terms(f).map(|(_, mon)| mon.deg()).max().unwrap_or(0) as usize <= (d - i).saturating_sub(1));
                }
            }
        }

        let poly_params = (0..=d).map(|i| PolynomialStats {
            D: d - i,
            m: reduced_m,
            log_coefficient_size: max_log2_coeff_size,
            monomials: usize::try_from(binomial(reduced_m + d - i, reduced_m)).unwrap()
        }).collect::<Vec<_>>();
        
        let plan = self.plan(&primes, &poly_params, 1, 0);
        plan.print();

        let create_part_eval = move |b_index: usize| {
            self.create_part_evaluator(
                primes.clone(), 
                &poly_params, 
                &plan,
                move |p: i64, _Zp2: RamPrime, poly_count: usize, split_var_index: Option<usize>| if let Some(split_var_index) = split_var_index {
                    TestEvaluator::<_, reduced_m>::new(<RamPrime>::new(p as u64), poly_ring, &polys[b_index].1[split_var_index][..poly_count])
                } else {
                    TestEvaluator::<_, reduced_m>::new(<RamPrime>::new((p * p) as u64), poly_ring, &polys[b_index].0[..poly_count])
                },
                move |p: i64, _Zp2: DiskPrime, poly_count: usize, split_var_index: Option<usize>| if let Some(split_var_index) = split_var_index {
                    TestEvaluator::<_, reduced_m>::new(<DiskPrime>::new(p as u64), poly_ring, &polys[b_index].1[split_var_index][..poly_count])
                } else {
                    TestEvaluator::<_, reduced_m>::new(<DiskPrime>::new((p * p) as u64), poly_ring, &polys[b_index].0[..poly_count])
                }
            )
        };
        
        ASHEEvaluator {
            evaluators: (0..(1 << m)).map(create_part_eval).collect(),
            d: d,
            test_polynomial_tracker: (0..component_count).map(|i| TestPolynomialTracker::create_test(poly_ring, &polys[i].0)).collect::<Vec<_>>()
        }
    }
}

#[cfg(test)]
use feanor_math::ring::*;
#[cfg(test)]
use feanor_math::rings::multivariate::ordered::*;
#[cfg(test)]
use feanor_math::rings::multivariate::*;
#[cfg(test)]
use feanor_math::default_memory_provider;
#[cfg(test)]
use feanor_math::mempool::DefaultMemoryProvider;
#[cfg(test)]
use crate::eval::TestEvaluator;
#[cfg(test)]
use feanor_math::rings::poly::dense_poly::DensePolyRing;
#[cfg(test)]
use feanor_math::integer::BigIntRing;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::zn::zn_rns;
#[cfg(test)]
use crate::ashe::CiphertextRing;
#[cfg(test)]
use rand_chacha::ChaCha12Rng;
#[cfg(test)]
use crate::eval::test_poly;
#[cfg(test)]
use feanor_math::rings::zn::zn_64::ZnFastmul;
#[cfg(test)]
use crate::eval::m;
#[cfg(test)]
use feanor_math::integer::IntegerRingStore;
#[cfg(test)]
use feanor_math::homomorphism::Homomorphism;
#[cfg(test)]
use feanor_math::rings::extension::FreeAlgebraStore;
#[cfg(test)]
use feanor_math::rings::poly::PolyRingStore;
#[cfg(test)]
use rand::{RngCore, SeedableRng};
#[cfg(test)]
use feanor_math::divisibility::DivisibilityRingStore;

#[test]
fn test_evaluate() {
    let primes = vec![TopLevelPrime::new(257), TopLevelPrime::new(65537)];
    let (d, Pm, original_poly, processed_polys) = test_poly();

    for polys_b in &processed_polys {
        for (i, poly) in polys_b.iter().enumerate() {
            assert!(Pm.terms(poly).all(|(_, mon)| mon.deg() as usize == d - i));
        }
    }

    let P_red_m: MultivariatePolyRingImpl<_, _, _, reduced_m> = MultivariatePolyRingImpl::new(*Pm.base_ring(), DegRevLex, default_memory_provider!());

    let specialize_last = |f: &_| Pm.evaluate::<_, [_; m], _>(&f, std::array::from_fn(|i| if i < reduced_m { P_red_m.indeterminate(i)} else { P_red_m.one() }), &P_red_m.inclusion());
    let diff = |f, k| P_red_m.from_terms(P_red_m.terms(&f).map(|(c, mon)| (c * mon[k] as i64, Monomial::new(std::array::from_fn(|i| if i == k { mon[i].saturating_sub(1) } else { mon[i] })))));
    let split_processed_polys = processed_polys.iter().map(|polys| (
        polys.iter()
            .map(|f| specialize_last(f))
            .collect::<Vec<_>>(),
        std::array::from_fn(|k| polys.iter()
            .map(|f| specialize_last(f))
            .map(|f| diff(f, k))
            .collect::<Vec<_>>())
    )).collect::<Vec<_>>();

    let evaluator = StrategyL2Zp2.create_test(primes.iter().copied().collect(), d, &split_processed_polys, &P_red_m);
    let d = Pm.terms(&original_poly).map(|(_, mon)| mon.deg()).max().unwrap_or(0) as usize;

    let rns_base = zn_rns::Zn::new(primes.clone(), BigIntRing::RING, default_memory_provider!());
    let C = <CiphertextRing as RingStore>::Type::new(rns_base.clone(), rns_base.get_ring().iter().map(|Fp| ZnFastmul::new(*Fp)).collect(), 3, default_memory_provider!());

    let CY = DensePolyRing::new(&C, "Y");

    let mut rng = ChaCha12Rng::from_seed([0; 32]);

    let a_els: [_; m] = std::array::from_fn(|_| {
        let mut result = C.get_ring().sample_uniform(|| rng.next_u64());
        while !C.is_unit(&result) {
            result = C.get_ring().sample_uniform(|| rng.next_u64());
        }
        return result;
    });
    let b_els: [_; m] = std::array::from_fn(|_| {
        let mut result = C.zero();
        for (i, Zp) in C.get_ring().rns_base().iter().enumerate() {
            for j in 0..C.rank() {
                *C.get_ring().fourier_coefficient_mut(i, j, &mut result) = Zp.int_hom().map((rng.next_u32() % 2) as i32);
            }
        }
        result
    });

    let b_els_ref = &b_els;
    let C_ref = &C;
    let b_components_its: [_; m] = std::array::from_fn(|k| C_ref.get_ring().rns_base().iter().enumerate().flat_map(
        move |(i, Zp)| (0..C_ref.rank()).map(move |j| Zp.smallest_positive_lift(*C_ref.get_ring().fourier_coefficient(i, j, &b_els_ref[k])) as u8)
    ));
    let evaluation_point: [_; m] = std::array::from_fn(|k| CY.from_terms([(C.clone_el(&b_els[k]), 0), (C.clone_el(&a_els[k]), 1)].into_iter()));

    let actual = evaluator.evaluate(&C, std::array::from_fn(|i| &a_els[i]), b_components_its);
    let expected = Pm.evaluate(&original_poly, evaluation_point, &CY.inclusion().compose(CY.base_ring().inclusion()).compose(rns_base.can_hom(&StaticRing::<i64>::RING).unwrap()));

    // perform some intermediate tests
    for (i, Zp) in C.get_ring().rns_base().iter().enumerate() {
        for j in 0..C.rank() {
            let b_index = (0..m).map(|k| Zp.smallest_positive_lift(*C.get_ring().fourier_coefficient(i, j, &b_els_ref[k])) << k).sum::<i64>() as usize;
            let ZpY = DensePolyRing::new(Zp, "Y");
            let value = &Pm.evaluate(&original_poly, std::array::from_fn::<_, m, _>(|k| ZpY.from_terms([
                (*C.get_ring().fourier_coefficient(i, j, &b_els[k]), 0),
                (*C.get_ring().fourier_coefficient(i, j, &a_els[k]), 1),
            ].into_iter())), &ZpY.inclusion().compose(Zp.can_hom(&StaticRing::<i64>::RING).unwrap()));
            for k in 0..=d {
                // check that processed_polys matches original_poly
                assert_el_eq!(
                    Zp, 
                    ZpY.coefficient_at(&value, k), 
                    &Pm.evaluate::<_, [_; m], _>(&processed_polys[b_index][d - k], std::array::from_fn(|l| *C.get_ring().fourier_coefficient(i, j, &a_els[l])), &Zp.can_hom(&StaticRing::<i64>::RING).unwrap())
                );

                // check that homogeneous evaluation matches split_processed_polys
                assert_el_eq!(
                    Zp, 
                    ZpY.coefficient_at(&value, k), 
                    &Zp.mul(
                        P_red_m.evaluate::<_, [_; reduced_m], _>(&split_processed_polys[b_index].0[d - k], std::array::from_fn(|l| Zp.checked_div(
                            C.get_ring().fourier_coefficient(i, j, &a_els[l]),
                            C.get_ring().fourier_coefficient(i, j, &a_els[m - 1])
                        ).unwrap()), &Zp.can_hom(&StaticRing::<i64>::RING).unwrap()),
                        Zp.pow(*C.get_ring().fourier_coefficient(i, j, &a_els[m - 1]), k)
                    )
                );
                
                // check that total evaluation indeed corresponds to CRT-wise evaluation
                assert_el_eq!(
                    Zp, 
                    ZpY.coefficient_at(&value, k),
                    C.get_ring().fourier_coefficient(i, j, &expected[k])
                );
            }
        }
    }

    // check final resutl
    assert_eq!(actual.len(), d + 1);
    assert!(CY.degree(&expected).unwrap() <= d);

    for i in 0..=d {
        assert_el_eq!(&C, CY.coefficient_at(&expected, i), &actual[i]);
    }
}