use std::cell::RefCell;
use std::{cmp::max, ops::Range};

use feanor_math::divisibility::DivisibilityRingStore;
use feanor_math::field::{Field, FieldStore};
use feanor_math::homomorphism::*;
use feanor_math::iters::{clone_slice, multi_cartesian_product};
use feanor_math::vector::*;
use feanor_math::vector::subvector::*;
use feanor_math::ring::*;

///
/// Iterates over the bounds of a monomial block, resp. a top-level block in the interpolation matrix.
/// 
/// More concretely, given variables `x_1, ..., x_m`, we consider all monomials
/// of degree at most `d` in lex-order, so `1 < x_1 < ... < x_1^d < x_2 < ... < x_2 x_1^(d - 1) < ... < x_m^d`.
/// Now the i-th block is the block of monomials in which `x_m` is taken to the `i`-th power.
/// The elements given by this iterator are tuples `(i, block_index_range)`. 
/// 
#[derive(Clone, Copy)]
pub struct BlockIter {
    d: usize,
    m: usize,
    current_index: usize,
    current_start: usize,
    back_index: usize,
    back_end: usize
}

impl BlockIter {

    pub fn new(d: usize, m: usize) -> Self {
        BlockIter { 
            d: d, 
            m: m, 
            current_index: 0, 
            current_start: 0, 
            back_index: d + 1, 
            back_end: usize::try_from(binomial(d + m, m)).unwrap()
        }
    }
}

impl Iterator for BlockIter {
    
    type Item = (usize, Range<usize>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index < self.back_index {
            let end = self.current_start + binomial(self.d - self.current_index + self.m - 1, self.m - 1) as usize;
            let result = (self.current_index, self.current_start..end);
            self.current_index += 1;
            self.current_start = end;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl DoubleEndedIterator for BlockIter {

    fn next_back(&mut self) -> Option<Self::Item> {
        if self.back_index > self.current_index {
            self.back_index -= 1;
            let start = self.back_end - binomial(self.d - self.back_index + self.m - 1, self.m - 1) as usize;
            let result = (self.back_index, start..self.back_end);
            self.back_end = start;
            return Some(result);
        } else {
            return None;
        }
    }
}

impl ExactSizeIterator for BlockIter {

    fn len(&self) -> usize {
        self.back_index - self.current_index
    }
}

fn get_two<T>(data: &mut [T], fst: Range<usize>, snd: Range<usize>) -> (&mut [T], &mut [T]) {
    assert!(fst.end <= snd.start);
    let (a, b) = data.split_at_mut(fst.end);
    return (&mut a[fst.clone()], &mut b[(snd.start - fst.end)..(snd.end - fst.end)]);
}

///
/// Uses multivariate Horner schema to evaluate the polynomial of degree `d` given
/// by `poly`.
/// 
/// The order of coefficients in `poly` is corresponds to the reverse lexicographic ordering
/// of monomials, as described in more detail in [`InterpolationMatrix`].
/// 
pub fn evaluate_poly<V, W, R, S, H>(poly: W, d: usize, point: V, hom: &H) -> S::Element
    where V: SelfSubvectorView<S::Element> + Clone,
        W: SelfSubvectorFn<R::Element> + Clone,
        R: RingBase,
        S: RingBase,
        H: Homomorphism<R, S>
{
    let m = point.len();
    if m == 0 {
        assert_eq!(poly.len(), 1);
        return hom.map(poly.at(0));
    }
    assert_eq!(poly.len(), usize::try_from(binomial(d + m, m)).unwrap());
    let mut current = hom.codomain().zero();
    for (i, range) in BlockIter::new(d, m).rev() {
        hom.codomain().mul_assign_ref(&mut current, point.at(point.len() - 1));
        hom.codomain().add_assign(&mut current, evaluate_poly(
            poly.clone().subvector(range), 
            d - i, 
            point.clone().subvector(0..(point.len() - 1)), 
            hom
        ));
    }
    return current;
}

///
/// Represents the interpolation matrix in our variant of multivariate total-degree interpolation.
/// In particular, we consider the matrix that represents the evaluation of the multivariate polynomial
/// in `m` variables of total degree `D` with "unknown" coefficients at all points `(p[1, i1], ..., p[m, im])`
/// for `i1 + ... + im <= D`. Here `p` is an arbitrary, given array, but it will usually be initialized by
/// `p[i, j] = j`.
/// 
/// The standard use case is to appropriately construct this matrix and then use [`Self::solve_inplace`] to
/// solve the associated linear system and hence compute the coefficients of an interpolation polynomial.
/// 
/// # Ordering of rows and columns
/// 
/// Since this matrix corresponds to the evaluation map, we have to define an ordering of the monomials
/// `X1^i1 ... Xm^im` with `i1 + ... + im <= D` (input basis), and an ordering of the evaluation points 
/// `(p[1, i1], ..., p[m, im])` with `i1 + ... + im <= D` (output basis). Both of these are isomorphic to
/// tuples `(i1, ..., im)` with `i1 + ... + im <= D`, and we choose both orderings to be the same as the
/// lexicographic order of these tuples, starting the comparison with the *last* tuple index.
/// 
/// In other words, we have
/// `(0, 0, 0) < (1, 0, 0) < (2, 0, 0) < (0, 1, 0) < (1, 1, 0) < (0, 2, 0) < (0, 0, 1) < (1, 0, 1) < ...`.
/// 
pub struct InterpolationMatrix<R, V>
    where V: VectorView<El<R>>,
        R: RingStore
{
    data: Option<(V, Box<InterpolationMatrix<R, V>>)>,
    m: usize,
    ring: R,
    symmetric_poly_eval: RefCell<Vec<Vec<El<R>>>>
}

impl<R, V> InterpolationMatrix<R, V>
    where V: VectorView<El<R>>,
        R: RingStore
{
    pub fn new<I>(grid: I, ring: R) -> Self
        where I: Iterator<Item = V>,
            R: Clone
    {
        return *Self::create_recursive(grid, ring);
    }

    pub fn size(&self, d: usize) -> usize {
        usize::try_from(binomial(self.m + d, self.m)).unwrap()
    }

    pub fn point_at_index(&self, d: usize, index: usize) -> Vec<El<R>> {
        assert!(index < self.size(d));
        if self.m == 0 {
            return Vec::new();
        }
        for (i, indices) in BlockIter::new(d, self.m()) {
            if indices.contains(&index) {
                let mut result = self.recursive().point_at_index(d - i, index - indices.start);
                result.push(self.ring.clone_el(self.current_points().at(i)));
                return result;
            }
        }
        unreachable!()
    }

    fn create_recursive<I>(mut grid: I, ring: R) -> Box<Self>
        where I: Iterator<Item = V>,
            R: Clone
    {
        if let Some(points) = grid.next() {
            let recursive = Self::create_recursive(grid, ring.clone());
            let m = recursive.m + 1;
            Box::new(InterpolationMatrix { 
                symmetric_poly_eval: RefCell::from(vec![(0..=points.len()).map(|_| ring.one()).collect()]),
                data: Some((points, recursive)), 
                m: m, 
                ring: ring, 
            })
        } else {
            Box::new(InterpolationMatrix { data: None, m: 0, ring: ring, symmetric_poly_eval: RefCell::from(Vec::new()) })
        }
    }

    fn m(&self) -> usize {
        self.m
    }

    fn recursive<'a>(&'a self) -> &'a Self {
        &*self.data.as_ref().unwrap().1
    }

    fn current_points<'a>(&'a self) -> &'a V {
        &self.data.as_ref().unwrap().0
    }

    fn complete_symmetric_polynomial(&self, order: usize, vars: usize) -> El<R> {
        {
            let sym_poly_evals = self.symmetric_poly_eval.borrow();
            assert!(vars <= self.current_points().len());
            if order < sym_poly_evals.len() && vars < sym_poly_evals[order].len() {
                return self.ring.clone_el(&sym_poly_evals[order][vars]);
            }
        }
        let result = if vars == 0 {
            self.ring.zero()
        } else {
            self.ring.add(
                self.complete_symmetric_polynomial(order, vars - 1),
                self.ring.mul_ref_snd(
                    self.complete_symmetric_polynomial(order - 1, vars), 
                    self.current_points().at(vars - 1)
                )
            )
        };
        {
            let mut sym_poly_evals = self.symmetric_poly_eval.borrow_mut();
            let new_len = max(sym_poly_evals.len(), order + 1);
            sym_poly_evals.resize_with(new_len, || Vec::new());
            sym_poly_evals[order].push(result);
            debug_assert_eq!(sym_poly_evals[order].len(), vars + 1);
            return self.ring.clone_el(sym_poly_evals[order].last().unwrap());
        }
    }

    fn U_block_factor<S, H>(&self, i: usize, j: usize, hom: &H) -> S::Element
        where S: RingBase,
            H: Homomorphism<R::Type, S>
    {
        if i > j {
            hom.codomain().zero()
        } else {
            hom.codomain().mul(
                hom.codomain().prod(Iterator::map(0..i, |k| 
                    hom.map(self.ring.sub_ref(self.current_points().at(i), self.current_points().at(k)))
                )),
                hom.map(self.complete_symmetric_polynomial(j - i, i + 1))
            )
        }
    }

    fn L_block_factor<S, H>(&self, i: usize, j: usize, hom: &H) -> S::Element
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        if i < j {
            hom.codomain().zero()
        } else {
            hom.codomain().prod(Iterator::map(0..j, |k| hom.codomain().div(
                &hom.map(self.ring.sub_ref(self.current_points().at(i), self.current_points().at(k))),
                &hom.map(self.ring.sub_ref(self.current_points().at(j), self.current_points().at(k)))
            )))
        }
    }

    fn add_select_lower_degree<S, H>(&self, out_deg: usize, in_deg: usize, out: &mut [S::Element], input: &[S::Element], factor: S::Element, hom: &H) 
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        assert!(in_deg >= out_deg);
        assert_eq!(out.len(), usize::try_from(binomial(out_deg + self.m(), self.m())).unwrap());
        assert_eq!(input.len(), usize::try_from(binomial(in_deg + self.m(), self.m())).unwrap());
        let mut i = 0;
        let mut j = 0;
        // we use cartesian_product here, despite it yielding some unnecessary elements as
        // the iteration order is already correct
        for tuple in multi_cartesian_product(Iterator::map(0..self.m(), |_| 0..=in_deg), clone_slice, |_, x| *x) {
            if tuple.iter().copied().sum::<usize>() <= out_deg {
                hom.codomain().add_assign(&mut out[i], hom.codomain().mul_ref(&input[j], &factor));
                i += 1;
            }
            if tuple.iter().copied().sum::<usize>() <= in_deg {
                j += 1;
            }
        }
        assert_eq!(out.len(), i);
        assert_eq!(input.len(), j);
    }

    pub fn add_mul_A<S, H>(&self, d_rows: usize, d_cols: usize, input: &[S::Element], result: &mut [S::Element], factor: S::Element, hom: &H)
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        debug_assert_eq!(input.len(), usize::try_from(binomial(d_cols + self.m(), self.m())).unwrap());
        debug_assert_eq!(result.len(), usize::try_from(binomial(d_rows + self.m(), self.m())).unwrap());
        let ring = hom.codomain();
        if self.m() == 0 {
            ring.add_assign(
                result.at_mut(0), 
                ring.mul_ref_snd(factor, input.at(0))
            );
            return;
        }
        for (i, rows) in BlockIter::new(d_rows, self.m()) {
            let current_block = &mut result[rows];
            for (j, cols) in BlockIter::new(d_cols, self.m()) {
                self.recursive().add_mul_A(
                    d_rows - i,
                    d_cols - j,
                    input.subvector(cols),
                    current_block,
                    ring.mul_ref_fst(&factor, ring.pow(hom.map_ref(self.current_points().at(i)), j)),
                    hom
                )
            }
        }
    }

    fn solve_U_inplace<S, H>(&self, rhs: &mut [S::Element], d: usize, hom: &H) 
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        let ring = hom.codomain();
        for (j, cols) in BlockIter::new(d, self.m()).rev() {
            self.recursive().solve_A_inplace(
                &mut rhs[cols.clone()], 
                d - j, 
                hom
            );
            let scale = ring.invert(&self.U_block_factor(j, j, hom)).unwrap();
            for i in cols.clone() {
                ring.mul_assign_ref(rhs.at_mut(i), &scale);
            }
            
            for (i, rows) in BlockIter::new(d, self.m()).take(j) {
                let (rhs_current, rhs_pivot) = get_two(rhs, rows, cols.clone());
                self.recursive().add_mul_A(
                    d - i, 
                    d - j, 
                    &rhs_pivot[..], 
                    rhs_current, 
                    ring.negate(self.U_block_factor(i, j, hom)), 
                    hom
                );
            }
        }
    }

    fn solve_L_inplace<S, H>(&self, rhs: &mut [S::Element], d: usize, hom: &H)
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        let ring = hom.codomain();
        for (j, cols) in BlockIter::new(d, self.m()) {
            // the diagonal blocks are identity matrices, 
            // so no pivot division necessary

            for (i, rows) in BlockIter::new(d, self.m()).skip(j + 1) {
                let (rhs_pivot, rhs_current) = get_two(rhs, cols.clone(), rows);
                self.recursive().add_select_lower_degree(
                    d - i, 
                    d - j, 
                    rhs_current, 
                    &rhs_pivot[..], 
                    ring.negate(self.L_block_factor(i, j, hom)), 
                    hom
                );
            }
        }
    }

    fn solve_A_inplace<S, H>(&self, rhs: &mut [S::Element], d: usize, hom: &H)
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        if self.m() == 0 {
            return;
        }
        self.solve_L_inplace(rhs, d, hom);
        self.solve_U_inplace(rhs, d, hom);
    }

    pub fn solve_inplace<S, H>(&self, rhs: &mut [S::Element], d: usize, hom: &H)
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        self.solve_A_inplace(rhs, d, hom);
    }

    pub fn mul<S, H>(&self, rhs: &[S::Element], d: usize, hom: &H, dst: &mut [S::Element])
        where S: Field,
            H: Homomorphism<R::Type, S>
    {
        for i in 0..dst.len() {
            dst[i] = hom.codomain().zero();
        }
        self.add_mul_A(d, d, rhs, dst, hom.codomain().one(), hom);
    }
}

#[cfg(test)]
use feanor_math::rings::zn::zn_static::*;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::vector::vec_fn::*;

use crate::strategy::binomial;

#[test]
fn test_binomial() {
    assert_eq!(6, binomial(6, 1));
    assert_eq!(6, binomial(4, 2));
}

#[test]
fn test_block_iter() {
    let iter = BlockIter::new(4, 3);
    assert_eq!(vec![(0, 0..15), (1, 15..25), (2, 25..31), (3, 31..34), (4, 34..35)], iter.collect::<Vec<_>>());

    let mut it = iter;
    assert_eq!(Some((0, 0..15)), it.next());
    assert_eq!(Some((4, 34..35)), it.next_back());
    assert_eq!(Some((3, 31..34)), it.next_back());
    assert_eq!(Some((1, 15..25)), it.next());
    assert_eq!(Some((2, 25..31)), it.next_back());
    assert_eq!(None, it.next_back());
    assert_eq!(None, it.next());
    
    let mut it = iter;
    assert_eq!(Some((4, 34..35)), it.next_back());
    assert_eq!(Some((3, 31..34)), it.next_back());
    assert_eq!(Some((0, 0..15)), it.next());
    assert_eq!(Some((2, 25..31)), it.next_back());
    assert_eq!(Some((1, 15..25)), it.next());
    assert_eq!(None, it.next());
    assert_eq!(None, it.next_back());
}

#[test]
fn test_add_mul_A() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3]].into_iter(), ring);

    let mut actual = [0, 0, 0, 0];
    grid.add_mul_A(3, 2, &[1, 0, 0], &mut actual, ring.one(), &ring.can_hom(&ring).unwrap());
    assert_eq!([1, 1, 1, 1], actual);

    actual = [0, 0, 0, 0];
    grid.add_mul_A(3, 3, &[0, 1, 0, 0], &mut actual, ring.one(), &ring.can_hom(&ring).unwrap());
    assert_eq!([0, 1, 2, 3], actual);

    actual = [0, 0, 0, 0];
    grid.add_mul_A(3, 4, &[0, 0, 1, 0, 0], &mut actual, ring.one(), &ring.can_hom(&ring).unwrap());
    assert_eq!([0, 1, 4, 9], actual);
}

#[test]
fn test_complete_symmetric_polynomial() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3]].into_iter(), ring);

    let i = ring.int_hom();
    assert_el_eq!(&ring, &i.map(1), &grid.complete_symmetric_polynomial(1, 2));
    assert_el_eq!(&ring, &i.map(1 + 2), &grid.complete_symmetric_polynomial(1, 3));
    assert_el_eq!(&ring, &i.map(1 * 1 + 1 * 2 + 2 * 2), &grid.complete_symmetric_polynomial(2, 3));
}

#[test]
fn test_U_block_factor() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3]].into_iter(), ring);

    let i = ring.int_hom();
    assert_el_eq!(&ring, &i.map(1), &grid.U_block_factor(0, 0, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(0), &grid.U_block_factor(0, 1, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(0), &grid.U_block_factor(0, 2, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(0), &grid.U_block_factor(0, 3, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(1), &grid.U_block_factor(1, 1, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(1), &grid.U_block_factor(1, 2, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(1), &grid.U_block_factor(1, 3, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(2), &grid.U_block_factor(2, 2, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(6), &grid.U_block_factor(2, 3, &ring.can_hom(&ring).unwrap()));
    assert_el_eq!(&ring, &i.map(6), &grid.U_block_factor(3, 3, &ring.can_hom(&ring).unwrap()));
}

#[test]
fn test_solve_A_univariate() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3]].into_iter(), ring);

    let mut actual = [1, 0, 0, 0];
    grid.solve_inplace(&mut actual, 3, &ring.identity());
    assert_eq!([1, 1, 1, 14], actual);
}

#[test]
fn test_solve_A_bivariate() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3], [0, 1, 2, 3]].into_iter(), ring);

    let mut actual = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    grid.solve_inplace(&mut actual, 3, &ring.identity());
    assert_eq!([1, 1, 1, 14, 1, 2, 8, 1, 8, 14], actual);
}

#[test]
fn test_solve_A_triariate() {
    let ring = F17;
    let grid = InterpolationMatrix::new([[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]].into_iter(), ring);

    let mut actual = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    grid.solve_inplace(&mut actual, 3, &ring.identity());
    assert_eq!([1, 1, 1, 14, 1, 2, 8, 1, 8, 14, 1, 2, 8, 2, 16, 8, 1, 8, 8, 14], actual);
}

#[test]
fn test_evaluate_poly() {
    let ring = F17;
    // 1 + 2x^2 + y + xy + x^2y + 2y^2 + 2y^3
    let poly = [1, 0, 2, 0, 1, 1, 1, 2, 0, 2];

    assert_el_eq!(&ring, &10, &evaluate_poly(SubvectorFn::new((&poly[..]).into_fn()), 3, Subvector::new([1, 1]), &ring.identity()));
    assert_el_eq!(&ring, &3, &evaluate_poly(SubvectorFn::new((&poly[..]).into_fn()), 3, Subvector::new([1, 0]), &ring.identity()));
    assert_el_eq!(&ring, &6, &evaluate_poly(SubvectorFn::new((&poly[..]).into_fn()), 3, Subvector::new([0, 1]), &ring.identity()));
    assert_el_eq!(&ring, &16, &evaluate_poly(SubvectorFn::new((&poly[..]).into_fn()), 3, Subvector::new([1, 2]), &ring.identity()));
}