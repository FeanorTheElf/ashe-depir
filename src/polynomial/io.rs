use std::convert::identity;
use std::mem::size_of;
use std::io::{BufReader, BufWriter, Read, Write};
use std::hash::Hasher;
use std::io::Error;
use std::fs::File;
use std::path::Path;

use base64::Engine;
use feanor_math::homomorphism::{CanHomFrom, Homomorphism};
use feanor_math::integer::int_cast;
use feanor_math::primitive_int::*;
use feanor_math::ring::*;
use feanor_math::rings::zn::{ZnRing, ZnRingStore};

pub type EvaluationsUInt = u16;
pub type PolyCoeffInt = i64;

pub trait IOUInt: Sized + Copy + TryFrom<i64> + TryInto<i64> {

    fn to_le_bytes(self) -> [u8; size_of::<Self>()];
    fn from_le_bytes(bytes: [u8; size_of::<Self>()]) -> Self;
    fn max() -> Self;
}

impl IOUInt for PolyCoeffInt {

    fn from_le_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
        Self::from_le_bytes(bytes)
    }

    fn to_le_bytes(self) -> [u8; size_of::<Self>()] {
        Self::to_le_bytes(self)
    }

    fn max() -> Self {
        Self::MAX
    }
}

impl IOUInt for u8 {

    fn from_le_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
        u8::from_le_bytes(bytes)
    }

    fn to_le_bytes(self) -> [u8; size_of::<Self>()] {
        u8::to_le_bytes(self)
    }

    fn max() -> Self {
        u8::MAX
    }
}

impl IOUInt for u16 {

    fn from_le_bytes(bytes: [u8; size_of::<Self>()]) -> Self {
        u16::from_le_bytes(bytes)
    }

    fn to_le_bytes(self) -> [u8; size_of::<Self>()] {
        u16::to_le_bytes(self)
    }

    fn max() -> Self {
        u16::MAX
    }
}

pub fn poly_hash<R: RingStore>(poly: &[El<R>], ring: R) -> String
    where R::Type: HashableElRing
{
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for c in poly {
        ring.hash(c, &mut hasher);
    }
    let hash = hasher.finish();
    let hash_bytes = [(hash & 0xFF) as u8, ((hash >> 8) & 0xFF) as u8, ((hash >> 16) & 0xFF) as u8, (hash >> 24) as u8];
    return base64::engine::general_purpose::STANDARD_NO_PAD.encode(&hash_bytes).replace("/", "_");
}

pub fn decode_zn_el<R: ZnRingStore, Int: IOUInt>(ring: R, data: &[u8]) -> El<R>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        [(); size_of::<Int>()]: Sized
{
    debug_assert_eq!(size_of::<Int>(), data.len());
    ring.can_hom::<StaticRing<i64>>(&StaticRing::<i64>::RING).unwrap().map(Int::from_le_bytes(std::array::from_fn(|i| data[i])).try_into().ok().unwrap())
}

pub fn encode_zn_el<R: ZnRingStore, Int: IOUInt>(ring: R, x: El<R>) -> [u8; size_of::<Int>()]
    where R::Type: ZnRing,
        [(); size_of::<Int>()]: Sized
{
    Int::try_from(int_cast(ring.smallest_positive_lift(x), StaticRing::<i64>::RING, ring.integer_ring())).ok().unwrap().to_le_bytes()
}

pub fn read_file<'a, R: ZnRingStore, Int: IOUInt>(ring: R, filename: &str) -> impl 'a + Iterator<Item = Result<El<R>, Error>>
    where R::Type: ZnRing + CanHomFrom<StaticRingBase<i64>>,
        R: 'a,
        [(); size_of::<Int>()]: Sized
{
    assert!(int_cast(ring.integer_ring().clone_el(ring.modulus()), &StaticRing::<i64>::RING, ring.integer_ring()) <= Int::max().try_into().ok().unwrap() as i64);
    let reader = BufReader::new(File::open(format!("{}{}", DATASTRUCTURE_PATH, filename)).expect(format!("Cannot find file {}{}", DATASTRUCTURE_PATH, filename).as_str()));
    let from_int = ring.into_can_hom(StaticRing::<i64>::RING).ok().unwrap();
    return reader.bytes().scan((0, [0; size_of::<Int>()]), |(i, state), byte| {
        if byte.is_err() {
            return Some(Some(Err(byte.unwrap_err())));
        }
        state[*i] = byte.unwrap();
        *i += 1;
        if *i == size_of::<Int>() {
            *i = 0;
            Some(Some(Ok(*state)))
        } else {
            Some(None)
        }
    }).filter_map(identity).map(move |x: Result<[u8; size_of::<Int>()], Error>| Ok(from_int.map(Int::from_le_bytes(x?).try_into().ok().unwrap())))
}

pub fn write_file<I, Int: IOUInt>(data: I, filename: &str, always_replace: bool)
    where I: Iterator<Item = i64>,
        [(); size_of::<Int>()]: Sized
{
    if !Path::new(format!("{}{}", DATASTRUCTURE_PATH, filename).as_str()).exists() || always_replace {
        let mut writer = BufWriter::new(File::create(format!("{}{}", DATASTRUCTURE_PATH, filename)).unwrap());
        for element in data {
            assert!(element <= <Int as TryInto<i64>>::try_into(Int::max()).ok().unwrap() as i64);
            assert!(size_of::<Int>() == writer.write(&Int::try_from(element).ok().unwrap().to_le_bytes()).unwrap());
        }
        drop(writer);
    }
}

#[cfg(test)]
use std::fs::remove_file;
#[cfg(test)]
use std::panic::catch_unwind;
#[cfg(test)]
use feanor_math::assert_el_eq;
#[cfg(test)]
use feanor_math::rings::zn::zn_static::F17;

use crate::DATASTRUCTURE_PATH;

#[test]
fn test_read_write() {
    let ring = F17;
    write_file::<_, PolyCoeffInt>((0..1000).map(|x| x), "testfile", true);
    let result = catch_unwind(|| {
        let read_data = read_file::<_, PolyCoeffInt>(ring, "testfile").collect::<Result<Vec<_>, _>>().unwrap();
        assert_eq!(1000, read_data.len());
        for i in 0..1000 {
            assert_el_eq!(&ring, &ring.int_hom().map(i), &read_data[i as usize]);
        }
    });
    remove_file(format!("{}testfile", DATASTRUCTURE_PATH)).unwrap();
    result.unwrap();
}