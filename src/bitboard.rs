// ---- Bitboard utilities ----

use crate::types::*;

pub const FILE_A: u64 = 0x0101010101010101;
pub const FILE_B: u64 = 0x0202020202020202;
pub const FILE_G: u64 = 0x4040404040404040;
pub const FILE_H: u64 = 0x8080808080808080;
pub const RANK_1: u64 = 0x00000000000000FF;
pub const RANK_3: u64 = 0x0000000000FF0000;
pub const RANK_6: u64 = 0x0000FF0000000000;
pub const RANK_8: u64 = 0xFF00000000000000;
pub const NOT_FILE_A: u64 = !FILE_A;
pub const NOT_FILE_H: u64 = !FILE_H;
pub const NOT_FILE_AB: u64 = !(FILE_A | FILE_B);
pub const NOT_FILE_GH: u64 = !(FILE_G | FILE_H);

pub const fn bit(sq: Square) -> u64 {
    1u64 << sq
}

pub const fn lsb(bb: u64) -> Square {
    bb.trailing_zeros() as Square
}

#[inline(always)]
pub fn pop_lsb(bb: &mut u64) -> Square {
    let sq = lsb(*bb);
    *bb &= *bb - 1;
    sq
}

pub const fn popcount(bb: u64) -> u32 {
    bb.count_ones()
}

#[inline(always)]
pub fn shift_north(bb: u64) -> u64 {
    bb << 8
}

#[inline(always)]
pub fn shift_south(bb: u64) -> u64 {
    bb >> 8
}

#[inline(always)]
pub fn shift_east(bb: u64) -> u64 {
    (bb << 1) & NOT_FILE_A
}

#[inline(always)]
pub fn shift_west(bb: u64) -> u64 {
    (bb >> 1) & NOT_FILE_H
}

#[inline(always)]
pub fn shift_ne(bb: u64) -> u64 {
    (bb << 9) & NOT_FILE_A
}

#[inline(always)]
pub fn shift_nw(bb: u64) -> u64 {
    (bb << 7) & NOT_FILE_H
}

#[inline(always)]
pub fn shift_se(bb: u64) -> u64 {
    (bb >> 7) & NOT_FILE_A
}

#[inline(always)]
pub fn shift_sw(bb: u64) -> u64 {
    (bb >> 9) & NOT_FILE_H
}

/// Iterate over set bits of a bitboard
pub struct BitIter(pub u64);

impl Iterator for BitIter {
    type Item = Square;

    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(pop_lsb(&mut self.0))
        }
    }
}

/// Print bitboard for debugging
#[allow(dead_code)]
pub fn print_bb(bb: u64) {
    for rank in (0..8).rev() {
        for file in 0..8 {
            if bb & bit(sq(file, rank)) != 0 {
                print!("1 ");
            } else {
                print!(". ");
            }
        }
        println!();
    }
    println!();
}
