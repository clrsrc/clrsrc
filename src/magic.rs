// ---- Magic Bitboard move generation for sliding pieces ----
// Magics are found at startup using a deterministic PRNG.
// This takes ~10-50ms and produces reproducible results.

use crate::types::*;
use crate::bitboard::*;

// ---- Magic entry per square ----
struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

static mut BISHOP_MAGICS: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
static mut ROOK_MAGICS: [MagicEntry; 64] = unsafe { std::mem::zeroed() };
static mut ATTACK_TABLE: Vec<u64> = Vec::new();

// Total attack table size (sum of all per-square tables)
// Bishop: ~5K entries, Rook: ~100K entries
static mut TABLE_SIZE: usize = 0;

// ---- Slow ray-traced attack generation (for init only) ----

fn bishop_attacks_slow(square: Square, occ: u64) -> u64 {
    let mut attacks = 0u64;
    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (df, dr) in directions {
        let mut f = file_of(square) as i8 + df;
        let mut r = rank_of(square) as i8 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let s = sq(f as u8, r as u8);
            attacks |= bit(s);
            if occ & bit(s) != 0 { break; }
            f += df;
            r += dr;
        }
    }
    attacks
}

fn rook_attacks_slow(square: Square, occ: u64) -> u64 {
    let mut attacks = 0u64;
    let directions: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
    for (df, dr) in directions {
        let mut f = file_of(square) as i8 + df;
        let mut r = rank_of(square) as i8 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let s = sq(f as u8, r as u8);
            attacks |= bit(s);
            if occ & bit(s) != 0 { break; }
            f += df;
            r += dr;
        }
    }
    attacks
}

// ---- Relevant occupancy masks (excluding edges) ----

fn bishop_mask(square: Square) -> u64 {
    let mut mask = 0u64;
    let directions: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
    for (df, dr) in directions {
        let mut f = file_of(square) as i8 + df;
        let mut r = rank_of(square) as i8 + dr;
        // Stop one before the edge
        while f > 0 && f < 7 && r > 0 && r < 7 {
            mask |= bit(sq(f as u8, r as u8));
            f += df;
            r += dr;
        }
    }
    mask
}

fn rook_mask(square: Square) -> u64 {
    let mut mask = 0u64;
    let file = file_of(square) as i8;
    let rank = rank_of(square) as i8;
    // Horizontal: exclude edges
    for f in (file + 1)..7 {
        mask |= bit(sq(f as u8, rank as u8));
    }
    for f in 1..file {
        mask |= bit(sq(f as u8, rank as u8));
    }
    // Vertical: exclude edges
    for r in (rank + 1)..7 {
        mask |= bit(sq(file as u8, r as u8));
    }
    for r in 1..rank {
        mask |= bit(sq(file as u8, r as u8));
    }
    mask
}

// ---- Enumerate all subsets of a mask (Carry-Rippler) ----

fn enumerate_subsets(mask: u64) -> Vec<u64> {
    let mut subsets = Vec::new();
    let mut subset = 0u64;
    loop {
        subsets.push(subset);
        subset = subset.wrapping_sub(mask) & mask;
        if subset == 0 { break; }
    }
    subsets
}

// ---- PRNG for finding magic numbers ----

struct MagicRng(u64);

impl MagicRng {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545F4914F6CDD1D)
    }

    /// Generate a sparse random number (few bits set) — good for magics
    fn sparse_u64(&mut self) -> u64 {
        self.next_u64() & self.next_u64() & self.next_u64()
    }
}

// ---- Find magic number for one square ----

fn find_magic(
    sq: Square,
    mask: u64,
    is_bishop: bool,
    rng: &mut MagicRng,
) -> (u64, Vec<u64>) {
    let bits = popcount(mask);
    let shift = 64 - bits;
    let n = 1usize << bits;

    let subsets = enumerate_subsets(mask);
    let attacks: Vec<u64> = subsets
        .iter()
        .map(|&occ| {
            if is_bishop {
                bishop_attacks_slow(sq, occ)
            } else {
                rook_attacks_slow(sq, occ)
            }
        })
        .collect();

    loop {
        let magic = rng.sparse_u64();
        if popcount((mask.wrapping_mul(magic)) >> 56) < 6 {
            continue;
        }

        let mut table = vec![0u64; n];
        let mut fail = false;

        for i in 0..subsets.len() {
            let idx = (subsets[i].wrapping_mul(magic) >> shift) as usize;
            if table[idx] == 0 {
                table[idx] = attacks[i];
            } else if table[idx] != attacks[i] {
                fail = true;
                break;
            }
        }

        if !fail {
            return (magic, table);
        }
    }
}

// ---- Initialize all magic tables ----

pub fn init() {
    let mut rng = MagicRng(0xDEADBEEF12345678);
    let mut all_tables: Vec<u64> = Vec::new();

    unsafe {
        // Bishops
        for sq in 0u8..64 {
            let mask = bishop_mask(sq);
            let bits = popcount(mask);
            let shift = 64 - bits;
            let (magic, table) = find_magic(sq, mask, true, &mut rng);

            BISHOP_MAGICS[sq as usize] = MagicEntry {
                mask,
                magic,
                shift,
                offset: all_tables.len(),
            };
            all_tables.extend_from_slice(&table);
        }

        // Rooks
        for sq in 0u8..64 {
            let mask = rook_mask(sq);
            let bits = popcount(mask);
            let shift = 64 - bits;
            let (magic, table) = find_magic(sq, mask, false, &mut rng);

            ROOK_MAGICS[sq as usize] = MagicEntry {
                mask,
                magic,
                shift,
                offset: all_tables.len(),
            };
            all_tables.extend_from_slice(&table);
        }

        TABLE_SIZE = all_tables.len();
        ATTACK_TABLE = all_tables;
    }
}

// ---- Lookup functions ----

// ATTACK_TABLE is filled once in init() before any search starts, then only ever
// read — the shared-ref-to-mutable-static lint is a false positive for this pattern.
#[allow(static_mut_refs)]
#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: u64) -> u64 {
    unsafe {
        let entry = &BISHOP_MAGICS[sq as usize];
        let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        *ATTACK_TABLE.get_unchecked(entry.offset + idx)
    }
}

#[allow(static_mut_refs)]
#[inline(always)]
pub fn rook_attacks(sq: Square, occ: u64) -> u64 {
    unsafe {
        let entry = &ROOK_MAGICS[sq as usize];
        let idx = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;
        *ATTACK_TABLE.get_unchecked(entry.offset + idx)
    }
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: u64) -> u64 {
    bishop_attacks(sq, occ) | rook_attacks(sq, occ)
}
