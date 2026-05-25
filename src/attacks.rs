// ---- Pre-computed attack tables for non-sliding pieces ----

use crate::types::*;
use crate::bitboard::*;

static mut KNIGHT_ATTACKS: [u64; 64] = [0; 64];
static mut KING_ATTACKS: [u64; 64] = [0; 64];
static mut PAWN_ATTACKS: [[u64; 64]; 2] = [[0; 64]; 2]; // [color][square]

/// Between-ray table: squares strictly between two squares on the same line
static mut BETWEEN: [[u64; 64]; 64] = [[0; 64]; 64];
/// Line table: full line through two squares (if on same rank/file/diagonal)
static mut LINE: [[u64; 64]; 64] = [[0; 64]; 64];

pub fn init() {
    init_knight_attacks();
    init_king_attacks();
    init_pawn_attacks();
    init_rays();
}

fn init_knight_attacks() {
    for s in 0u8..64 {
        let bb = bit(s);
        let mut attacks = 0u64;
        attacks |= (bb << 17) & NOT_FILE_A;
        attacks |= (bb << 15) & NOT_FILE_H;
        attacks |= (bb << 10) & NOT_FILE_AB;
        attacks |= (bb << 6) & NOT_FILE_GH;
        attacks |= (bb >> 17) & NOT_FILE_H;
        attacks |= (bb >> 15) & NOT_FILE_A;
        attacks |= (bb >> 10) & NOT_FILE_GH;
        attacks |= (bb >> 6) & NOT_FILE_AB;
        unsafe { KNIGHT_ATTACKS[s as usize] = attacks; }
    }
}

fn init_king_attacks() {
    for s in 0u8..64 {
        let bb = bit(s);
        let mut attacks = 0u64;
        attacks |= shift_north(bb) | shift_south(bb);
        attacks |= shift_east(bb) | shift_west(bb);
        attacks |= shift_ne(bb) | shift_nw(bb);
        attacks |= shift_se(bb) | shift_sw(bb);
        unsafe { KING_ATTACKS[s as usize] = attacks; }
    }
}

fn init_pawn_attacks() {
    for s in 0u8..64 {
        let bb = bit(s);
        unsafe {
            // White pawn attacks: NE + NW
            PAWN_ATTACKS[WHITE.index()][s as usize] = shift_ne(bb) | shift_nw(bb);
            // Black pawn attacks: SE + SW
            PAWN_ATTACKS[BLACK.index()][s as usize] = shift_se(bb) | shift_sw(bb);
        }
    }
}

fn init_rays() {
    // Compute between and line tables using the magic attack tables
    // This must be called AFTER magic::init()
    for s1 in 0u8..64 {
        for s2 in 0u8..64 {
            if s1 == s2 { continue; }

            let bb1 = bit(s1);
            let bb2 = bit(s2);

            // Check if they're on the same bishop or rook line
            let rook_s1 = crate::magic::rook_attacks(s1, 0);
            let bishop_s1 = crate::magic::bishop_attacks(s1, 0);

            if rook_s1 & bb2 != 0 {
                // Same rank or file
                let between = crate::magic::rook_attacks(s1, bb2) & crate::magic::rook_attacks(s2, bb1);
                let line = (rook_s1 & crate::magic::rook_attacks(s2, 0)) | bb1 | bb2;
                unsafe {
                    BETWEEN[s1 as usize][s2 as usize] = between;
                    LINE[s1 as usize][s2 as usize] = line;
                }
            } else if bishop_s1 & bb2 != 0 {
                // Same diagonal
                let between = crate::magic::bishop_attacks(s1, bb2) & crate::magic::bishop_attacks(s2, bb1);
                let line = (bishop_s1 & crate::magic::bishop_attacks(s2, 0)) | bb1 | bb2;
                unsafe {
                    BETWEEN[s1 as usize][s2 as usize] = between;
                    LINE[s1 as usize][s2 as usize] = line;
                }
            }
        }
    }
}

// ---- Lookup functions ----

#[inline(always)]
pub fn knight_attacks(sq: Square) -> u64 {
    unsafe { KNIGHT_ATTACKS[sq as usize] }
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> u64 {
    unsafe { KING_ATTACKS[sq as usize] }
}

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> u64 {
    unsafe { PAWN_ATTACKS[color.index()][sq as usize] }
}

#[allow(dead_code)] // geometry primitive (squares strictly between s1 and s2); kept for completeness
#[inline(always)]
pub fn between(s1: Square, s2: Square) -> u64 {
    unsafe { BETWEEN[s1 as usize][s2 as usize] }
}

#[allow(dead_code)] // geometry primitive (full line through s1 and s2); kept for completeness
#[inline(always)]
pub fn line(s1: Square, s2: Square) -> u64 {
    unsafe { LINE[s1 as usize][s2 as usize] }
}

/// Is square attacked by the given side?
/// Bitboard of all squares attacked by `color` given board state.
pub fn attacks_by_color(pos_pieces: &[u64; 6], pos_colors: &[u64; 2], color: Color) -> u64 {
    use crate::bitboard::pop_lsb;
    let us = pos_colors[color.index()];
    let occ = pos_colors[0] | pos_colors[1];
    let mut atk: u64 = 0;

    // pawns
    let mut p = pos_pieces[PAWN.index()] & us;
    while p != 0 {
        let sq = pop_lsb(&mut p);
        atk |= pawn_attacks(color, sq);
    }
    // knights
    let mut n = pos_pieces[KNIGHT.index()] & us;
    while n != 0 {
        let sq = pop_lsb(&mut n);
        atk |= knight_attacks(sq);
    }
    // bishops + queens (diagonal)
    let mut bq = (pos_pieces[BISHOP.index()] | pos_pieces[QUEEN.index()]) & us;
    while bq != 0 {
        let sq = pop_lsb(&mut bq);
        atk |= crate::magic::bishop_attacks(sq, occ);
    }
    // rooks + queens (straight)
    let mut rq = (pos_pieces[ROOK.index()] | pos_pieces[QUEEN.index()]) & us;
    while rq != 0 {
        let sq = pop_lsb(&mut rq);
        atk |= crate::magic::rook_attacks(sq, occ);
    }
    // king
    let mut k = pos_pieces[KING.index()] & us;
    if k != 0 {
        let sq = pop_lsb(&mut k);
        atk |= king_attacks(sq);
    }
    atk
}

#[inline]
pub fn is_attacked(pos_pieces: &[u64; 6], pos_colors: &[u64; 2], sq: Square, by_color: Color) -> bool {
    debug_assert!(sq < 64, "is_attacked called with sq>=64");
    let them = pos_colors[by_color.index()];

    // Pawn attacks
    if pawn_attacks(by_color.flip(), sq) & pos_pieces[PAWN.index()] & them != 0 {
        return true;
    }
    // Knight attacks
    if knight_attacks(sq) & pos_pieces[KNIGHT.index()] & them != 0 {
        return true;
    }
    // King attacks
    if king_attacks(sq) & pos_pieces[KING.index()] & them != 0 {
        return true;
    }
    // Sliding attacks
    let occ = pos_colors[0] | pos_colors[1];
    let bishop_queen = (pos_pieces[BISHOP.index()] | pos_pieces[QUEEN.index()]) & them;
    if crate::magic::bishop_attacks(sq, occ) & bishop_queen != 0 {
        return true;
    }
    let rook_queen = (pos_pieces[ROOK.index()] | pos_pieces[QUEEN.index()]) & them;
    if crate::magic::rook_attacks(sq, occ) & rook_queen != 0 {
        return true;
    }
    false
}
