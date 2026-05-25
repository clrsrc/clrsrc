// ---- Hand-Crafted Evaluation (HCE) ----
// Simple material + PST evaluation for bootstrapping.
// Will be replaced by NNUE later.

use crate::types::*;
use crate::bitboard::*;
use crate::board::Position;
use crate::attacks;
use crate::magic;

// Piece values (midgame / endgame)
const MG_PIECE: [i32; 6] = [100, 320, 330, 500, 1000, 0]; // P N B R Q K
const EG_PIECE: [i32; 6] = [120, 310, 320, 550, 1050, 0];

// Piece-square tables (midgame, from white's perspective, A1=index 0)
// Values are centipawns added to base piece value
#[rustfmt::skip]
const MG_PAWN_PST: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const MG_KNIGHT_PST: [i32; 64] = [
   -50,-40,-30,-30,-30,-30,-40,-50,
   -40,-20,  0,  5,  5,  0,-20,-40,
   -30,  5, 10, 15, 15, 10,  5,-30,
   -30,  0, 15, 20, 20, 15,  0,-30,
   -30,  5, 15, 20, 20, 15,  5,-30,
   -30,  0, 10, 15, 15, 10,  0,-30,
   -40,-20,  0,  0,  0,  0,-20,-40,
   -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const MG_BISHOP_PST: [i32; 64] = [
   -20,-10,-10,-10,-10,-10,-10,-20,
   -10,  5,  0,  0,  0,  0,  5,-10,
   -10, 10, 10, 10, 10, 10, 10,-10,
   -10,  0, 10, 10, 10, 10,  0,-10,
   -10,  5,  5, 10, 10,  5,  5,-10,
   -10,  0,  5, 10, 10,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const MG_ROOK_PST: [i32; 64] = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const MG_QUEEN_PST: [i32; 64] = [
   -20,-10,-10, -5, -5,-10,-10,-20,
   -10,  0,  5,  0,  0,  0,  0,-10,
   -10,  5,  5,  5,  5,  5,  0,-10,
     0,  0,  5,  5,  5,  5,  0, -5,
    -5,  0,  5,  5,  5,  5,  0, -5,
   -10,  0,  5,  5,  5,  5,  0,-10,
   -10,  0,  0,  0,  0,  0,  0,-10,
   -20,-10,-10, -5, -5,-10,-10,-20,
];

#[rustfmt::skip]
const MG_KING_PST: [i32; 64] = [
    20, 30, 10,  0,  0, 10, 30, 20,
    20, 20,  0,  0,  0,  0, 20, 20,
   -10,-20,-20,-20,-20,-20,-20,-10,
   -20,-30,-30,-40,-40,-30,-30,-20,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
   -30,-40,-40,-50,-50,-40,-40,-30,
];

#[rustfmt::skip]
const EG_KING_PST: [i32; 64] = [
   -50,-30,-30,-30,-30,-30,-30,-50,
   -30,-20,-10,  0,  0,-10,-20,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 30, 40, 40, 30,-10,-30,
   -30,-10, 20, 30, 30, 20,-10,-30,
   -30,-30,  0,  0,  0,  0,-30,-30,
   -50,-30,-30,-30,-30,-30,-30,-50,
];

const MG_PST: [&[i32; 64]; 6] = [
    &MG_PAWN_PST, &MG_KNIGHT_PST, &MG_BISHOP_PST,
    &MG_ROOK_PST, &MG_QUEEN_PST, &MG_KING_PST,
];

// Phase values for tapering
const PHASE_VALS: [i32; 6] = [0, 1, 1, 2, 4, 0]; // P=0, N=1, B=1, R=2, Q=4
const TOTAL_PHASE: i32 = 24; // 4*1 + 4*1 + 4*2 + 2*4

/// Static evaluation in centipawns, from the side-to-move's perspective
pub fn evaluate(pos: &Position) -> i32 {
    let mut mg = [0i32; 2]; // midgame score per color
    let mut eg = [0i32; 2]; // endgame score per color
    let mut phase = TOTAL_PHASE;

    // Material + PST
    for color_idx in 0..2 {
        let color_bb = pos.colors[color_idx];
        for pt in 0..6 {
            let mut pieces = pos.pieces[pt] & color_bb;
            while pieces != 0 {
                let sq = pop_lsb(&mut pieces);
                // Mirror square for black (PST is from white's perspective)
                let pst_sq = if color_idx == 0 { sq } else { mirror_sq(sq) };

                mg[color_idx] += MG_PIECE[pt] + MG_PST[pt][pst_sq as usize];
                eg[color_idx] += EG_PIECE[pt] + if pt == 5 {
                    EG_KING_PST[pst_sq as usize]
                } else {
                    MG_PST[pt][pst_sq as usize] / 2 // Simple EG PST: half of MG
                };

                phase -= PHASE_VALS[pt];
            }
        }
    }

    // Bishop pair bonus
    for color_idx in 0..2 {
        let bishops = pos.pieces[BISHOP.index()] & pos.colors[color_idx];
        if popcount(bishops) >= 2 {
            mg[color_idx] += 30;
            eg[color_idx] += 50;
        }
    }

    // Rook on open/semi-open file
    for color_idx in 0..2 {
        let our_pawns = pos.pieces[PAWN.index()] & pos.colors[color_idx];
        let their_pawns = pos.pieces[PAWN.index()] & pos.colors[1 - color_idx];
        let mut rooks = pos.pieces[ROOK.index()] & pos.colors[color_idx];
        while rooks != 0 {
            let sq = pop_lsb(&mut rooks);
            let file_mask = FILE_A << file_of(sq);
            if our_pawns & file_mask == 0 {
                if their_pawns & file_mask == 0 {
                    mg[color_idx] += 40; // open file
                    eg[color_idx] += 20;
                } else {
                    mg[color_idx] += 20; // semi-open
                    eg[color_idx] += 10;
                }
            }
        }
    }

    // Passed pawns (simple: no opposing pawn on same or adjacent files ahead)
    for color_idx in 0..2 {
        let their_pawns = pos.pieces[PAWN.index()] & pos.colors[1 - color_idx];
        let mut pawns = pos.pieces[PAWN.index()] & pos.colors[color_idx];
        while pawns != 0 {
            let sq = pop_lsb(&mut pawns);
            let file = file_of(sq);
            let rank = rank_of(sq);
            let rel_rank = if color_idx == 0 { rank } else { 7 - rank };

            // Build mask of squares ahead on same + adjacent files
            let mut block_mask = 0u64;
            for f in file.saturating_sub(1)..=(file + 1).min(7) {
                let file_bb = FILE_A << f;
                if color_idx == 0 {
                    // White: ranks above
                    block_mask |= file_bb & !((1u64 << ((rank + 1) * 8)) - 1);
                } else {
                    // Black: ranks below
                    block_mask |= file_bb & ((1u64 << (rank * 8)) - 1);
                }
            }

            if their_pawns & block_mask == 0 {
                // Passed pawn bonus by relative rank
                let bonus = match rel_rank {
                    1 => 10,
                    2 => 15,
                    3 => 25,
                    4 => 50,
                    5 => 80,
                    6 => 130,
                    _ => 0,
                };
                eg[color_idx] += bonus;
                mg[color_idx] += bonus / 3;
            }
        }
    }

    // Mobility (simple: count attacked squares for knights and bishops)
    let occ = pos.occupancy();
    for color_idx in 0..2 {
        let our = pos.colors[color_idx];
        // Knight mobility
        let mut knights = pos.pieces[KNIGHT.index()] & our;
        while knights != 0 {
            let sq = pop_lsb(&mut knights);
            let mob = popcount(attacks::knight_attacks(sq) & !our) as i32;
            mg[color_idx] += (mob - 4) * 4;
            eg[color_idx] += (mob - 4) * 4;
        }
        // Bishop mobility
        let mut bishops = pos.pieces[BISHOP.index()] & our;
        while bishops != 0 {
            let sq = pop_lsb(&mut bishops);
            let mob = popcount(magic::bishop_attacks(sq, occ) & !our) as i32;
            mg[color_idx] += (mob - 5) * 5;
            eg[color_idx] += (mob - 5) * 5;
        }
    }

    // Tempo bonus for side to move
    let tempo = 15;

    // Taper between midgame and endgame
    let phase = phase.max(0);
    let mg_score = mg[pos.side.index()] - mg[pos.side.flip().index()];
    let eg_score = eg[pos.side.index()] - eg[pos.side.flip().index()];
    let score = (mg_score * phase + eg_score * (TOTAL_PHASE - phase)) / TOTAL_PHASE;

    score + tempo
}
