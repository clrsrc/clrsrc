// ---- Move generation ----
// Generates pseudo-legal moves. Legality is checked after make_move
// by verifying the king isn't left in check.

use crate::types::*;
use crate::bitboard::*;
use crate::attacks;
use crate::magic;
use crate::board::Position;

/// Generate all pseudo-legal moves
pub fn generate_all(pos: &Position, list: &mut MoveList) {
    let us = pos.side;
    let occ = pos.occupancy();
    let our = pos.us();
    let their = pos.them();

    generate_pawns(pos, list, us, occ, our, their);
    generate_knights(list, us, occ, our, their, pos);
    generate_bishops(list, occ, our, their, pos);
    generate_rooks(list, occ, our, their, pos);
    generate_queens(list, occ, our, their, pos);
    generate_king(pos, list, us, occ, our, their);
    generate_castling(pos, list, us, occ);
}

/// Generate only capture moves (for quiescence search)
pub fn generate_captures(pos: &Position, list: &mut MoveList) {
    let us = pos.side;
    let occ = pos.occupancy();
    let our = pos.us();
    let their = pos.them();
    let pawns = pos.pieces[PAWN.index()] & our;

    generate_pawn_captures(pos, list, us, pawns, their);
    generate_piece_captures(list, occ, our, their, pos, KNIGHT);
    generate_piece_captures(list, occ, our, their, pos, BISHOP);
    generate_piece_captures(list, occ, our, their, pos, ROOK);
    generate_piece_captures(list, occ, our, their, pos, QUEEN);
    generate_king_captures(list, our, their, pos);
}

// ---- Pawn moves ----

fn generate_pawns(pos: &Position, list: &mut MoveList, us: Color, occ: u64, our: u64, their: u64) {
    let pawns = pos.pieces[PAWN.index()] & our;
    let empty = !occ;

    // Direct branches per color — avoids fn-pointer dispatch that LLVM may
    // not devirtualize across crate boundaries.
    let (single, double, promo_rank) = if us == WHITE {
        let s = shift_north(pawns) & empty;
        let d = shift_north(s & RANK_3) & empty;
        (s, d, RANK_8)
    } else {
        let s = shift_south(pawns) & empty;
        let d = shift_south(s & RANK_6) & empty;
        (s, d, RANK_1)
    };
    let promo_pushes = single & promo_rank;
    let normal_pushes = single & !promo_rank;

    // Promotions from pushes
    for to in BitIter(promo_pushes) {
        let from = if us == WHITE { to - 8 } else { to + 8 };
        list.push(Move::new(from, to, Move::FLAG_QUEEN_PROMO));
        list.push(Move::new(from, to, Move::FLAG_ROOK_PROMO));
        list.push(Move::new(from, to, Move::FLAG_BISHOP_PROMO));
        list.push(Move::new(from, to, Move::FLAG_KNIGHT_PROMO));
    }

    // Normal pushes
    for to in BitIter(normal_pushes) {
        let from = if us == WHITE { to - 8 } else { to + 8 };
        list.push(Move::new(from, to, Move::FLAG_QUIET));
    }

    // Double pushes
    for to in BitIter(double) {
        let from = if us == WHITE { to - 16 } else { to + 16 };
        list.push(Move::new(from, to, Move::FLAG_DOUBLE_PAWN));
    }

    // Captures — reuse the pawns bitboard we already loaded
    generate_pawn_captures(pos, list, us, pawns, their);
}

fn generate_pawn_captures(pos: &Position, list: &mut MoveList, us: Color, pawns: u64, their: u64) {
    let promo_rank = if us == WHITE { RANK_8 } else { RANK_1 };

    let (left, right, left_shift, right_shift) = if us == WHITE {
        (shift_nw(pawns) & their, shift_ne(pawns) & their, -7i8, -9i8)
    } else {
        (shift_sw(pawns) & their, shift_se(pawns) & their, 9i8, 7i8)
    };
    for to in BitIter(left & !promo_rank) {
        let from = (to as i8 + left_shift) as u8;
        list.push(Move::new(from, to, Move::FLAG_CAPTURE));
    }
    for to in BitIter(left & promo_rank) {
        let from = (to as i8 + left_shift) as u8;
        list.push(Move::new(from, to, Move::FLAG_QUEEN_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_ROOK_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_BISHOP_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_KNIGHT_PROMO_CAP));
    }

    for to in BitIter(right & !promo_rank) {
        let from = (to as i8 + right_shift) as u8;
        list.push(Move::new(from, to, Move::FLAG_CAPTURE));
    }
    for to in BitIter(right & promo_rank) {
        let from = (to as i8 + right_shift) as u8;
        list.push(Move::new(from, to, Move::FLAG_QUEEN_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_ROOK_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_BISHOP_PROMO_CAP));
        list.push(Move::new(from, to, Move::FLAG_KNIGHT_PROMO_CAP));
    }

    // En passant
    if pos.ep_square != NO_SQ {
        let attackers = attacks::pawn_attacks(us.flip(), pos.ep_square) & pawns;
        for from in BitIter(attackers) {
            list.push(Move::new(from, pos.ep_square, Move::FLAG_EP));
        }
    }
}

// ---- Piece moves ----

fn generate_knights(list: &mut MoveList, _us: Color, occ: u64, our: u64, their: u64, pos: &Position) {
    let knights = pos.pieces[KNIGHT.index()] & our;
    let empty = !occ;
    for from in BitIter(knights) {
        let atk = attacks::knight_attacks(from);
        for to in BitIter(atk & their) {
            list.push(Move::new(from, to, Move::FLAG_CAPTURE));
        }
        for to in BitIter(atk & empty) {
            list.push(Move::new(from, to, Move::FLAG_QUIET));
        }
    }
}

fn generate_bishops(list: &mut MoveList, occ: u64, our: u64, their: u64, pos: &Position) {
    let bishops = pos.pieces[BISHOP.index()] & our;
    let empty = !occ;
    for from in BitIter(bishops) {
        let atk = magic::bishop_attacks(from, occ);
        for to in BitIter(atk & their) {
            list.push(Move::new(from, to, Move::FLAG_CAPTURE));
        }
        for to in BitIter(atk & empty) {
            list.push(Move::new(from, to, Move::FLAG_QUIET));
        }
    }
}

fn generate_rooks(list: &mut MoveList, occ: u64, our: u64, their: u64, pos: &Position) {
    let rooks = pos.pieces[ROOK.index()] & our;
    let empty = !occ;
    for from in BitIter(rooks) {
        let atk = magic::rook_attacks(from, occ);
        for to in BitIter(atk & their) {
            list.push(Move::new(from, to, Move::FLAG_CAPTURE));
        }
        for to in BitIter(atk & empty) {
            list.push(Move::new(from, to, Move::FLAG_QUIET));
        }
    }
}

fn generate_queens(list: &mut MoveList, occ: u64, our: u64, their: u64, pos: &Position) {
    let queens = pos.pieces[QUEEN.index()] & our;
    let empty = !occ;
    for from in BitIter(queens) {
        let atk = magic::queen_attacks(from, occ);
        for to in BitIter(atk & their) {
            list.push(Move::new(from, to, Move::FLAG_CAPTURE));
        }
        for to in BitIter(atk & empty) {
            list.push(Move::new(from, to, Move::FLAG_QUIET));
        }
    }
}

fn generate_king(pos: &Position, list: &mut MoveList, _us: Color, occ: u64, _our: u64, their: u64) {
    let ksq = pos.king_sq(pos.side);
    let atk = attacks::king_attacks(ksq);
    for to in BitIter(atk & their) {
        list.push(Move::new(ksq, to, Move::FLAG_CAPTURE));
    }
    for to in BitIter(atk & !occ) {
        list.push(Move::new(ksq, to, Move::FLAG_QUIET));
    }
}

fn generate_king_captures(list: &mut MoveList, _our: u64, their: u64, pos: &Position) {
    let ksq = pos.king_sq(pos.side);
    let atk = attacks::king_attacks(ksq) & their;
    for to in BitIter(atk) {
        list.push(Move::new(ksq, to, Move::FLAG_CAPTURE));
    }
}

fn generate_piece_captures(list: &mut MoveList, occ: u64, our: u64, their: u64, pos: &Position, pt: PieceType) {
    let pieces = pos.pieces[pt.index()] & our;
    for from in BitIter(pieces) {
        let atk = match pt {
            PieceType::Knight => attacks::knight_attacks(from),
            PieceType::Bishop => magic::bishop_attacks(from, occ),
            PieceType::Rook => magic::rook_attacks(from, occ),
            PieceType::Queen => magic::queen_attacks(from, occ),
            _ => 0,
        };
        for to in BitIter(atk & their) {
            list.push(Move::new(from, to, Move::FLAG_CAPTURE));
        }
    }
}

// ---- Castling ----

fn generate_castling(pos: &Position, list: &mut MoveList, us: Color, occ: u64) {
    if us == WHITE {
        // Kingside: e1-g1, f1 and g1 must be empty, e1/f1/g1 not attacked
        if pos.castling.has(CastlingRights::WK)
            && occ & (bit(F1) | bit(G1)) == 0
            && !attacks::is_attacked(&pos.pieces, &pos.colors, E1, BLACK)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, F1, BLACK)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, G1, BLACK)
        {
            list.push(Move::new(E1, G1, Move::FLAG_KING_CASTLE));
        }
        // Queenside: e1-c1, b1/c1/d1 must be empty, e1/d1/c1 not attacked
        if pos.castling.has(CastlingRights::WQ)
            && occ & (bit(B1) | bit(C1) | bit(D1)) == 0
            && !attacks::is_attacked(&pos.pieces, &pos.colors, E1, BLACK)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, D1, BLACK)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, C1, BLACK)
        {
            list.push(Move::new(E1, C1, Move::FLAG_QUEEN_CASTLE));
        }
    } else {
        if pos.castling.has(CastlingRights::BK)
            && occ & (bit(F8) | bit(G8)) == 0
            && !attacks::is_attacked(&pos.pieces, &pos.colors, E8, WHITE)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, F8, WHITE)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, G8, WHITE)
        {
            list.push(Move::new(E8, G8, Move::FLAG_KING_CASTLE));
        }
        if pos.castling.has(CastlingRights::BQ)
            && occ & (bit(B8) | bit(C8) | bit(D8)) == 0
            && !attacks::is_attacked(&pos.pieces, &pos.colors, E8, WHITE)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, D8, WHITE)
            && !attacks::is_attacked(&pos.pieces, &pos.colors, C8, WHITE)
        {
            list.push(Move::new(E8, C8, Move::FLAG_QUEEN_CASTLE));
        }
    }
}

// ---- Legal move generation (filters pseudo-legal) ----

/// Generate all legal moves
pub fn generate_legal(pos: &mut Position, list: &mut MoveList) {
    let mut pseudo = MoveList::new();
    generate_all(pos, &mut pseudo);

    for i in 0..pseudo.len {
        let mv = pseudo.moves[i];
        let undo = pos.make_move(mv);
        // After making the move, side has switched.
        // Check if the side that just moved left their king in check.
        let ksq = pos.king_sq(pos.side.flip());
        if !attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side) {
            list.push(mv);
        }
        pos.unmake_move(mv, undo);
    }
}
