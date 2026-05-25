// ---- Move Ordering ----
// Priority: TT move > Good captures (MVV-LVA + SEE) > Killers > Counter move > Quiet (history + cont) > Bad captures

use std::mem::MaybeUninit;

use crate::types::*;
use crate::board::Position;
use crate::attacks;
use crate::magic;
use crate::bitboard::*;

const TT_SCORE: i32 = 10_000_000;
const GOOD_CAPTURE_BASE: i32 = 5_000_000;
const KILLER1_SCORE: i32 = 900_000;
const KILLER2_SCORE: i32 = 800_000;
const COUNTER_MOVE_SCORE: i32 = 700_000;
const BAD_CAPTURE: i32 = -1_000_000;

// MVV-LVA values: victim * 10 - attacker (higher = search first)
const MVV_LVA: [[i32; 6]; 6] = [
    // Attacker: P   N   B   R   Q   K    Victim: Pawn
    [15, 14, 13, 12, 11, 10],
    // Victim: Knight
    [25, 24, 23, 22, 21, 20],
    // Victim: Bishop
    [35, 34, 33, 32, 31, 30],
    // Victim: Rook
    [45, 44, 43, 42, 41, 40],
    // Victim: Queen
    [55, 54, 53, 52, 51, 50],
    // Victim: King (shouldn't happen)
    [0, 0, 0, 0, 0, 0],
];

pub const MAX_PLY: usize = 128;

// ---- Full SEE (Static Exchange Evaluation) ----

const SEE_VAL: [i32; 7] = [100, 320, 330, 500, 1000, 20000, 0]; // P N B R Q K None

/// All pieces attacking a square given an occupancy
#[inline]
fn attackers_to_sq(pos: &Position, sq: Square, occ: u64) -> u64 {
    // Pawn attacks: white pawns that attack sq = squares a black pawn on sq would attack
    (attacks::pawn_attacks(BLACK, sq) & pos.pieces[PAWN.index()] & pos.colors[WHITE.index()])
    | (attacks::pawn_attacks(WHITE, sq) & pos.pieces[PAWN.index()] & pos.colors[BLACK.index()])
    | (attacks::knight_attacks(sq) & pos.pieces[KNIGHT.index()])
    | (magic::bishop_attacks(sq, occ) & (pos.pieces[BISHOP.index()] | pos.pieces[QUEEN.index()]))
    | (magic::rook_attacks(sq, occ) & (pos.pieces[ROOK.index()] | pos.pieces[QUEEN.index()]))
    | (attacks::king_attacks(sq) & pos.pieces[KING.index()])
}

/// Static Exchange Evaluation: returns true if SEE of move >= threshold
#[inline]
pub fn see_ge(pos: &Position, mv: Move, threshold: i32) -> bool {
    if mv.is_castle() { return threshold <= 0; }

    let from = mv.from();
    let to = mv.to();

    // Safety guard against invalid squares
    if from >= 64 || to >= 64 { return false; }

    // Compute initial balance after our capture
    let mut balance = if mv.is_ep() { SEE_VAL[0] }
                      else if mv.is_capture() { SEE_VAL[piece_type(pos.mailbox[to as usize]).index()] }
                      else { 0 };
    if mv.is_promotion() {
        balance += SEE_VAL[mv.promo_piece_type().index()] - SEE_VAL[0];
    }
    balance -= threshold;
    if balance < 0 { return false; }

    // Cost of losing our piece
    let our_val = if mv.is_promotion() { SEE_VAL[mv.promo_piece_type().index()] }
                  else { SEE_VAL[piece_type(pos.mailbox[from as usize]).index()] };
    balance -= our_val;
    if balance >= 0 { return true; }

    // Iterative exchange: occupancy with from and to removed (X-ray through both)
    let mut occ = pos.occupancy() ^ bit(from) ^ bit(to);
    if mv.is_ep() {
        occ ^= bit(if pos.side == WHITE { to - 8 } else { to + 8 });
    }

    let diag = pos.pieces[BISHOP.index()] | pos.pieces[QUEEN.index()];
    let orth = pos.pieces[ROOK.index()] | pos.pieces[QUEEN.index()];

    let mut attackers = attackers_to_sq(pos, to, occ) & occ;
    let mut stm = pos.side; // will be flipped at loop start
    let mut result = true;

    loop {
        stm = stm.flip();
        let stm_attackers = attackers & pos.colors[stm.index()];
        if stm_attackers == 0 { break; }

        result = !result;

        // Find least valuable attacker
        let (pt, piece_bb);
        if stm_attackers & pos.pieces[PAWN.index()] != 0 {
            pt = PAWN; piece_bb = stm_attackers & pos.pieces[PAWN.index()];
        } else if stm_attackers & pos.pieces[KNIGHT.index()] != 0 {
            pt = KNIGHT; piece_bb = stm_attackers & pos.pieces[KNIGHT.index()];
        } else if stm_attackers & pos.pieces[BISHOP.index()] != 0 {
            pt = BISHOP; piece_bb = stm_attackers & pos.pieces[BISHOP.index()];
        } else if stm_attackers & pos.pieces[ROOK.index()] != 0 {
            pt = ROOK; piece_bb = stm_attackers & pos.pieces[ROOK.index()];
        } else if stm_attackers & pos.pieces[QUEEN.index()] != 0 {
            pt = QUEEN; piece_bb = stm_attackers & pos.pieces[QUEEN.index()];
        } else {
            // King: if opponent still has attackers, king can't hold
            if attackers & pos.colors[stm.flip().index()] != 0 {
                result = !result;
            }
            break;
        }

        // Negamax balance
        balance = -balance - 1 - SEE_VAL[pt.index()];
        if balance >= 0 { break; }

        // Remove piece from occupancy, update X-ray
        occ ^= bit(lsb(piece_bb));
        if pt == PAWN || pt == BISHOP || pt == QUEEN {
            attackers |= magic::bishop_attacks(to, occ) & diag;
        }
        if pt == ROOK || pt == QUEEN {
            attackers |= magic::rook_attacks(to, occ) & orth;
        }
        attackers &= occ;
    }

    result
}

// ---- Killer moves (2 per ply) ----

pub struct Killers {
    pub moves: [[Move; 2]; MAX_PLY],
}

impl Killers {
    pub fn new() -> Self {
        Killers {
            moves: [[Move::NULL; 2]; MAX_PLY],
        }
    }

    pub fn store(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY { return; }
        if self.moves[ply][0] != mv {
            self.moves[ply][1] = self.moves[ply][0];
            self.moves[ply][0] = mv;
        }
    }

    #[inline]
    pub fn is_killer(&self, ply: usize, mv: Move) -> Option<i32> {
        if ply >= MAX_PLY { return None; }
        if self.moves[ply][0] == mv {
            Some(KILLER1_SCORE)
        } else if self.moves[ply][1] == mv {
            Some(KILLER2_SCORE)
        } else {
            None
        }
    }

    pub fn clear(&mut self) {
        for k in self.moves.iter_mut() {
            *k = [Move::NULL; 2];
        }
    }
}

// ---- History heuristic [color][from][to] ----

pub struct History {
    pub table: [[[i32; 64]; 64]; 2],
}

impl History {
    pub fn new() -> Self {
        History {
            table: [[[0; 64]; 64]; 2],
        }
    }

    #[inline]
    pub fn get(&self, color: Color, mv: Move) -> i32 {
        self.table[color.index()][mv.from() as usize][mv.to() as usize]
    }

    #[inline]
    pub fn update(&mut self, color: Color, mv: Move, depth: i32, is_good: bool) {
        let bonus = if is_good { depth * depth } else { -(depth * depth) };
        let entry = &mut self.table[color.index()][mv.from() as usize][mv.to() as usize];
        // Gravity: dampen old values
        *entry += bonus - *entry * bonus.abs() / 16384;
    }

    /// Age all entries by halving (call at start of each search)
    pub fn age(&mut self) {
        for color in self.table.iter_mut() {
            for from in color.iter_mut() {
                for to in from.iter_mut() {
                    *to /= 2;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.table = [[[0; 64]; 64]; 2];
    }
}

// ---- Counter Move heuristic [color][piece_type][to] ----

pub struct CounterMoves {
    table: [[[Move; 64]; 6]; 2],
}

impl CounterMoves {
    pub fn new() -> Self {
        CounterMoves { table: [[[Move::NULL; 64]; 6]; 2] }
    }

    #[inline]
    pub fn get(&self, color: Color, pt: PieceType, to: Square) -> Move {
        self.table[color.index()][pt.index()][to as usize]
    }

    #[inline]
    pub fn store(&mut self, color: Color, pt: PieceType, to: Square, mv: Move) {
        self.table[color.index()][pt.index()][to as usize] = mv;
    }

    pub fn clear(&mut self) {
        self.table = [[[Move::NULL; 64]; 6]; 2];
    }
}

// ---- Continuation History [prev_pt][prev_to][cur_pt][cur_to] ----

pub struct ContHistory {
    table: Box<[[[[i16; 64]; 6]; 64]; 6]>,
}

impl ContHistory {
    pub fn new() -> Self {
        ContHistory { table: Box::new([[[[0i16; 64]; 6]; 64]; 6]) }
    }

    #[inline]
    pub fn get(&self, prev_pt: PieceType, prev_to: Square, cur_pt: PieceType, cur_to: Square) -> i32 {
        self.table[prev_pt.index()][prev_to as usize][cur_pt.index()][cur_to as usize] as i32
    }

    #[inline]
    pub fn update(&mut self, prev_pt: PieceType, prev_to: Square, cur_pt: PieceType, cur_to: Square, bonus: i32) {
        let entry = &mut self.table[prev_pt.index()][prev_to as usize][cur_pt.index()][cur_to as usize];
        let b = bonus.max(-400).min(400) as i16;
        *entry += b - (*entry as i32 * b.abs() as i32 / 16384) as i16;
    }

    pub fn age(&mut self) {
        for a in self.table.iter_mut() {
            for b in a.iter_mut() {
                for c in b.iter_mut() {
                    for d in c.iter_mut() {
                        *d /= 2;
                    }
                }
            }
        }
    }

    pub fn clear(&mut self) {
        for a in self.table.iter_mut() {
            for b in a.iter_mut() {
                for c in b.iter_mut() {
                    for d in c.iter_mut() {
                        *d = 0;
                    }
                }
            }
        }
    }
}

// ---- Capture History [attacker_piece_type][to_sq][victim_piece_type] ----

pub struct CaptureHistory {
    table: [[[i32; 6]; 64]; 6],
}

impl CaptureHistory {
    pub fn new() -> Self {
        CaptureHistory { table: [[[0; 6]; 64]; 6] }
    }

    #[inline]
    pub fn get(&self, attacker: PieceType, to: Square, victim: PieceType) -> i32 {
        self.table[attacker.index()][to as usize][victim.index()]
    }

    #[inline]
    pub fn update(&mut self, attacker: PieceType, to: Square, victim: PieceType, depth: i32, is_good: bool) {
        let bonus = if is_good { depth * depth } else { -(depth * depth) };
        let entry = &mut self.table[attacker.index()][to as usize][victim.index()];
        *entry += bonus - *entry * bonus.abs() / 16384;
    }

    pub fn age(&mut self) {
        for a in self.table.iter_mut() {
            for t in a.iter_mut() {
                for v in t.iter_mut() {
                    *v /= 2;
                }
            }
        }
    }

    pub fn clear(&mut self) {
        self.table = [[[0; 6]; 64]; 6];
    }
}

// ---- Move scoring ----

/// Score all moves in the list for ordering. Writes the first `list.len`
/// entries of `scores` (the rest may stay uninit — `pick_move` only reads
/// the same range, so the MaybeUninit invariant is preserved).
pub fn score_moves(
    pos: &Position,
    list: &mut MoveList,
    scores: &mut [MaybeUninit<i32>; MAX_MOVES],
    tt_move: Move,
    killers: &Killers,
    history: &History,
    counter_move: Move,
    cont_history: &ContHistory,
    capture_history: &CaptureHistory,
    prev_pt: PieceType,
    prev_to: Square,
    ply: usize,
) {
    let us = pos.side;

    for i in 0..list.len {
        let mv = list.moves[i];

        let s = if mv == tt_move {
            TT_SCORE
        } else if mv.is_capture() || mv.is_ep() {
            // MVV-LVA + SEE classification + capture history tie-breaker
            let victim_pt = if mv.is_ep() {
                PAWN
            } else {
                piece_type(pos.mailbox[mv.to() as usize])
            };
            let attacker_pt = piece_type(pos.mailbox[mv.from() as usize]);
            let mvv_lva = MVV_LVA[victim_pt.index()][attacker_pt.index()];
            let cap_hist = capture_history.get(attacker_pt, mv.to(), victim_pt) / 32;

            // Quick check: attacker <= victim is always good (skip expensive SEE)
            if (attacker_pt as u8) <= (victim_pt as u8) || victim_pt == QUEEN || see_ge(pos, mv, 0) {
                GOOD_CAPTURE_BASE + mvv_lva + cap_hist
            } else {
                BAD_CAPTURE + mvv_lva + cap_hist
            }
        } else if mv.is_promotion() {
            GOOD_CAPTURE_BASE + 60
        } else if let Some(killer_score) = killers.is_killer(ply, mv) {
            killer_score
        } else if mv == counter_move && counter_move != Move::NULL {
            COUNTER_MOVE_SCORE
        } else {
            // Quiet move: history + continuation history
            let h = history.get(us, mv);
            let ch = if prev_pt != PieceType::None {
                let cur_pt = piece_type(pos.mailbox[mv.from() as usize]);
                cont_history.get(prev_pt, prev_to, cur_pt, mv.to())
            } else { 0 };
            h + ch
        };
        scores[i] = MaybeUninit::new(s);
    }
}

// ---- Correction History [color][pawn_hash & MASK] ----
// Tracks average error between static eval and search score,
// used to correct static eval for pruning decisions.

const CORR_HIST_SIZE: usize = 16384;
const CORR_HIST_MASK: usize = CORR_HIST_SIZE - 1;
const CORR_HIST_MAX: i32 = 256 * 32; // clamp range

pub struct CorrectionHistory {
    table: Box<[[(i32, i32); CORR_HIST_SIZE]; 2]>, // (weighted_sum, weight)
}

impl CorrectionHistory {
    pub fn new() -> Self {
        CorrectionHistory { table: Box::new([[(0i32, 0i32); CORR_HIST_SIZE]; 2]) }
    }

    /// Get correction value for current position (in centipawns)
    pub fn get(&self, color: Color, pawn_hash: u64) -> i32 {
        let idx = pawn_hash as usize & CORR_HIST_MASK;
        let (sum, weight) = self.table[color.index()][idx];
        if weight == 0 { 0 } else { sum / weight }
    }

    /// Update correction after a search completes at a node.
    /// `diff` = search_score - static_eval, `depth` = search depth (used as weight)
    pub fn update(&mut self, color: Color, pawn_hash: u64, diff: i32, depth: i32) {
        let idx = pawn_hash as usize & CORR_HIST_MASK;
        let w = (depth as i32).min(16);
        let (ref mut sum, ref mut weight) = self.table[color.index()][idx];
        *sum += diff * w;
        *weight += w;
        // Decay to prevent unbounded growth: when weight gets large, halve both
        if *weight > 512 {
            *sum /= 2;
            *weight /= 2;
        }
        // Clamp the correction value
        if *weight > 0 {
            let corr = *sum / *weight;
            if corr > CORR_HIST_MAX {
                *sum = CORR_HIST_MAX * *weight;
            } else if corr < -CORR_HIST_MAX {
                *sum = -CORR_HIST_MAX * *weight;
            }
        }
    }

    pub fn clear(&mut self) {
        for color in self.table.iter_mut() {
            for entry in color.iter_mut() {
                *entry = (0, 0);
            }
        }
    }
}

/// Pick the best move from index `start` onwards (selection sort, one step)
/// Selection-sort one move into position `start`.
///
/// SAFETY: caller guarantees `scores[start..list.len]` are all initialized
/// (the `score_moves` loop wrote them).
pub fn pick_move(list: &mut MoveList, scores: &mut [MaybeUninit<i32>; MAX_MOVES], start: usize) -> Move {
    let mut best_idx = start;
    let mut best_score = unsafe { scores[start].assume_init() };

    for i in (start + 1)..list.len {
        let s = unsafe { scores[i].assume_init() };
        if s > best_score {
            best_score = s;
            best_idx = i;
        }
    }

    // Swap. MaybeUninit<T> is Copy when T is Copy, so this swaps the (possibly
    // initialized) cells just like the previous i32 swap did.
    if best_idx != start {
        list.moves.swap(start, best_idx);
        scores.swap(start, best_idx);
    }

    list.moves[start]
}
