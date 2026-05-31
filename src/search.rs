// ---- Search: PVS with Iterative Deepening ----
// Phase 4: IIR, Singular Extensions, Futility, LMP, SEE pruning, ProbCut,
//          Improving heuristic, Counter moves, Continuation history

use std::mem::MaybeUninit;

use crate::types::*;
use crate::board::Position;
use crate::movegen;
use crate::eval;
use crate::tt::{SharedTT, FLAG_EXACT, FLAG_LOWER, FLAG_UPPER, FLAG_NONE};
use crate::ordering::{self, Killers, History, CounterMoves, ContHistory, CaptureHistory, CorrectionHistory, MAX_PLY};
use crate::time::{self, TimeManager};
use crate::attacks;
use crate::nnue::{self, Nnue, Accumulator};
use crate::tune;
use crate::tablebase;

pub const INFINITY: i32 = 30000;
pub const MATE_SCORE: i32 = 29000;
pub const MATE_IN_MAX: i32 = MATE_SCORE - MAX_PLY as i32;
// TB-win scores live in [TB_WIN_IN_MAX, MATE_IN_MAX); pure mate scores in [MATE_IN_MAX, MATE_SCORE].
// Both ranges are ply-relative and need TT adjustment + UCI mate-display.
pub const TB_WIN_IN_MAX: i32 = MATE_IN_MAX - MAX_PLY as i32;

/// Is this a decisive (mate or TB-win) score?
fn is_mate_score(score: i32) -> bool {
    score.abs() >= TB_WIN_IN_MAX
}

/// Adjust mate/TB-win score for TT storage (convert ply-relative to position-relative)
fn score_to_tt(score: i32, ply: i32) -> i16 {
    if score >= TB_WIN_IN_MAX {
        (score + ply) as i16
    } else if score <= -TB_WIN_IN_MAX {
        (score - ply) as i16
    } else {
        score as i16
    }
}

/// Adjust mate/TB-win score from TT (convert position-relative to ply-relative)
fn score_from_tt(score: i16, ply: i32) -> i32 {
    let s = score as i32;
    if s >= TB_WIN_IN_MAX {
        s - ply
    } else if s <= -TB_WIN_IN_MAX {
        s + ply
    } else {
        s
    }
}

pub struct SearchInfo {
    pub nodes: u64,
    // Shared across all SMP threads: sum of helper-thread nodes.
    // Helpers periodically add their delta; main thread reads for UCI nps.
    pub helper_nodes: std::sync::Arc<std::sync::atomic::AtomicU64>,
    // Shared TB-probe-hit counter across SMP threads (UCI tbhits).
    pub tb_hits: std::sync::Arc<std::sync::atomic::AtomicU64>,
    pub synced_nodes: u64,
    pub tt: SharedTT,
    pub killers: Killers,
    pub history: History,
    pub counter_moves: CounterMoves,
    pub cont_history: ContHistory,
    pub cont_history_2: ContHistory,
    pub capture_history: CaptureHistory,
    pub correction_history: CorrectionHistory,
    pub pv: [[Move; MAX_PLY]; MAX_PLY],
    pub pv_len: [usize; MAX_PLY],
    pub seldepth: i32,
    // Per-ply tracking
    pub eval_stack: [i32; MAX_PLY],
    pub move_stack: [Move; MAX_PLY],
    pub piece_stack: [PieceType; MAX_PLY],
    pub reduction_stack: [i32; MAX_PLY],
    // NNUE
    pub nnue: Nnue,
    pub acc_stack: Vec<Accumulator>,
    // Repetition detection: hashes from game start + search
    pub hash_history: Vec<u64>,
    pub game_history_len: usize, // length of game history (before search starts)
    // Datagen support
    pub silent: bool,
    // Output gate (embedded lib): when false, suppress the stdout UCI info lines
    // WITHOUT changing search behaviour. Distinct from `silent`, which also skips
    // the root TB probe and routes node counts through the SMP helper counter.
    // UCI default true (prints as before); the embedded engine sets it false.
    pub print_info: bool,
    pub root_score: i32,
    // Deepest fully-completed iterative-deepening depth this search (0 if none).
    // Used by the experience-learning write path (uci.rs) as the save-depth gate.
    pub completed_depth: i32,
    // Node-based time management
    pub best_move_nodes: u64,
    // Forced-move detection (Patch B Phase 2): tracked per root pvs() call.
    // root_second_best is the highest non-best root score seen this call (i32::MIN if only one move).
    // root_moves_count is the number of root moves searched (1 => forced).
    pub root_second_best: i32,
    pub root_moves_count: u32,
    // Per-search stop flag (avoids global STOP race in multi-threaded datagen)
    pub stopped: bool,
}

impl SearchInfo {
    pub fn new(tt: SharedTT) -> Self {
        let mut acc_stack = Vec::with_capacity(MAX_PLY);
        for _ in 0..MAX_PLY {
            acc_stack.push(Accumulator::new());
        }
        SearchInfo {
            nodes: 0,
            helper_nodes: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            tb_hits: std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)),
            synced_nodes: 0,
            tt,
            killers: Killers::new(),
            history: History::new(),
            counter_moves: CounterMoves::new(),
            cont_history: ContHistory::new(),
            cont_history_2: ContHistory::new(),
            capture_history: CaptureHistory::new(),
            correction_history: CorrectionHistory::new(),
            pv: [[Move::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
            seldepth: 0,
            eval_stack: [0; MAX_PLY],
            move_stack: [Move::NULL; MAX_PLY],
            piece_stack: [PieceType::None; MAX_PLY],
            reduction_stack: [0; MAX_PLY],
            nnue: Nnue::new(),
            acc_stack,
            hash_history: Vec::with_capacity(512),
            game_history_len: 0,
            silent: false,
            print_info: true,
            root_score: 0,
            completed_depth: 0,
            best_move_nodes: 0,
            root_second_best: i32::MIN,
            root_moves_count: 0,
            stopped: false,
        }
    }

    pub fn clear_for_search(&mut self) {
        self.nodes = 0;
        self.synced_nodes = 0;
        // Main thread resets the shared helper counter (helpers will re-fill it).
        if !self.silent {
            self.helper_nodes.store(0, std::sync::atomic::Ordering::Relaxed);
            self.tb_hits.store(0, std::sync::atomic::Ordering::Relaxed);
        }
        self.killers.clear();
        self.seldepth = 0;
        self.history.age();
        self.cont_history.age();
        self.cont_history_2.age();
        self.capture_history.age();
        self.best_move_nodes = 0;
        self.completed_depth = 0;
        self.tt.new_generation();
        // Truncate any hashes added during previous search, keep game history
        self.hash_history.truncate(self.game_history_len);
        for i in 0..MAX_PLY {
            self.pv_len[i] = 0;
            self.eval_stack[i] = 0;
            self.move_stack[i] = Move::NULL;
            self.piece_stack[i] = PieceType::None;
            self.reduction_stack[i] = 0;
        }
    }
}

/// Material-based contempt for explicit-draw positions (50mvc, repetition).
/// When the side-to-move has a clear material advantage, treat a draw as
/// slightly negative — this nudges the engine to find any non-drawing move
/// (worth more than -CONTEMPT_VALUE cp) instead of shuffling into repetition.
/// Conversely, when losing, accept the draw eagerly.
/// Pure material — does not include positional factors so it stays cheap and
/// can be called from the early draw-detection block.
#[inline]
fn contempt_draw_score(pos: &Position) -> i32 {
    const CONTEMPT_THRESHOLD: i32 = 200;
    const CONTEMPT_VALUE: i32 = 10;
    const PIECE_VAL: [i32; 6] = [100, 320, 330, 500, 900, 0]; // P N B R Q K
    let stm = pos.side.index();
    let nstm = pos.side.flip().index();
    let mut mat: i32 = 0;
    for pt in 0..6 {
        let our = (pos.pieces[pt] & pos.colors[stm]).count_ones() as i32;
        let their = (pos.pieces[pt] & pos.colors[nstm]).count_ones() as i32;
        mat += (our - their) * PIECE_VAL[pt];
    }
    if mat > CONTEMPT_THRESHOLD {
        -CONTEMPT_VALUE
    } else if mat < -CONTEMPT_THRESHOLD {
        CONTEMPT_VALUE
    } else {
        0
    }
}

/// Check if the current position is a repetition.
/// Only checks positions since last irreversible move (capture, pawn move).
/// Returns true on 2-fold repetition (single repeat = draw during search).
#[inline]
fn is_repetition(info: &SearchInfo, hash: u64, halfmove_clock: u16) -> bool {
    let len = info.hash_history.len();
    if len < 3 { return false; }
    // Only look back halfmove_clock positions (since last irreversible move)
    let start = if (halfmove_clock as usize) < len {
        len - halfmove_clock as usize
    } else {
        0
    };
    // Current position is at hash_history[len-1] (pushed before recursive pvs call).
    // Same side to move is at len-3, len-5, len-7, ... (step by 2 from len-1).
    let mut i = len - 1;
    while i >= start + 2 {
        i -= 2;
        if info.hash_history[i] == hash {
            return true;
        }
    }
    false
}

/// Evaluate position using NNUE if loaded, otherwise HCE
#[inline]
fn evaluate_pos(pos: &Position, info: &SearchInfo, ply: usize) -> i32 {
    if info.nnue.is_loaded() {
        let bucket = crate::nnue::output_bucket_for_occupied(crate::bitboard::popcount(pos.occupancy()));
        info.nnue.evaluate(&info.acc_stack[ply], pos.side, bucket)
    } else {
        eval::evaluate(pos)
    }
}

/// Iterative deepening entry point
pub fn search(pos: &mut Position, info: &mut SearchInfo, tm: &mut TimeManager) -> Move {
    init_lmr();
    info.clear_for_search();
    info.stopped = false;

    // Initialize NNUE accumulator for root position
    if info.nnue.is_loaded() {
        info.nnue.refresh(pos, &mut info.acc_stack[0]);
    }

    let mut best_move = Move::NULL;
    let mut best_score = -INFINITY;

    if !info.silent && info.print_info {
        let tt_entries = info.tt.entry_count();
        let tt_mb = tt_entries * 16 / (1024 * 1024);
        println!("info string TT: {} entries ({}MB), hashfull at start: {}", tt_entries, tt_mb, info.tt.hashfull());
    }

    // Root TB probe: if in a TB position, play the DTZ-optimal move directly.
    // Skipped during datagen (silent) so iterative deepening still produces samples.
    if !info.silent {
        if let Some((mv_uci, wdl)) = tablebase::probe_root(pos) {
            let wdl_str = match wdl { 1 => "WIN", -1 => "LOSS", _ => "DRAW" };
            eprintln!("info string TB root probe: {} (move {})", wdl_str, mv_uci);
            if let Some(parsed) = pos.parse_uci_move(&mv_uci) {
                let tb_score = match wdl {
                    1  =>  TB_WIN_IN_MAX,
                    -1 => -TB_WIN_IN_MAX,
                    _  =>  0,
                };
                info.tb_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                info.pv[0][0] = parsed;
                info.pv_len[0] = 1;
                info.root_score = tb_score;

                if info.print_info {
                    let elapsed = tm.elapsed_ms().max(1);
                    print!("info depth 1 seldepth 1 score ");
                    if is_mate_score(tb_score) {
                        let mate_in = if tb_score > 0 {
                            (MATE_SCORE - tb_score + 1) / 2
                        } else {
                            -(MATE_SCORE + tb_score + 1) / 2
                        };
                        print!("mate {} ", mate_in);
                    } else {
                        print!("cp 0 ");
                    }
                    println!("nodes 1 nps 0 hashfull 0 tbhits 1 time {} pv {}", elapsed, mv_uci);
                }
                return parsed;
            }
        }
    }

    // Iterative deepening
    for depth in 1..=tm.max_depth() {
        if depth > 1 && (!tm.can_start_iteration() || tm.should_stop_soft(info.nodes)) {
            break;
        }

        let score = aspiration_search(pos, info, tm, depth, best_score);

        if (info.stopped || time::should_stop()) && depth > 1 {
            break;
        }

        best_score = score;
        // This iteration completed without an abort (the stopped/should_stop break above
        // returns before this point for depth>1), so `depth` is a fully-searched depth.
        info.completed_depth = depth;
        if info.pv_len[0] > 0 {
            best_move = info.pv[0][0];
        }

        // Print UCI info (unless silent mode for datagen, or print_info off for embedded)
        if !info.silent && info.print_info {
            let elapsed = tm.elapsed_ms().max(1);
            // Include helper-thread nodes for honest UCI nps under SMP.
            let total_nodes = info.nodes
                + info.helper_nodes.load(std::sync::atomic::Ordering::Relaxed);
            let nps = total_nodes * 1000 / elapsed as u64;
            let hashfull = info.tt.hashfull();

            print!("info depth {} seldepth {} score ", depth, info.seldepth);
            if is_mate_score(best_score) {
                let mate_in = if best_score > 0 {
                    (MATE_SCORE - best_score + 1) / 2
                } else {
                    -(MATE_SCORE + best_score + 1) / 2
                };
                print!("mate {} ", mate_in);
            } else {
                print!("cp {} ", best_score);
            }
            let tb_hits = info.tb_hits.load(std::sync::atomic::Ordering::Relaxed);
            print!("nodes {} nps {} hashfull {} tbhits {} time {} pv",
                total_nodes, nps, hashfull, tb_hits, elapsed);
            for i in 0..info.pv_len[0] {
                print!(" {}", info.pv[0][i]);
            }
            println!();
        }

        tm.update_score(best_score, depth);
        tm.update_best_move(best_move.0);

        // Node-based time management
        if info.best_move_nodes > 0 && info.nodes > 0 {
            tm.update_node_fraction(info.best_move_nodes, info.nodes);
        }

        // Patch B Phase 2: forced-move detection (1 legal => 0.01x; eval-diff>400 => 0.39x; >170 => 0.63x).
        // Skip on mate-score iterations: forced-detection vs mate-PV is misleading and the iteration
        // will break out anyway.
        if !is_mate_score(best_score) {
            let eval_diff = if info.root_second_best == i32::MIN {
                i32::MAX // only one root move tracked => effectively infinite gap
            } else {
                best_score - info.root_second_best
            };
            tm.update_forced(info.root_moves_count, eval_diff);
        }

        // Stop if we found mate
        if is_mate_score(best_score) {
            break;
        }
    }

    info.root_score = best_score;
    best_move
}

/// Aspiration window search
fn aspiration_search(
    pos: &mut Position,
    info: &mut SearchInfo,
    tm: &mut TimeManager,
    depth: i32,
    prev_score: i32,
) -> i32 {
    if depth <= 4 {
        return pvs(pos, info, tm, -INFINITY, INFINITY, depth, 0, false, Move::NULL, false);
    }

    let mut delta = tune::get(&tune::ASP_DELTA);
    let mut alpha = (prev_score - delta).max(-INFINITY);
    let mut beta = (prev_score + delta).min(INFINITY);

    loop {
        let score = pvs(pos, info, tm, alpha, beta, depth, 0, false, Move::NULL, false);

        if info.stopped || time::should_stop() {
            return score;
        }

        if score <= alpha {
            alpha = (score - delta).max(-INFINITY);
            beta = (score + delta).min(INFINITY);
        } else if score >= beta {
            beta = (score + delta).min(INFINITY);
        } else {
            return score;
        }

        delta += delta / 2;
        if delta > 500 {
            alpha = -INFINITY;
            beta = INFINITY;
        }
    }
}

/// Principal Variation Search (negamax)
fn pvs(
    pos: &mut Position,
    info: &mut SearchInfo,
    tm: &mut TimeManager,
    mut alpha: i32,
    mut beta: i32,
    mut depth: i32,
    ply: i32,
    is_null: bool,
    excluded: Move,
    cut_node: bool,
) -> i32 {
    let ply_u = ply as usize;
    info.pv_len[ply_u] = 0;
    let is_pv = beta - alpha > 1;
    let in_check = pos.is_in_check();
    let is_root = ply == 0;

    // Check extension (limit relative to nominal depth to avoid search explosion)
    if in_check && ply < depth as i32 + 16 { depth += 1; }

    // Quiescence at leaf
    if depth <= 0 {
        return quiescence(pos, info, tm, alpha, beta, ply);
    }

    // Node count + stop check
    info.nodes += 1;
    if info.nodes & 2047 == 0 {
        // Helper threads flush their node delta to the shared counter
        // so main thread's UCI nps reflects the whole cluster.
        if info.silent {
            let delta = info.nodes - info.synced_nodes;
            info.helper_nodes.fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
            info.synced_nodes = info.nodes;
        }
        if tm.should_stop_hard(info.nodes) {
            info.stopped = true;
            return 0;
        }
    }

    if ply >= MAX_PLY as i32 - 1 {
        return evaluate_pos(pos, info, ply_u);
    }

    // Seldepth tracking
    if ply > info.seldepth {
        info.seldepth = ply;
    }

    // Draw detection: fifty-move rule and repetition. Material-aware contempt
    // so the side with a material advantage avoids drawing into a repetition.
    // See [[rep_in_winning_endgame]] for the motivating Lichess incidents.
    if pos.halfmove_clock >= 100 {
        return contempt_draw_score(pos);
    }
    if !is_root && is_repetition(info, pos.hash, pos.halfmove_clock) {
        return contempt_draw_score(pos);
    }

    // Syzygy tablebase probe (non-root, non-excluded, sufficient depth)
    if !is_root && excluded == Move::NULL && !in_check
        && depth >= tablebase::probe_depth() as i32
    {
        if let Some(wdl) = tablebase::probe_wdl(pos) {
            info.tb_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            info.nodes += 1; // count TB probes as nodes
            let tb_score = match wdl {
                1 => MATE_SCORE - MAX_PLY as i32 - ply,   // TB win (large but below mate)
                -1 => -MATE_SCORE + MAX_PLY as i32 + ply, // TB loss
                _ => 0,                                      // TB draw
            };
            // Always trust tablebases: exact score
            info.tt.store(pos.hash, 127, FLAG_EXACT, is_pv, score_to_tt(tb_score, ply), 0, Move::NULL);
            return tb_score;
        }
    }

    // Mate distance pruning: if we already found a shorter mate, prune
    if !is_root {
        let mating = MATE_SCORE - ply;
        if mating < beta {
            beta = mating;
            if alpha >= mating {
                return mating;
            }
        }
        let mated = -MATE_SCORE + ply;
        if mated > alpha {
            alpha = mated;
            if beta <= mated {
                return mated;
            }
        }
    }

    // ---- TT Probe ----
    let tt_move;
    let mut tt_score = None;
    let mut tt_eval = None;
    let mut tt_depth: i8 = -1;
    let mut tt_flag: u8 = FLAG_NONE;
    let mut tt_pv = false;

    if let Some(entry) = info.tt.probe(pos.hash) {
        tt_move = entry.best_move;
        let s = score_from_tt(entry.score, ply);
        tt_eval = Some(entry.eval as i32);
        tt_depth = entry.depth;
        tt_flag = entry.flag;

        // TT cutoff: not at PV, not excluded (singular verification), sufficient depth
        if !is_pv && excluded == Move::NULL && entry.depth as i32 >= depth {
            match entry.flag & crate::tt::FLAG_TYPE_MASK {
                FLAG_EXACT => return s,
                FLAG_LOWER if s >= beta => return s,
                FLAG_UPPER if s <= alpha => return s,
                _ => {}
            }
        }
        tt_score = Some(s);
        tt_pv = (entry.flag & crate::tt::FLAG_WAS_PV) != 0 || (entry.flag & crate::tt::FLAG_TYPE_MASK) == FLAG_EXACT;
    } else {
        tt_move = Move::NULL;
    }

    // ---- Ensure NNUE accumulator is computed ----
    if info.nnue.is_loaded() && !info.acc_stack[ply_u].computed {
        info.nnue.refresh(pos, &mut info.acc_stack[ply_u]);
    }

    // ---- Static Eval with Correction History ----
    let raw_eval = if in_check {
        -INFINITY
    } else if let Some(e) = tt_eval {
        e
    } else {
        evaluate_pos(pos, info, ply_u)
    };

    let static_eval = if in_check {
        raw_eval
    } else {
        raw_eval + info.correction_history.get(pos.side, pos.pawn_hash)
    };

    info.eval_stack[ply_u] = static_eval;

    // Improving: is our static eval better than 2 plies ago?
    let improving = !in_check && ply >= 2 && static_eval > info.eval_stack[ply_u - 2];

    // ---- IIR: Internal Iterative Reduction ----
    if tt_move == Move::NULL && depth >= 4 && !in_check {
        depth -= 1;
    }

    // ---- Pre-move Pruning (non-PV, not in check, not excluded for SE) ----
    if !is_pv && !in_check && excluded == Move::NULL {
        // Reverse Futility Pruning (margin adjusts with improving)
        let rfp_margin = if improving { tune::get(&tune::RFP_MARGIN_IMP) } else { tune::get(&tune::RFP_MARGIN_NIMP) };
        if depth <= tune::get(&tune::RFP_DEPTH) && static_eval - rfp_margin * depth >= beta && !is_mate_score(beta) {
            return static_eval;
        }

        // Null Move Pruning
        if !is_null && depth >= 3 && static_eval >= beta
            && pos.non_pawn_material(pos.side) > 0
        {
            let r = tune::get(&tune::NMP_BASE) + depth / tune::get(&tune::NMP_DEPTH_DIV) + ((static_eval - beta) / tune::get(&tune::NMP_EVAL_DIV)).min(tune::get(&tune::NMP_EVAL_MAX)) + (!improving as i32);
            // Make null move
            let old_ep = pos.ep_square;
            let old_hash = pos.hash;
            if pos.ep_square != NO_SQ {
                pos.hash ^= crate::zobrist::ep_key(crate::types::file_of(pos.ep_square));
            }
            pos.ep_square = NO_SQ;
            pos.side = pos.side.flip();
            pos.hash ^= crate::zobrist::side_key();

            // Store null move info in stacks
            info.move_stack[ply_u] = Move::NULL;
            info.piece_stack[ply_u] = PieceType::None;

            // NNUE: copy accumulator (position unchanged, just side flips).
            // Borrow parent and child via split_at_mut to avoid cloning a ~4 KB accumulator.
            if info.nnue.is_loaded() {
                let (left, right) = info.acc_stack.split_at_mut(ply_u + 1);
                info.nnue.copy_acc(&left[ply_u], &mut right[0]);
            }

            info.hash_history.push(pos.hash);
            let score = -pvs(pos, info, tm, -beta, -beta + 1, depth - r - 1, ply + 1, true, Move::NULL, !cut_node);

            // Unmake null move
            pos.side = pos.side.flip();
            pos.ep_square = old_ep;
            pos.hash = old_hash;
            info.hash_history.pop();

            if info.stopped || time::should_stop() { return 0; }
            if score >= beta {
                return beta;
            }
        }

        // Razoring
        if depth <= tune::get(&tune::RAZOR_DEPTH) && static_eval + tune::get(&tune::RAZOR_MARGIN) * depth < alpha {
            let score = quiescence(pos, info, tm, alpha, beta, ply);
            if score <= alpha {
                return score;
            }
        }

        // ProbCut: at high depths, do a shallow search to verify if position is much above beta
        if depth >= tune::get(&tune::PROBCUT_DEPTH) && !is_mate_score(beta) {
            let pb_beta = beta + tune::get(&tune::PROBCUT_MARGIN);
            let pb_depth = depth - 4;

            let mut pb_list = MoveList::new();
            movegen::generate_captures(pos, &mut pb_list);

            let mut pb_scores = [MaybeUninit::<i32>::uninit(); MAX_MOVES];
            for i in 0..pb_list.len {
                let mv = pb_list.moves[i];
                let victim_pt = if mv.is_ep() { PAWN } else { piece_type(pos.mailbox[mv.to() as usize]) };
                let attacker_pt = piece_type(pos.mailbox[mv.from() as usize]);
                pb_scores[i] = MaybeUninit::new((victim_pt as i32) * 10 - (attacker_pt as i32));
            }

            for i in 0..pb_list.len {
                let mv = ordering::pick_move(&mut pb_list, &mut pb_scores, i);

                if !ordering::see_ge(pos, mv, pb_beta - static_eval) {
                    continue;
                }

                // NNUE: compute delta before make_move
                let pb_delta = if info.nnue.is_loaded() {
                    Some(nnue::compute_delta(pos, mv))
                } else {
                    None
                };

                let undo = pos.make_move(mv);
                info.tt.prefetch(pos.hash);
                let ksq = pos.king_sq(pos.side.flip());
                if ksq >= 64 || attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side)
                    || pos.king_sq(pos.side) >= 64
                {
                    pos.unmake_move(mv, undo);
                    continue;
                }

                // NNUE: incrementally update accumulator
                if let Some(ref delta) = pb_delta {
                    if (delta.king_bucket_changed && info.nnue.is_bucketed()) || info.nnue.needs_full_refresh_per_move() {
                        info.nnue.refresh(pos, &mut info.acc_stack[ply_u + 1]);
                    } else {
                        let (parent, child) = if ply_u + 1 < info.acc_stack.len() {
                            let (left, right) = info.acc_stack.split_at_mut(ply_u + 1);
                            (&left[ply_u], &mut right[0])
                        } else {
                            unreachable!()
                        };
                        info.nnue.update_inc(parent, child, delta);
                    }
                }

                info.hash_history.push(pos.hash);
                let score = -pvs(pos, info, tm, -pb_beta, -pb_beta + 1, pb_depth, ply + 1, false, Move::NULL, !cut_node);
                pos.unmake_move(mv, undo);
                info.hash_history.pop();

                if info.stopped || time::should_stop() { return 0; }
                if score >= pb_beta {
                    return score;
                }
            }
        }
    }

    // ---- Move Generation + Ordering ----
    let mut list = MoveList::new();
    movegen::generate_all(pos, &mut list);

    let (prev_pt, prev_to) = if ply > 0 {
        (info.piece_stack[ply_u - 1], info.move_stack[ply_u - 1].to())
    } else {
        (PieceType::None, 0)
    };

    let (pp_pt, pp_to) = if ply >= 2 {
        (info.piece_stack[ply_u - 2], info.move_stack[ply_u - 2].to())
    } else {
        (PieceType::None, 0)
    };

    let counter_move = if prev_pt != PieceType::None {
        info.counter_moves.get(pos.side.flip(), prev_pt, prev_to)
    } else {
        Move::NULL
    };

    let mut scores = [MaybeUninit::<i32>::uninit(); MAX_MOVES];
    ordering::score_moves(
        pos, &mut list, &mut scores, tt_move,
        &info.killers, &info.history, counter_move,
        &info.cont_history, &info.capture_history,
        prev_pt, prev_to, ply_u,
    );

    let mut best_score = -INFINITY;
    let mut best_move = Move::NULL;
    let mut moves_searched = 0u32;
    let old_alpha = alpha;

    // Patch B Phase 2 (Forced-Move Detection): reset per-call root tracking.
    // The LAST aspiration_search pvs() to return successfully overwrites this,
    // so post-aspiration info reflects the final root iteration.
    if is_root {
        info.root_second_best = i32::MIN;
        info.root_moves_count = 0;
    }

    // Track quiet moves for history penalty
    let mut quiet_moves = [Move::NULL; MAX_MOVES];
    let mut quiet_count = 0usize;

    // LMP threshold
    let lmp_base = tune::get(&tune::LMP_BASE);
    let lmp_threshold = if improving { lmp_base + depth * depth } else { (lmp_base + depth * depth) / 2 };

    // Futility flag
    let can_futility = !is_pv && !in_check && depth <= tune::get(&tune::FP_DEPTH)
        && static_eval + depth * tune::get(&tune::FP_MARGIN_MUL) + tune::get(&tune::FP_MARGIN_ADD) <= alpha;

    for i in 0..list.len {
        let mv = ordering::pick_move(&mut list, &mut scores, i);
        // Skip excluded move (for singular extension verification)
        if mv == excluded && excluded != Move::NULL {
            continue;
        }

        let is_quiet_move = !mv.is_capture() && !mv.is_promotion() && !mv.is_ep();
        // SAFETY: score_moves wrote all positions in 0..list.len; pick_move
        // preserves initialization on swap. We're inside `for i in 0..list.len`.
        let hist_score = unsafe { scores[i].assume_init() };

        // ---- Pre-move pruning (non-root, non-PV, not in check) ----
        if !is_root && !is_pv && !in_check && best_score > -MATE_IN_MAX && moves_searched > 0 {
            // Late Move Pruning
            if is_quiet_move && moves_searched >= lmp_threshold as u32 {
                continue;
            }

            // History Pruning: skip quiet moves with very negative history
            if is_quiet_move && depth <= tune::get(&tune::HIST_PRUNE_DEPTH) && hist_score < -tune::get(&tune::HIST_PRUNE_MARGIN) * depth {
                continue;
            }

            // Futility Pruning for quiet moves
            if is_quiet_move && can_futility {
                continue;
            }

            // SEE-based pruning
            if depth <= tune::get(&tune::SEE_DEPTH) {
                let see_thresh = if mv.is_capture() || mv.is_ep() {
                    -depth * tune::get(&tune::SEE_CAP_MUL)
                } else {
                    -depth * tune::get(&tune::SEE_QUIET_MUL)
                };
                if !ordering::see_ge(pos, mv, see_thresh) {
                    continue;
                }
            }
        }

        // ---- Singular Extension ----
        let mut extension = 0;
        if depth >= tune::get(&tune::SE_DEPTH) && mv == tt_move && excluded == Move::NULL
            && tt_score.is_some() && !is_mate_score(tt_score.unwrap())
            && tt_depth as i32 >= depth - 3
            && (tt_flag == FLAG_LOWER || tt_flag == FLAG_EXACT)
        {
            let se_beta = tt_score.unwrap() - depth * tune::get(&tune::SE_MARGIN_MUL);
            let se_depth = (depth - 1) / 2;
            let se_score = pvs(pos, info, tm, se_beta - 1, se_beta, se_depth, ply, false, mv, cut_node);

            if info.stopped || time::should_stop() { return 0; }

            if se_score < se_beta {
                extension = 1; // Singular: this move is much better than alternatives
            } else if se_score >= beta {
                return se_score; // Multi-cut: even without TT move, we're >= beta
            } else if cut_node {
                // Negative extension: not singular at expected cut node
                extension = -1;
            }
        }

        // Passed pawn extension: pawn push to 6th/7th rank (close to promotion)
        if extension == 0 && !mv.is_promotion() {
            let from_pc = pos.mailbox[mv.from() as usize];
            if from_pc != EMPTY && piece_type(from_pc) == PAWN {
                let to_rank = mv.to() / 8;
                if (pos.side == WHITE && to_rank >= 5) || (pos.side == BLACK && to_rank <= 2) {
                    extension = 1;
                }
            }
        }

        // NNUE: compute delta before make_move
        let mv_delta = if info.nnue.is_loaded() {
            Some(nnue::compute_delta(pos, mv))
        } else {
            None
        };

        let undo = pos.make_move(mv);
        info.tt.prefetch(pos.hash);

        // Legality check (guard against invalid king sq from hash collisions)
        let ksq = pos.king_sq(pos.side.flip());
        if ksq >= 64 || attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side) {
            pos.unmake_move(mv, undo);
            continue;
        }
        if pos.king_sq(pos.side) >= 64 {
            pos.unmake_move(mv, undo);
            continue;
        }

        // Push hash for repetition detection
        info.hash_history.push(pos.hash);

        // NNUE: incrementally update accumulator (bucketed: refresh on king moves)
        if let Some(ref delta) = mv_delta {
            if (delta.king_bucket_changed && info.nnue.is_bucketed()) || info.nnue.needs_full_refresh_per_move() {
                info.nnue.refresh(pos, &mut info.acc_stack[ply_u + 1]);
            } else {
                let (parent, child) = {
                    let (left, right) = info.acc_stack.split_at_mut(ply_u + 1);
                    (&left[ply_u], &mut right[0])
                };
                info.nnue.update_inc(parent, child, delta);
            }
        }

        moves_searched += 1;

        // Track quiet moves for history penalty on cutoff
        if is_quiet_move {
            if quiet_count < MAX_MOVES {
                quiet_moves[quiet_count] = mv;
                quiet_count += 1;
            }
        }

        // Store move info in stacks
        let moved_piece = pos.mailbox[mv.to() as usize];
        let moved_pt = if mv.is_promotion() {
            mv.promo_piece_type()
        } else if moved_piece != EMPTY {
            piece_type(moved_piece)
        } else {
            PieceType::None
        };
        info.move_stack[ply_u] = mv;
        info.piece_stack[ply_u] = moved_pt;

        let new_depth = depth - 1 + extension;
        let mut score;

        // Node counting for root time management
        let nodes_before = if is_root { info.nodes } else { 0 };

        // LMR: Late Move Reductions (quiets only)
        if moves_searched > 3 && depth >= 3 && is_quiet_move && !in_check {
            // gives_check is only consumed here, so the king-attack scan is deferred
            // into the LMR gate — skipped for first moves, captures, and depth < 3.
            let gives_check = pos.is_in_check();
            let r = lmr_reduction(depth, moves_searched, is_pv, tt_pv, improving, hist_score, cut_node, static_eval, alpha, gives_check);
            let reduced_depth = (new_depth - r).max(1);
            score = -pvs(pos, info, tm, -alpha - 1, -alpha, reduced_depth, ply + 1, false, Move::NULL, !cut_node);

            // Re-search at full depth if it improved alpha
            if score > alpha && reduced_depth < new_depth {
                score = -pvs(pos, info, tm, -alpha - 1, -alpha, new_depth, ply + 1, false, Move::NULL, !cut_node);
            }
        } else if !is_pv || moves_searched > 1 {
            // Non-PV: null window search
            score = -pvs(pos, info, tm, -alpha - 1, -alpha, new_depth, ply + 1, false, Move::NULL, !cut_node);
        } else {
            // First move in PV: full window
            score = alpha + 1; // Force re-search below
        }

        // PV re-search at full window
        if is_pv && (moves_searched == 1 || score > alpha) {
            score = -pvs(pos, info, tm, -beta, -alpha, new_depth, ply + 1, false, Move::NULL, false);
        }

        pos.unmake_move(mv, undo);
        info.hash_history.pop();

        if info.stopped || time::should_stop() { return 0; }

        // Root: track node fraction for best move (v02 form — first-move heuristic).
        if is_root && moves_searched == 1 {
            info.best_move_nodes = info.nodes - nodes_before;
        }

        // Patch B Phase 2: root tracking for forced-move detection.
        // Track moves seen + the highest non-best score (2nd-best).
        if is_root {
            info.root_moves_count += 1;
            if score > best_score {
                if best_score != -INFINITY && best_score > info.root_second_best {
                    info.root_second_best = best_score;
                }
            } else if score > info.root_second_best {
                info.root_second_best = score;
            }
        }

        if score > best_score {
            best_score = score;
            best_move = mv;

            if score > alpha {
                alpha = score;

                // Update PV
                info.pv[ply_u][0] = mv;
                let next_len = info.pv_len[ply_u + 1];
                for j in 0..next_len {
                    info.pv[ply_u][j + 1] = info.pv[ply_u + 1][j];
                }
                info.pv_len[ply_u] = next_len + 1;

                if score >= beta {
                    // Beta cutoff: update move ordering heuristics
                    if is_quiet_move {
                        info.killers.store(ply_u, mv);
                        info.history.update(pos.side, mv, depth, true);

                        // Counter move
                        if prev_pt != PieceType::None {
                            info.counter_moves.store(pos.side.flip(), prev_pt, prev_to, mv);
                        }

                        // 1-ply continuation history: bonus for cutoff move
                        if prev_pt != PieceType::None {
                            let cur_pt = piece_type(pos.mailbox[mv.from() as usize]);
                            info.cont_history.update(prev_pt, prev_to, cur_pt, mv.to(), depth * depth);
                        }

                        // 2-ply follow-up history: bonus for cutoff move
                        if pp_pt != PieceType::None {
                            let cur_pt = piece_type(pos.mailbox[mv.from() as usize]);
                            info.cont_history_2.update(pp_pt, pp_to, cur_pt, mv.to(), depth * depth);
                        }

                        // History penalty for all quiet moves that didn't cause cutoff
                        for q in 0..quiet_count {
                            let qm = quiet_moves[q];
                            if qm != mv {
                                info.history.update(pos.side, qm, depth, false);
                                if prev_pt != PieceType::None || pp_pt != PieceType::None {
                                    let qpt = piece_type(pos.mailbox[qm.from() as usize]);
                                    if prev_pt != PieceType::None {
                                        info.cont_history.update(prev_pt, prev_to, qpt, qm.to(), -(depth * depth));
                                    }
                                    if pp_pt != PieceType::None {
                                        info.cont_history_2.update(pp_pt, pp_to, qpt, qm.to(), -(depth * depth));
                                    }
                                }
                            }
                        }
                    } else if mv.is_capture() || mv.is_ep() {
                        // Capture history: bonus for capture that caused cutoff
                        let cap_victim = if mv.is_ep() { PAWN } else { piece_type(pos.mailbox[mv.to() as usize]) };
                        let cap_attacker = piece_type(pos.mailbox[mv.from() as usize]);
                        info.capture_history.update(cap_attacker, mv.to(), cap_victim, depth, true);
                    }

                    break;
                }
            }
        }
    }

    // Checkmate or stalemate
    if moves_searched == 0 {
        if excluded != Move::NULL {
            // In singular extension verification, don't return mate/stalemate
            return alpha;
        }
        if in_check {
            return -MATE_SCORE + ply; // checkmate
        } else {
            return 0; // stalemate
        }
    }

    // Store in TT (don't store if excluded move search)
    if excluded == Move::NULL {
        let flag = if best_score >= beta {
            FLAG_LOWER
        } else if alpha != old_alpha {
            FLAG_EXACT
        } else {
            FLAG_UPPER
        };

        info.tt.store(
            pos.hash,
            depth as i8,
            flag,
            is_pv || tt_pv,
            score_to_tt(best_score, ply),
            raw_eval as i16,
            best_move,
        );

        // Update correction histories: learn from the difference between
        // raw static eval and search result (only for non-mate, non-check, quiet nodes)
        if !in_check && !is_mate_score(best_score) && raw_eval != -INFINITY {
            let diff = best_score - raw_eval;
            info.correction_history.update(pos.side, pos.pawn_hash, diff, depth);
        }
    }

    best_score
}

/// Quiescence search — only captures
fn quiescence(
    pos: &mut Position,
    info: &mut SearchInfo,
    tm: &mut TimeManager,
    mut alpha: i32,
    beta: i32,
    ply: i32,
) -> i32 {
    info.nodes += 1;

    if info.nodes & 2047 == 0 {
        // Helper threads flush their node delta to the shared counter
        // so main thread's UCI nps reflects the whole cluster.
        if info.silent {
            let delta = info.nodes - info.synced_nodes;
            info.helper_nodes.fetch_add(delta, std::sync::atomic::Ordering::Relaxed);
            info.synced_nodes = info.nodes;
        }
        if tm.should_stop_hard(info.nodes) {
            info.stopped = true;
            return 0;
        }
    }

    if ply > info.seldepth {
        info.seldepth = ply;
    }

    let ply_u = ply as usize;

    if ply >= MAX_PLY as i32 - 1 {
        return evaluate_pos(pos, info, ply_u);
    }

    // Ensure NNUE accumulator is computed
    if info.nnue.is_loaded() && !info.acc_stack[ply_u].computed {
        info.nnue.refresh(pos, &mut info.acc_stack[ply_u]);
    }

    // Only handle check evasions if not too deep (prevent search explosion)
    let in_check = pos.is_in_check() && ply < 80;

    // TT probe in qsearch
    let mut qs_tt_move = Move::NULL;
    if let Some(entry) = info.tt.probe(pos.hash) {
        let tt_s = score_from_tt(entry.score, ply);
        qs_tt_move = entry.best_move;
        // TT cutoff in qsearch (not when in check — need to verify we have evasions)
        if !in_check {
            match entry.flag & crate::tt::FLAG_TYPE_MASK {
                FLAG_EXACT => return tt_s,
                FLAG_LOWER if tt_s >= beta => return tt_s,
                FLAG_UPPER if tt_s <= alpha => return tt_s,
                _ => {}
            }
        }
    }

    // Standing pat (not allowed when in check — must find an evasion)
    let stand_pat = if in_check {
        -INFINITY
    } else {
        evaluate_pos(pos, info, ply_u)
    };

    if !in_check {
        if stand_pat >= beta {
            return beta;
        }
    }
    let old_alpha = alpha;
    if stand_pat > alpha {
        alpha = stand_pat;
    }

    // Generate moves: all evasions when in check, captures only otherwise
    let mut list = MoveList::new();
    if in_check {
        movegen::generate_all(pos, &mut list);
    } else {
        movegen::generate_captures(pos, &mut list);
    }

    // Move scoring: TT move first, then MVV-LVA for captures, 0 for quiet evasions.
    // Every position 0..list.len is written below — quiet evasions get an explicit 0.
    let mut scores = [MaybeUninit::<i32>::uninit(); MAX_MOVES];
    for i in 0..list.len {
        let mv = list.moves[i];
        let s = if mv == qs_tt_move {
            10_000_000
        } else if mv.is_capture() || mv.is_ep() {
            let victim_pt = if mv.is_ep() {
                PAWN
            } else {
                piece_type(pos.mailbox[mv.to() as usize])
            };
            let attacker_pt = piece_type(pos.mailbox[mv.from() as usize]);
            (victim_pt as i32) * 10 - (attacker_pt as i32)
        } else {
            // Quiet evasion — searched after captures
            0
        };
        scores[i] = MaybeUninit::new(s);
    }

    let mut best_qs_move = Move::NULL;
    let mut legal_moves = 0u32;

    for i in 0..list.len {
        let mv = ordering::pick_move(&mut list, &mut scores, i);

        // Pruning only for captures when not in check
        if !in_check {
            // Delta pruning: skip captures that can't possibly improve alpha
            let victim_val = match piece_type(pos.mailbox[mv.to() as usize]) {
                PieceType::Pawn => 100,
                PieceType::Knight => 320,
                PieceType::Bishop => 330,
                PieceType::Rook => 500,
                PieceType::Queen => 1000,
                _ => 0,
            };
            if stand_pat + victim_val + 200 < alpha && !mv.is_promotion() {
                continue;
            }

            // SEE pruning in qsearch: skip losing captures
            let attacker_pt = piece_type(pos.mailbox[mv.from() as usize]);
            let victim_pt = if mv.is_ep() { PAWN } else { piece_type(pos.mailbox[mv.to() as usize]) };
            if (attacker_pt as u8) > (victim_pt as u8) && victim_pt != QUEEN {
                if !ordering::see_ge(pos, mv, 0) {
                    continue;
                }
            }
        }

        // NNUE: compute delta before make_move
        let qs_delta = if info.nnue.is_loaded() {
            Some(nnue::compute_delta(pos, mv))
        } else {
            None
        };

        let undo = pos.make_move(mv);
        info.tt.prefetch(pos.hash);

        // Legality check (guard against invalid king sq from hash collisions)
        let ksq = pos.king_sq(pos.side.flip());
        if ksq >= 64 || attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side)
            || pos.king_sq(pos.side) >= 64
        {
            pos.unmake_move(mv, undo);
            continue;
        }

        legal_moves += 1;

        // NNUE: incrementally update accumulator (bucketed: refresh on king moves)
        if let Some(ref delta) = qs_delta {
            if (delta.king_bucket_changed && info.nnue.is_bucketed()) || info.nnue.needs_full_refresh_per_move() {
                info.nnue.refresh(pos, &mut info.acc_stack[ply_u + 1]);
            } else {
                let (parent, child) = {
                    let (left, right) = info.acc_stack.split_at_mut(ply_u + 1);
                    (&left[ply_u], &mut right[0])
                };
                info.nnue.update_inc(parent, child, delta);
            }
        }

        let score = -quiescence(pos, info, tm, -beta, -alpha, ply + 1);

        pos.unmake_move(mv, undo);

        if info.stopped || time::should_stop() { return 0; }

        if score >= beta {
            // Store TT entry for beta cutoff
            info.tt.store(pos.hash, 0, FLAG_LOWER, beta - old_alpha > 1, score_to_tt(score, ply), stand_pat as i16, mv);
            return beta;
        }
        if score > alpha {
            alpha = score;
            best_qs_move = mv;
        }
    }

    // Checkmate detection in qsearch (in check, no legal moves)
    if in_check && legal_moves == 0 {
        return -MATE_SCORE + ply;
    }

    // Store TT entry
    let qs_flag = if alpha > old_alpha { FLAG_EXACT } else { FLAG_UPPER };
    let eval_store = if in_check { 0i16 } else { stand_pat as i16 };
    info.tt.store(pos.hash, 0, qs_flag, beta - old_alpha > 1, score_to_tt(alpha, ply), eval_store, best_qs_move);

    alpha
}

/// Pre-computed LMR reduction table: LMR_TABLE[depth][moves_searched]
/// Avoids ln() calls in the hot path.
static mut LMR_TABLE: [[i32; MAX_PLY]; MAX_PLY] = [[0i32; MAX_PLY]; MAX_PLY];

/// Initialize LMR table at startup (called from main.rs) and at the start of each
/// search to pick up SPSA-tuned LMR_DIVISOR changes via setoption.
pub fn init_lmr() {
    let divisor = tune::get(&tune::LMR_DIVISOR) as f64 / 100.0;
    unsafe {
        for d in 1..MAX_PLY {
            for m in 1..MAX_PLY {
                LMR_TABLE[d][m] = ((d as f64).ln() * (m as f64).ln() / divisor) as i32;
            }
        }
    }
}

/// LMR reduction with history, improving, cut-node, eval-gap, and gives-check adjustments
// LMR_TABLE is filled once in init_lmr() at startup, then only read.
#[allow(static_mut_refs)]
#[inline]
fn lmr_reduction(depth: i32, moves_searched: u32, is_pv: bool, tt_pv: bool, improving: bool, hist_score: i32, cut_node: bool, static_eval: i32, alpha: i32, gives_check: bool) -> i32 {
    let d = (depth as usize).min(MAX_PLY - 1);
    let m = (moves_searched as usize).min(MAX_PLY - 1);
    // SAFETY: d, m are clamped to MAX_PLY-1; LMR_TABLE is [[i32; MAX_PLY]; MAX_PLY].
    let mut r = unsafe { *LMR_TABLE.get_unchecked(d).get_unchecked(m) };

    // PV / TT-PV nodes: reduce less (PV now or was PV in previous search)
    if is_pv || tt_pv { r -= 1; }

    // Not improving: reduce more
    if !improving { r += 1; }

    // Cut nodes: reduce more (expected to fail high, so less effort on alternatives)
    if cut_node { r += 1; }

    // Good history: reduce less; bad history: reduce more
    r -= (hist_score / tune::get(&tune::LMR_HIST_DIV)).max(-2).min(2);

    // Eval gap: when static eval is well above alpha, reduce more aggressively
    // Position is likely winning, so less effort on late quiet moves
    if static_eval - alpha >= 150 {
        r += 1;
    }

    // Gives check: tactically important, reduce less
    if gives_check {
        r -= 1;
    }

    r.max(0).min(depth - 2)
}
