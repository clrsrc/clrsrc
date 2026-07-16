// ---- clrsrc self-play book generation (JBK2 overlay) ----
// Plays clrsrc-vs-clrsrc games and records the engine's OWN opening choices as JBK2 book entries,
// so the resulting book is tuned for *clrsrc* (move selection + WDL reflect clrsrc's own play),
// unlike a book built from jugernaut self-play or human Lichess games.
//
// Usage: clrsrc genbook <games> <soft_nodes> <output.overlay> <threads> [diversity_book.book]
//
// Diversity source: a JBK2 book (v4 main book) — every move in the recording window first tries
// the diversity book with variety=2 (±30cp + WDL + count-floor pool, uniform random pick). If a
// book move is returned, clrsrc plays it and a quick mini-search produces clrsrc's own score for
// that position; if the book is out-of-line, clrsrc full-searches and the chosen move is recorded.
// Both kinds of moves land in the overlay → recording window = [0, RECORD_PLIES) from move 1.
// Rationale: by anchoring self-play to v4's main-line distribution, the resulting overlay overlaps
// the positions the A/B test will actually reach (when run from startpos with BookVariety>0).
//
// Output is a JBK2 OVERLAY (source = SELFPLAY|clrsrc = 0x28, cp eval in clrsrc_score), the same
// format/convention the lichess-bot harvest writes. Fold it into a main book offline with
//   clrsrc expmerge <base.book> <this.overlay> <out.book> --clrsrc-mirror
// then A/B-SPRT the result against the base book *with the clrsrc binary*, from startpos with
// BookVariety>0 (ab_book_sprt.sh).

use crate::types::*;
use crate::board::Position;
use crate::movegen;
use crate::search::{self, SearchInfo, MINIMUM_MATE_SCORE};
use crate::time::{TimeControl, TimeManager};
use crate::attacks;
use crate::book::{
    self, ExpBook, ExpEntry, OverlayWriter,
    EXP_SOURCE_ENGINE, EXP_SOURCE_SELFPLAY, EXP_FLAG_VALIDATED, EXP_FLAG_MATE, EXP_SCORE_NONE,
};

use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// ---- Recording window ----
/// Last ply (exclusive) recorded into the book. Self-play records clrsrc's chosen moves for plies
/// [0, RECORD_PLIES). 16 matched the lichess-bot harvest depth; 18 (pilot3, 29.05) extends the
/// window 2 plies to bank time + carry clrsrc's vetted picks deeper into the early middlegame.
/// At plies 16-17 v4's diversity coverage thins out, so those entries fall through to a full
/// soft_nodes search — i.e. clrsrc's own full-search pick, not the 10k mini-search of book hits.
/// NB: now deeper than the bot's 16-ply harvest — the genbook candidate is a separate A/B book.
const RECORD_PLIES: u32 = 18;
/// BookVariety used internally to probe the diversity book during self-play. Capped at 2 by
/// `ExpBook::probe_best` (±30cp tolerance), beyond that has no effect.
const DIVERSITY_VARIETY: u8 = 2;
/// Soft-node budget for the mini-search that produces clrsrc's own score for a book-played move.
/// Same budget the previous opening-verify pass used — quick but stable enough for the field.
const MINI_SEARCH_NODES: u64 = 10_000;

// ---- Game termination / adjudication (mirrors datagen for speed) ----
const MAX_GAME_MOVES: u32 = 512;
const WIN_ADJ_SCORE: i32 = 3000;
const WIN_ADJ_COUNT: i32 = 5;
const DRAW_ADJ_SCORE: i32 = 10;
const DRAW_ADJ_COUNT: i32 = 10;
const DRAW_ADJ_MIN_PLY: u32 = 70;
const HARD_NODE_LIMIT: u64 = 10_000_000;

// ---- xoshiro256** PRNG (same as datagen) ----
struct Rng {
    state: [u64; 4],
}
impl Rng {
    fn new(seed: u64) -> Self {
        let mut s = [0u64; 4];
        let mut z = seed;
        for i in 0..4 {
            z = z.wrapping_add(0x9e3779b97f4a7c15);
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
            s[i] = z ^ (z >> 31);
        }
        Rng { state: s }
    }
    fn next_u64(&mut self) -> u64 {
        let result = self.state[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }
}

fn get_legal_moves(pos: &mut Position) -> Vec<Move> {
    let mut list = MoveList::new();
    movegen::generate_all(pos, &mut list);
    let mut legal = Vec::new();
    for i in 0..list.len {
        let mv = list.moves[i];
        let undo = pos.make_move(mv);
        let ksq = pos.king_sq(pos.side.flip());
        if !attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side) {
            legal.push(mv);
        }
        pos.unmake_move(mv, undo);
    }
    legal
}

/// A recorded opening decision, pending the game's final result (which fills WDL).
struct PendingEntry {
    key: u64,
    packed_move: u16,
    clrsrc_score: i16,
    depth: i16,
    mate: bool,
    mover_is_white: bool,
}

/// Run the engine on `pos` with the configured node budget; returns (best_move, stm_score, depth).
fn search_pos(pos: &mut Position, info: &mut SearchInfo, hash_history: &[u64], soft_nodes: u64) -> (Move, i32, i32) {
    info.hash_history = hash_history.to_vec();
    info.game_history_len = info.hash_history.len();
    info.clear_for_search();
    crate::time::set_stop(false);
    // infinite=true bypasses the wtime=0 → 100ms-cap default so soft_nodes actually bounds the search.
    let tc = TimeControl {
        wtime: 0, btime: 0, winc: 0, binc: 0,
        movestogo: 0, movetime: 0, depth: 0,
        infinite: true, nodes: HARD_NODE_LIMIT, soft_nodes,
    };
    let mut tm = TimeManager::new(&tc, pos.side == WHITE, 0);
    let best = search::search(pos, info, &mut tm);
    (best, info.root_score, info.completed_depth)
}

/// Play one clrsrc self-play game; return the recorded opening entries (WDL already filled).
fn play_game(
    info_white: &mut SearchInfo,
    info_black: &mut SearchInfo,
    rng: &mut Rng,
    book: Option<&ExpBook>,
    soft_nodes: u64,
) -> Vec<ExpEntry> {
    let mut pos = Position::startpos();
    let mut hash_history: Vec<u64> = vec![pos.hash];

    let mut pending: Vec<PendingEntry> = Vec::new();
    let mut adj_white = 0i32;
    let mut adj_black = 0i32;
    let mut draw_adj = 0i32;
    let mut game_result: u8 = 1; // 2 = white win, 1 = draw, 0 = black win
    let mut ply: u32 = 0;

    loop {
        if ply >= MAX_GAME_MOVES { break; }

        let rep = hash_history.iter().filter(|&&h| h == pos.hash).count();
        if rep >= 3 { break; }
        if pos.halfmove_clock >= 100 { break; }

        let legal = get_legal_moves(&mut pos);
        if legal.is_empty() {
            if pos.is_in_check() {
                game_result = if pos.side == WHITE { 0 } else { 2 };
            }
            break;
        }

        let mover_is_white = pos.side == WHITE;
        let key = book::polyglot_hash(&pos);
        let info = if mover_is_white { &mut *info_white } else { &mut *info_black };

        // Within the recording window, first try the diversity book (v4) with variety>0 so
        // self-play spreads across v4's main-line distribution — the same distribution the A/B
        // test reaches when started from startpos with BookVariety>0. If the book has no entry,
        // full-search instead and record clrsrc's own pick.
        let in_window = ply < RECORD_PLIES;
        let book_move = if in_window {
            book.and_then(|b| b.probe_best(&mut pos, DIVERSITY_VARIETY, rng.next_u64()))
        } else { None };

        let (played_move, played_score, played_depth) = match book_move {
            Some(bm) => {
                // Mini-search to capture clrsrc's own evaluation of the pre-move position. v4 picked
                // the move (diversity); the score we record reflects clrsrc's stm view, not v4's.
                let (_m, mini_score, mini_depth) = search_pos(&mut pos, info, &hash_history, MINI_SEARCH_NODES);
                (bm, mini_score, mini_depth)
            }
            None => {
                let (best_move, score, depth) = search_pos(&mut pos, info, &hash_history, soft_nodes);
                if best_move == Move::NULL { break; }
                (best_move, score, depth)
            }
        };

        if in_window {
            pending.push(PendingEntry {
                key,
                packed_move: book::encode_poly_move(played_move),
                clrsrc_score: played_score.clamp(-32000, 32000) as i16,
                depth: played_depth.clamp(0, i16::MAX as i32) as i16,
                mate: played_score.abs() >= MINIMUM_MATE_SCORE,
                mover_is_white,
            });
        }

        // Bind shadowed locals so the adjudication block below reads naturally.
        let best_move = played_move;
        let score = played_score;

        // Win adjudication.
        if score.abs() >= WIN_ADJ_SCORE {
            if (mover_is_white && score > 0) || (!mover_is_white && score < 0) {
                adj_white += 1; adj_black = 0;
            } else {
                adj_black += 1; adj_white = 0;
            }
        } else {
            adj_white = 0; adj_black = 0;
        }
        if adj_white >= WIN_ADJ_COUNT { game_result = 2; break; }
        if adj_black >= WIN_ADJ_COUNT { game_result = 0; break; }

        // Draw adjudication (late game only).
        if ply >= DRAW_ADJ_MIN_PLY && score.abs() <= DRAW_ADJ_SCORE { draw_adj += 1; } else { draw_adj = 0; }
        if draw_adj >= DRAW_ADJ_COUNT { game_result = 1; break; }

        pos.make_move(best_move);
        hash_history.push(pos.hash);
        ply += 1;

        // Once past the recording window, stop early if there is nothing left to record and the
        // game is long — but we still need the result, so keep playing to termination.
    }

    // Fill WDL from the final result, per the mover's perspective.
    let mut out = Vec::with_capacity(pending.len());
    for p in pending {
        let (w, l) = match game_result {
            2 => if p.mover_is_white { (1u16, 0u16) } else { (0, 1) }, // white won
            0 => if p.mover_is_white { (0, 1) } else { (1, 0) },       // black won
            _ => (0, 0),                                                // draw
        };
        let mut flags = EXP_FLAG_VALIDATED;
        if p.mate { flags |= EXP_FLAG_MATE; }
        out.push(ExpEntry {
            key: p.key,
            packed_move: p.packed_move,
            score: p.clrsrc_score, // canonical mirror = clrsrc_score
            depth: p.depth,
            count: 1,
            source: EXP_SOURCE_SELFPLAY | EXP_SOURCE_ENGINE, // 0x28
            flags,
            wdl_w: w,
            wdl_l: l,
            nnue_eval: EXP_SCORE_NONE,
            jug_score: EXP_SCORE_NONE,
            sf_score: EXP_SCORE_NONE,
            clrsrc_score: p.clrsrc_score,
        });
    }
    out
}

// ---- Seeded mode (targeted line, e.g. anti-Kan White) ----
// genbook is deterministic out of book, so from a fixed seed position every game would collapse
// to one line. Seeded mode injects diversity: at the seed it forces one of a candidate-move list
// (clean per-move WDL attribution), then plays a few HCE-near-best random plies for spread, then
// full-searches to the result. Only the seed move is recorded → the overlay carries one entry per
// candidate at the seed key, count/WDL accumulated. Fold into v4 with expmerge, A/B with the
// clrsrc binary; enough decisive games let book.rs:716 (sibling-relative) demote losing siblings.
#[derive(Clone)]
pub struct SeedConfig {
    pub fen: String,
    pub cand_moves_uci: Vec<String>, // forced candidates at the seed; empty => clrsrc full-search best
    pub explore_plies: u32,          // random HCE-near-best plies after the seed move (diversity)
}
/// Explore-ply moves are picked uniformly among legal moves within this HCE margin of the best —
/// cheap anti-blunder filter (full NNUE per move would be costly; HCE catches hanging material).
const EXPLORE_HCE_MARGIN: i32 = 120;
/// Reject a seeded game whose position after the explore phase is already lopsided (a random ply
/// blundered) — keeps the candidate's WDL drawn from balanced continuations (mirrors datagen).
const SEED_VERIFY_LIMIT: i32 = 1000;

/// Pick a uniformly-random move among those within `margin` cp (HCE, side-to-move view) of the best.
fn pick_reasonable(pos: &mut Position, legal: &[Move], rng: &mut Rng, margin: i32) -> Move {
    let mut scored: Vec<(Move, i32)> = Vec::with_capacity(legal.len());
    for &mv in legal {
        let undo = pos.make_move(mv);
        // eval::evaluate returns the new side-to-move's view; negate for the mover we just played.
        let v = -crate::eval::evaluate(pos);
        pos.unmake_move(mv, undo);
        scored.push((mv, v));
    }
    let best = scored.iter().map(|x| x.1).max().unwrap();
    let pool: Vec<Move> = scored.iter().filter(|x| x.1 >= best - margin).map(|x| x.0).collect();
    pool[(rng.next_u64() % pool.len() as u64) as usize]
}

/// Play one seeded self-play game; record only the seed move (WDL filled from the final result).
fn play_game_seeded(
    info_white: &mut SearchInfo,
    info_black: &mut SearchInfo,
    rng: &mut Rng,
    seed: &SeedConfig,
    cand_moves: &[Move],
    soft_nodes: u64,
) -> Vec<ExpEntry> {
    let mut pos = Position::from_fen(&seed.fen).expect("valid seed FEN");
    let mut hash_history: Vec<u64> = vec![pos.hash];
    let seed_key = book::polyglot_hash(&pos);
    let mover_is_white = pos.side == WHITE;

    // --- ply 0: the measured move at the seed position ---
    let chosen = if cand_moves.is_empty() {
        let info0 = if mover_is_white { &mut *info_white } else { &mut *info_black };
        let (best, _s, _d) = search_pos(&mut pos, info0, &hash_history, soft_nodes);
        if best == Move::NULL { return Vec::new(); }
        best
    } else {
        cand_moves[(rng.next_u64() % cand_moves.len() as u64) as usize]
    };
    // clrsrc's eval of the move = mover-perspective eval of the child (mini-search, negated).
    pos.make_move(chosen);
    hash_history.push(pos.hash);
    let (mini_score, mini_depth) = {
        let info_c = if pos.side == WHITE { &mut *info_white } else { &mut *info_black };
        let (_m, child_stm, child_depth) = search_pos(&mut pos, info_c, &hash_history, MINI_SEARCH_NODES);
        (-child_stm, child_depth)
    };
    let pending = PendingEntry {
        key: seed_key,
        packed_move: book::encode_poly_move(chosen),
        clrsrc_score: mini_score.clamp(-32000, 32000) as i16,
        depth: mini_depth.clamp(0, i16::MAX as i32) as i16,
        mate: mini_score.abs() >= MINIMUM_MATE_SCORE,
        mover_is_white,
    };

    let mut game_result: u8 = 1; // 2=white win, 1=draw, 0=black win
    let mut ply: u32 = 1;

    // --- explore phase: HCE-near-best random plies for diversity (not recorded) ---
    let mut explored = 0u32;
    while explored < seed.explore_plies {
        let rep = hash_history.iter().filter(|&&h| h == pos.hash).count();
        if rep >= 3 || pos.halfmove_clock >= 100 { break; }
        let legal = get_legal_moves(&mut pos);
        if legal.is_empty() {
            if pos.is_in_check() { game_result = if pos.side == WHITE { 0 } else { 2 }; }
            return fill_seeded(pending, game_result);
        }
        let mv = pick_reasonable(&mut pos, &legal, rng, EXPLORE_HCE_MARGIN);
        pos.make_move(mv);
        hash_history.push(pos.hash);
        ply += 1;
        explored += 1;
    }

    // --- sanity gate: reject lopsided post-explore positions (random-ply blunder) ---
    {
        let info_g = if pos.side == WHITE { &mut *info_white } else { &mut *info_black };
        let (_m, stm_score, _d) = search_pos(&mut pos, info_g, &hash_history, MINI_SEARCH_NODES);
        if stm_score.abs() > SEED_VERIFY_LIMIT { return Vec::new(); } // drop, don't record
    }

    // --- play-out phase: full-search to the result with adjudication (not recorded) ---
    let mut adj_white = 0i32;
    let mut adj_black = 0i32;
    let mut draw_adj = 0i32;
    loop {
        if ply >= MAX_GAME_MOVES { break; }
        let rep = hash_history.iter().filter(|&&h| h == pos.hash).count();
        if rep >= 3 || pos.halfmove_clock >= 100 { break; }
        let legal = get_legal_moves(&mut pos);
        if legal.is_empty() {
            if pos.is_in_check() { game_result = if pos.side == WHITE { 0 } else { 2 }; }
            break;
        }
        let stm_white = pos.side == WHITE;
        let info = if stm_white { &mut *info_white } else { &mut *info_black };
        let (best_move, score, _depth) = search_pos(&mut pos, info, &hash_history, soft_nodes);
        if best_move == Move::NULL { break; }

        if score.abs() >= WIN_ADJ_SCORE {
            if (stm_white && score > 0) || (!stm_white && score < 0) { adj_white += 1; adj_black = 0; }
            else { adj_black += 1; adj_white = 0; }
        } else { adj_white = 0; adj_black = 0; }
        if adj_white >= WIN_ADJ_COUNT { game_result = 2; break; }
        if adj_black >= WIN_ADJ_COUNT { game_result = 0; break; }
        if ply >= DRAW_ADJ_MIN_PLY && score.abs() <= DRAW_ADJ_SCORE { draw_adj += 1; } else { draw_adj = 0; }
        if draw_adj >= DRAW_ADJ_COUNT { game_result = 1; break; }

        pos.make_move(best_move);
        hash_history.push(pos.hash);
        ply += 1;
    }
    fill_seeded(pending, game_result)
}

/// Fill WDL on the single seeded pending entry from the final result and emit the overlay entry.
fn fill_seeded(p: PendingEntry, game_result: u8) -> Vec<ExpEntry> {
    let (w, l) = match game_result {
        2 => if p.mover_is_white { (1u16, 0u16) } else { (0, 1) },
        0 => if p.mover_is_white { (0, 1) } else { (1, 0) },
        _ => (0, 0),
    };
    let mut flags = EXP_FLAG_VALIDATED;
    if p.mate { flags |= EXP_FLAG_MATE; }
    vec![ExpEntry {
        key: p.key,
        packed_move: p.packed_move,
        score: p.clrsrc_score,
        depth: p.depth,
        count: 1,
        source: EXP_SOURCE_SELFPLAY | EXP_SOURCE_ENGINE, // 0x28
        flags,
        wdl_w: w,
        wdl_l: l,
        nnue_eval: EXP_SCORE_NONE,
        jug_score: EXP_SCORE_NONE,
        sf_score: EXP_SCORE_NONE,
        clrsrc_score: p.clrsrc_score,
    }]
}

/// Load the NNUE the same way datagen does (cwd, then next to / above the exe).
fn load_nnue(info_white: &mut SearchInfo, info_black: &mut SearchInfo, verbose: bool) {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.to_path_buf()));
    for name in &["clrsrc_v32_seed_b.nnue", "clrsrc_v16_kb4.nnue", "clrsrc.nnue"] {
        let mut candidates = vec![std::path::PathBuf::from(name)];
        if let Some(ref dir) = exe_dir {
            candidates.push(dir.join(name));
            if let Some(proj) = dir.parent().and_then(|d| d.parent()) {
                candidates.push(proj.join(name));
            }
        }
        for path in &candidates {
            if path.exists() {
                let ps = path.to_string_lossy().to_string();
                if verbose { eprintln!("genbook: loading NNUE from {}", ps); }
                let _ = info_white.nnue.load(&ps);
                let _ = info_black.nnue.load(&ps);
                return;
            }
        }
    }
    if verbose { eprintln!("genbook: WARNING — no NNUE file found, using HCE eval"); }
}

/// Run multi-threaded clrsrc self-play book generation.
pub fn run_genbook(num_games: u32, soft_nodes: u64, output_path: &str, num_threads: u32, book_path: Option<&str>, seed: Option<SeedConfig>) {
    let threads = num_threads.max(1);
    let book: Option<Arc<ExpBook>> = book_path.and_then(|p| ExpBook::load(p).map(Arc::new));
    if book_path.is_some() && book.is_none() {
        eprintln!("genbook: ERROR — could not load diversity book {:?} (expected JBK2/v4 format)", book_path);
        return;
    }

    // Parse the seed candidate moves once (need a Position at the seed to decode UCI).
    let seed_parsed: Option<(SeedConfig, Vec<Move>)> = match seed {
        Some(sc) => {
            let probe = match Position::from_fen(&sc.fen) {
                Ok(p) => p,
                Err(e) => { eprintln!("genbook: ERROR — invalid --seed-fen {:?}: {}", sc.fen, e); return; }
            };
            let mut cmoves = Vec::new();
            for u in &sc.cand_moves_uci {
                match probe.parse_uci_move(u) {
                    Some(mv) => cmoves.push(mv),
                    None => { eprintln!("genbook: ERROR — illegal/unparsable candidate move {:?} at seed", u); return; }
                }
            }
            Some((sc, cmoves))
        }
        None => None,
    };

    let writer = Arc::new(Mutex::new(OverlayWriter::open(output_path)));
    let games_done = Arc::new(AtomicU32::new(0));
    let entries_total = Arc::new(AtomicU64::new(0));
    let start = std::time::Instant::now();

    match &seed_parsed {
        Some((sc, _cmoves)) => eprintln!(
            "genbook SEEDED: {} games, soft {} nodes, {} threads, seed={}, candidates={:?} (empty=full-search), explore_plies={}, src=0x28, output: {}",
            num_games, soft_nodes, threads, sc.fen, sc.cand_moves_uci, sc.explore_plies, output_path
        ),
        None => eprintln!(
            "genbook: {} games, soft {} nodes, {} threads, record plies [0,{}), src=0x28 (selfplay|clrsrc), {}, output: {}",
            num_games, soft_nodes, threads, RECORD_PLIES,
            if book.is_some() { "v4 diversity book (variety=2)" } else { "no book — deterministic engine play" },
            output_path
        ),
    }

    let mut handles = Vec::new();
    for thread_id in 0..threads {
        let writer = Arc::clone(&writer);
        let games_done = Arc::clone(&games_done);
        let entries_total = Arc::clone(&entries_total);
        let book = book.clone();
        let seed_cfg = seed_parsed.clone();
        let games_per_thread = num_games / threads;
        let extra = if thread_id < num_games % threads { 1 } else { 0 };
        let my_games = games_per_thread + extra;

        let handle = std::thread::spawn(move || {
            let tt_white = crate::tt::TTable::new_shared(4);
            let tt_black = crate::tt::TTable::new_shared(4);
            let mut info_white = SearchInfo::new(tt_white);
            let mut info_black = SearchInfo::new(tt_black);
            info_white.silent = true;
            info_black.silent = true;
            load_nnue(&mut info_white, &mut info_black, thread_id == 0);

            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64
                ^ ((thread_id as u64 + 1) * 0x517cc1b727220a95);
            let mut rng = Rng::new(seed);

            let mut local: Vec<ExpEntry> = Vec::new();
            for local_game in 0..my_games {
                if local_game % 100 == 0 {
                    info_white.tt.clear();
                    info_black.tt.clear();
                }
                for info in [&mut info_white, &mut info_black] {
                    info.history.clear();
                    info.killers.clear();
                    info.counter_moves.clear();
                    info.cont_history.clear();
                    info.capture_history.clear();
                }

                let entries = match &seed_cfg {
                    Some((sc, cmoves)) => play_game_seeded(&mut info_white, &mut info_black, &mut rng, sc, cmoves, soft_nodes),
                    None => play_game(&mut info_white, &mut info_black, &mut rng, book.as_deref(), soft_nodes),
                };
                local.extend(entries);

                if (local_game + 1) % 10 == 0 || local_game + 1 == my_games {
                    let flushed = local.len();
                    {
                        let mut w = writer.lock().unwrap();
                        for e in local.drain(..) { w.push(e); }
                        w.flush();
                    }
                    entries_total.fetch_add(flushed as u64, Ordering::Relaxed);
                    let batch = if (local_game + 1) % 10 == 0 { 10 } else { (local_game + 1) % 10 };
                    let done = games_done.fetch_add(batch, Ordering::Relaxed) + batch;
                    if thread_id == 0 {
                        let elapsed = start.elapsed().as_secs_f64().max(1e-9);
                        let et = entries_total.load(Ordering::Relaxed);
                        eprintln!(
                            "genbook ~{}/{}: {} entries ({:.1} games/s, {} threads)",
                            done, num_games, et, done as f64 / elapsed, threads
                        );
                    }
                }
            }
        });
        handles.push(handle);
    }

    for h in handles { h.join().expect("thread panicked"); }

    let et = entries_total.load(Ordering::Relaxed);
    let size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "genbook complete: {} entries from {} games in {:.1}s ({:.2} MB) -> {}",
        et, num_games, start.elapsed().as_secs_f64(), size as f64 / (1024.0 * 1024.0), output_path
    );
    eprintln!("genbook: fold in with  clrsrc expmerge <base.book> {} <out.book> --clrsrc-mirror", output_path);
}

// ---- Mainline FEN extraction for A/B test openings ----
// Walks the book by count-weighted random sampling for `target_ply` plies starting from startpos
// (no engine search — pure book traversal), emits the FEN after target_ply moves, dedups across
// games. Used to build a Mainline-aware opening set for ab_book_sprt.sh so the A/B starts in
// positions the self-play overlay actually reached.

/// Sample one count-weighted random move from a book position. Floors entries below `min_count`
/// so a single freak side-line doesn't bias the sample. Returns (move, count, kept_pool_size).
fn sample_book_move(book: &ExpBook, pos: &mut Position, rng: &mut Rng, min_count: u16) -> Option<Move> {
    let entries = book.probe(pos);
    if entries.is_empty() { return None; }
    let kept: Vec<&book::ExpEntry> = entries.iter().filter(|e| e.count >= min_count).collect();
    let pool = if kept.is_empty() { entries.iter().collect::<Vec<_>>() } else { kept };
    let total: u64 = pool.iter().map(|e| e.count as u64).sum();
    if total == 0 { return None; }
    let pick = rng.next_u64() % total;
    let mut acc = 0u64;
    for e in &pool {
        acc += e.count as u64;
        if pick < acc {
            return book::decode_poly_move(pos, e.packed_move);
        }
    }
    None
}

/// Walk a book and emit unique FENs after `target_ply` count-weighted-random moves.
pub fn run_bookfens(num_games: u32, target_ply: u32, output_path: &str, book_path: &str, min_count: u16) {
    let book = match ExpBook::load(book_path) {
        Some(b) => b,
        None => { eprintln!("bookfens: ERROR — could not load book {}", book_path); return; }
    };
    eprintln!("bookfens: {} samples, target_ply={}, min_count={}, book={} → {}",
        num_games, target_ply, min_count, book_path, output_path);

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() as u64;
    let mut rng = Rng::new(seed);

    let mut unique: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut tries = 0u32;
    let mut short_walks = 0u32;
    for _ in 0..num_games {
        tries += 1;
        let mut pos = Position::startpos();
        let mut ply = 0;
        let mut broke_early = false;
        while ply < target_ply {
            match sample_book_move(&book, &mut pos, &mut rng, min_count) {
                Some(mv) => { pos.make_move(mv); ply += 1; }
                None => { broke_early = true; break; }
            }
        }
        if broke_early { short_walks += 1; continue; }
        unique.insert(pos.to_fen());
    }

    let out = unique.iter().cloned().collect::<Vec<_>>().join("\n") + "\n";
    if let Err(e) = std::fs::write(output_path, out) {
        eprintln!("bookfens: write failed: {}", e);
        return;
    }
    eprintln!("bookfens: {} unique FENs from {} samples ({} short walks, broke out of book before ply {})",
        unique.len(), tries, short_walks, target_ply);
}
