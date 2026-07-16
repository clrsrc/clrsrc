//! clrsrc als Bibliothek — additive Fassade für die In-Process-Einbettung
//! in den Rust-lichess-bot (siehe BOT_ENGINE_INTEGRATION_PLAN.md, Schritt C1).
//!
//! WICHTIG: Diese lib ist rein additiv. Das UCI-Binary (`src/main.rs`) behält
//! seine eigenen `mod`-Deklarationen und `crate::`-Pfade unverändert — daran
//! hängt das gesamte SPRT-/fastchess-Test-Ökosystem (Bench-Gate 2.846.610).
//!
//! Exportiert wird nur der Dependency-Closure, den die Suche braucht:
//! `search` zieht `tune`/`tablebase` transitiv mit. Bot-spezifische bzw.
//! standalone-only Module (`datagen`, `selfplaybook`, `rescore`, `uci`) sind
//! bewusst NICHT Teil der Bibliothek.
//!
//! Die `search_embedded()`-Fassade + `EmbeddedLimits`/`SearchOutcome` (C3) und
//! der `TimeManager::with_deadline`-Konstruktor (C2) kommen in Folgeschritten
//! hinzu.

pub mod types;
pub mod bitboard;
pub mod zobrist;
pub mod magic;
pub mod attacks;
pub mod board;
pub mod movegen;
pub mod perft;
pub mod tt;
pub mod eval;
pub mod ordering;
pub mod nnue;
pub mod time;
pub mod tablebase;
pub mod tune;
pub mod search;
pub mod book;

// ============================================================================
// C3 — Embedded search facade (BOT_ENGINE_INTEGRATION_PLAN.md)
// Thin wrapper over `search::search` + `time::TimeManager::with_deadline`.
// The UCI binary does not use any of this — it is the entry the lichess-bot
// links against as a path-dependency.
// ============================================================================

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Once;
use std::time::Instant;

use crate::board::Position;
use crate::book::{Book, ExpBook};
use crate::search::SearchInfo;
use crate::time::{TimeControl, TimeManager};
use crate::types::{Move, WHITE};

/// Initialise the global lookup tables exactly once. The UCI binary does this in
/// `main()`; embedded callers get it via `EmbeddedEngine::init`. Idempotent, so it
/// is safe to construct multiple engines (e.g. one per game) in the same process.
static INIT: Once = Once::new();
pub fn init_tables() {
    INIT.call_once(|| {
        crate::zobrist::init();
        crate::magic::init();
        crate::attacks::init();
        crate::search::init_lmr();
    });
}

/// What the bot hands the engine for one move. clrsrc derives its form-aware
/// soft/hard budget from `tc`, but `max_deadline` (bot-computed from the real
/// Lichess clock − RTT − buffer) is the authoritative hard ceiling.
pub struct EmbeddedLimits {
    /// wtime/btime/winc/binc — clrsrc computes soft/hard from this.
    pub tc: TimeControl,
    /// Absolute wall-clock ceiling. min(clrsrc_computed_hard, this − start) wins.
    pub max_deadline: Instant,
    /// Pondering on the opponent's clock (no time-based stop while true).
    pub ponder: bool,
    /// Plies played from game start — feeds the TC scaling formula (TimeManager).
    /// The bot tracks game state, so it passes this explicitly (contract refinement
    /// vs. the original plan struct; without it the TM phase scaling is wrong).
    pub game_ply: u32,
    /// Optional fixed-depth override (0 = unlimited).
    pub depth: i32,
    /// Optional fixed-node override (0 = unlimited).
    pub nodes: u64,
}

/// Everything the bot needs back: best move + PV/score/depth/nodes for the
/// !pv chat, commentary and experience harvest (Bot request 019).
pub struct SearchOutcome {
    pub best: Move,
    pub ponder: Option<Move>,
    pub pv: Vec<Move>,
    pub score_cp: i32,
    pub depth: i32,
    pub nodes: u64,
    /// Mate-in-N (signed; negative = being mated). Seit dem Band-Rework (Fix 1,
    /// chess_engines #56) nur noch für ECHTE Matt-Scores (>= MINIMUM_MATE_SCORE) —
    /// TB-Wins liegen im eigenen Band und melden None (Phantom-`mate` behoben).
    pub mate: Option<i32>,
}

/// Run one embedded search. `info` is PERSISTENT across moves (the bot holds it
/// per game so TT/History/NNUE stay warm). `cancel` is the external abort.
///
/// Cancellation: for VServer `concurrency:1` the process-global `time::STOP`
/// suffices, so we bridge `cancel` → global STOP here (a pre-set cancel aborts
/// immediately, since `search()` does not clear global STOP). Live per-search
/// interruption under multi-game in-process is C4 (deferred, documented risk).
pub fn search_embedded(
    pos: &mut Position,
    info: &mut SearchInfo,
    limits: EmbeddedLimits,
    cancel: &AtomicBool,
) -> SearchOutcome {
    let mut limits = limits;
    // Fold optional depth/nodes overrides into the time control.
    if limits.depth > 0 { limits.tc.depth = limits.depth; }
    if limits.nodes > 0 { limits.tc.nodes = limits.nodes; }

    crate::time::set_stop(cancel.load(Ordering::Relaxed));
    crate::time::set_pondering(limits.ponder);

    let is_white = pos.side == WHITE;
    let mut tm = TimeManager::with_deadline(&limits.tc, is_white, limits.game_ply, limits.max_deadline);

    let best = crate::search::search(pos, info, &mut tm);

    // Pull the rest of the outcome from SearchInfo (populated by search()).
    // `best` is the move of the last fully-completed depth (search.rs:461, set
    // only after the abort-break). A time-aborted final iteration can leave
    // `info.pv[0]` holding the PARTIAL line of the aborted depth, whose pv[0][0]
    // may differ from `best` (e.g. a re-tried root move that transiently raised
    // alpha before the abort). update_pv keeps pv[0] internally consistent
    // (pv[0][0] + its own continuation), so the line is trustworthy IFF it still
    // starts with the returned best; otherwise it belongs to the discarded
    // aborted depth → fall back to a best-only PV so the reported/harvested PV
    // always matches the played move (Shg8ehfe: best=Bf2 vs pv[0]=Nb3). UCI is
    // unaffected — it prints the PV inside the loop, per completed depth.
    let pv_len = info.pv_len[0];
    let pv: Vec<Move> = if pv_len > 0 && info.pv[0][0] == best {
        info.pv[0][..pv_len].to_vec()
    } else {
        vec![best]
    };
    let ponder = if pv.len() >= 2 { Some(pv[1]) } else { None };

    let score = info.root_score;
    let mate = if score.abs() >= crate::search::MINIMUM_MATE_SCORE {
        Some(if score > 0 {
            (crate::search::MATE_SCORE - score + 1) / 2
        } else {
            -(crate::search::MATE_SCORE + score + 1) / 2
        })
    } else {
        None
    };

    // Single-threaded embedded search: `info.nodes` already counts every node
    // (search.rs increments it unconditionally). Do NOT add `helper_nodes` here:
    // when `info.silent` is set, the search flushes its OWN node delta into the
    // shared helper_nodes counter (search.rs:441, the SMP helper path), so adding
    // it would double-count. This mirrors run_bench, which reports info.nodes.
    // For a future SMP-embedded mode (C4), sum the per-helper SearchInfos instead.
    SearchOutcome {
        best,
        ponder,
        pv,
        score_cp: score,
        depth: info.completed_depth,
        nodes: info.nodes,
        mate,
    }
}

// ============================================================================
// Option A — stateful EmbeddedEngine with book + one-time setup (Postfach 041/042)
//
// Closes the book gap: the bare `search_embedded` above bypasses the opening
// book that the UCI `go`-handler probes before searching (uci.rs:152-174), so a
// bookless embedded engine would play different openings → not ELO-neutral.
// `EmbeddedEngine` owns the book/NNUE/TT/Syzygy state (loaded once in `init`,
// exactly the `cmd_setoption` paths) and probes the book before search, so book
// move and search move return over the same path. The bot holds one engine per
// game; the per-move `SearchInfo` (TT/History/NNUE) stays warm across moves.
// ============================================================================

/// One-time engine configuration. Mirrors the live UCI options. All optional
/// bits default to off/empty so a bare `EmbeddedConfig::default()` is a valid
/// classical-eval, bookless engine; the bot fills in what its config.yml sets.
pub struct EmbeddedConfig {
    pub eval_file: Option<String>,        // NNUE path; None/empty = classical eval
    pub hash_mb: usize,                   // TT size in MB
    pub syzygy_path: Option<String>,      // Syzygy TB dir
    pub syzygy_probe_limit: Option<usize>,// max pieces to probe (live mitigation = 5)
    pub syzygy_probe_depth: Option<usize>,
    pub syzygy_50move: Option<bool>,
    pub exp_file: Option<String>,         // JBK2 experience/opening book
    pub play_from_exp: bool,              // probe the JBK2 book before search
    pub book_variety: u8,                 // 0 = deterministic best, 1/2 = wider pool
    pub own_book: bool,                   // probe the Polyglot fallback book
    pub book_file: Option<String>,        // Polyglot book path
    pub best_book_move: bool,             // Polyglot: pick best vs weighted-random
    // TM-Trigger 2 (Konvertierungs-Floor; braucht nur ein geladenes NNUE, kein Sidecar).
    // Spiegelt die UCI-Optionen ConvFloor/-Threshold/-Extend/-MaxPieces/-MinRemaining.
    pub conv_floor: bool,                 // default true (public v1.3.0, SPRT-validiert)
    pub conv_threshold: Option<u32>,      // Default 200 (cp, Root-Eval stm)
    pub conv_extend: Option<u32>,         // Default 50 (% auf soft; 0 = nur Floor)
    pub conv_maxpieces: Option<u32>,      // Default 16 (popcount-Endspiel-Gate)
    pub conv_minremaining: Option<u32>,   // Default 5000 (ms Uhr-Gesundheits-Gate)
}

impl Default for EmbeddedConfig {
    fn default() -> Self {
        EmbeddedConfig {
            eval_file: None,
            hash_mb: 64,
            syzygy_path: None,
            syzygy_probe_limit: None,
            syzygy_probe_depth: None,
            syzygy_50move: None,
            exp_file: None,
            play_from_exp: false,
            book_variety: 0,
            own_book: false,
            book_file: None,
            best_book_move: true,
            conv_floor: true,
            conv_threshold: None,
            conv_extend: None,
            conv_maxpieces: None,
            conv_minremaining: None,
        }
    }
}

/// A persistent embedded engine: holds the warm `SearchInfo` (TT/History/NNUE)
/// and the opening books. One per game on the bot side.
pub struct EmbeddedEngine {
    info: SearchInfo,
    exp: Option<ExpBook>,
    book: Option<Book>,
    play_from_exp: bool,
    own_book: bool,
    best_book_move: bool,
    book_variety: u8,
    exp_rng: u64,
}

impl EmbeddedEngine {
    /// Load NNUE / TT / Syzygy / books once. Global lookup tables are initialised
    /// here too (idempotent), so the bot need not call anything else first.
    pub fn init(config: EmbeddedConfig) -> EmbeddedEngine {
        init_tables();

        let tt = crate::tt::TTable::new_shared(config.hash_mb.max(1).min(65536));
        let mut info = SearchInfo::new(tt);
        // Embedded: behaviour identical to UCI (silent=false → root TB probe active,
        // main-thread node accounting), but no stdout spam (print_info=false).
        info.silent = false;
        info.print_info = false;

        if let Some(ref p) = config.eval_file {
            if !p.is_empty() {
                if let Err(e) = info.nnue.load(p) {
                    eprintln!("info string embedded: NNUE load error: {}", e);
                }
            }
        }

        if let Some(ref sp) = config.syzygy_path {
            if !sp.is_empty() {
                crate::tablebase::init(sp);
            }
        }
        if let Some(l) = config.syzygy_probe_limit { crate::tablebase::set_probe_limit(l.min(7)); }
        if let Some(d) = config.syzygy_probe_depth { crate::tablebase::set_probe_depth(d.max(1).min(100)); }
        if let Some(b) = config.syzygy_50move { crate::tablebase::set_50move_rule(b); }

        // TM-Trigger 2 (Konvertierungs-Floor): exakt die UCI-setoption-Pfade.
        crate::time::set_conv_enabled(config.conv_floor);
        if let Some(v) = config.conv_threshold { crate::time::set_conv_threshold(v); }
        if let Some(v) = config.conv_extend { crate::time::set_conv_extend(v); }
        if let Some(v) = config.conv_maxpieces { crate::time::set_conv_maxpieces(v); }
        if let Some(v) = config.conv_minremaining { crate::time::set_conv_minremaining(v); }

        let exp = if config.play_from_exp {
            config.exp_file.as_ref().and_then(|p| if p.is_empty() { None } else { ExpBook::load(p) })
        } else { None };
        let book = if config.own_book {
            config.book_file.as_ref().and_then(|p| if p.is_empty() { None } else { Book::load(p) })
        } else { None };

        let exp_rng = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0x1234_5678_9abc_def0)
            | 1;

        EmbeddedEngine {
            info, exp, book,
            play_from_exp: config.play_from_exp,
            own_book: config.own_book,
            best_book_move: config.best_book_move,
            book_variety: config.book_variety,
            exp_rng,
        }
    }

    /// Whether an NNUE net is loaded. The bot can fail-fast on a wrong `eval_file`
    /// path instead of silently playing classical eval (~-200 ELO, invisible).
    pub fn nnue_loaded(&self) -> bool {
        self.info.nnue.is_loaded()
    }

    /// Search one position given as start-FEN ("startpos" accepted) + the UCI move
    /// list played from it. The engine rebuilds `pos` + repetition history internally
    /// (the bot never touches `SearchInfo`), probes the opening book before search
    /// (same precedence/gate as the UCI `go`-handler), and otherwise runs the timed
    /// search. On a book hit returns `{best=book move, depth=0, nodes=0}` so the bot
    /// does not have to distinguish book from search.
    pub fn search_position(
        &mut self,
        start_fen: &str,
        moves: &[&str],
        limits: EmbeddedLimits,
        cancel: &AtomicBool,
    ) -> SearchOutcome {
        // Build the position and the repetition history (mirrors uci.rs cmd_position).
        let mut pos = if start_fen == "startpos" || start_fen.is_empty() {
            Position::startpos()
        } else {
            match Position::from_fen(start_fen) {
                Ok(p) => p,
                Err(e) => { eprintln!("info string embedded: invalid FEN ({}); using startpos", e); Position::startpos() }
            }
        };
        self.info.hash_history.clear();
        self.info.hash_history.push(pos.hash);
        for mv_str in moves {
            match pos.parse_uci_move(mv_str) {
                Some(mv) => { pos.make_move(mv); self.info.hash_history.push(pos.hash); }
                None => { eprintln!("info string embedded: invalid move {}; stopping replay", mv_str); break; }
            }
        }
        self.info.game_history_len = self.info.hash_history.len();

        // Fold optional depth/nodes overrides into the time control (same as the primitive).
        let limits = {
            let mut l = limits;
            if l.depth > 0 { l.tc.depth = l.depth; }
            if l.nodes > 0 { l.tc.nodes = l.nodes; }
            l
        };

        // Opening-book probe before search — same gate and precedence as uci.rs:152-174
        // (not while pondering, finite, no fixed depth). JBK2 experience book first
        // (WDL-filtered), then Polyglot fallback.
        if !limits.ponder && !limits.tc.infinite && limits.tc.depth == 0 {
            if self.play_from_exp {
                if let Some(ref eb) = self.exp {
                    self.exp_rng = self.exp_rng.wrapping_add(0x9E3779B97F4A7C15);
                    let rng_val = self.exp_rng ^ pos.hash;
                    if let Some(bm) = eb.probe_best(&mut pos, self.book_variety, rng_val) {
                        return SearchOutcome { best: bm, ponder: None, pv: vec![bm], score_cp: 0, depth: 0, nodes: 0, mate: None };
                    }
                }
            }
            if self.own_book {
                if let Some(ref bk) = self.book {
                    let rng_val = pos.hash ^ (limits.game_ply as u64 * 6364136223846793005);
                    if let Some(bm) = bk.probe(&mut pos, rng_val, self.best_book_move) {
                        return SearchOutcome { best: bm, ponder: None, pv: vec![bm], score_cp: 0, depth: 0, nodes: 0, mate: None };
                    }
                }
            }
        }

        // Book miss → timed search over the warm SearchInfo (reuses the primitive).
        search_embedded(&mut pos, &mut self.info, limits, cancel)
    }
}
