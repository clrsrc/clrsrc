// ---- Syzygy Tablebase Probing ----
// Uses shakmaty-syzygy (pure Rust) for endgame tablebase lookups.
// Converts clrsrc Position to shakmaty Chess via FEN for probing.

use crate::board::Position;

use shakmaty::{Chess, CastlingMode};
use shakmaty::fen::Fen;
use shakmaty_syzygy::Tablebase;

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, AtomicBool, Ordering};

static TB: Mutex<Option<Tablebase<Chess>>> = Mutex::new(None);
static TB_MAX_PIECES: AtomicUsize = AtomicUsize::new(0);
static TB_PROBE_LIMIT: AtomicUsize = AtomicUsize::new(6);
static TB_PROBE_DEPTH: AtomicUsize = AtomicUsize::new(1);
static TB_50MOVE_RULE: AtomicBool = AtomicBool::new(true);

/// Initialize tablebases from a path (semicolon-separated directories)
pub fn init(path: &str) {
    let mut tb = Tablebase::new();

    for dir in path.split(';') {
        let dir = dir.trim();
        if dir.is_empty() { continue; }
        let dir_path = std::path::Path::new(dir);
        if dir_path.is_dir() {
            match tb.add_directory(dir_path) {
                Ok(count) => {
                    eprintln!("info string Syzygy: loaded {} files from {}", count, dir);
                }
                Err(e) => {
                    eprintln!("info string Syzygy error for {}: {}", dir, e);
                }
            }
        } else {
            eprintln!("info string Syzygy: directory not found: {}", dir);
        }
    }

    let max = tb.max_pieces();
    eprintln!("info string Syzygy: max {} pieces", max);

    TB_MAX_PIECES.store(max, Ordering::Relaxed);
    // Default probe limit to max available
    if TB_PROBE_LIMIT.load(Ordering::Relaxed) > max {
        TB_PROBE_LIMIT.store(max, Ordering::Relaxed);
    }
    *TB.lock().unwrap() = Some(tb);
}

/// Set probe limit (max pieces to probe)
pub fn set_probe_limit(limit: usize) {
    TB_PROBE_LIMIT.store(limit, Ordering::Relaxed);
}

/// Set probe depth (min search depth to probe)
pub fn set_probe_depth(depth: usize) {
    TB_PROBE_DEPTH.store(depth, Ordering::Relaxed);
}

/// Set whether to respect 50-move rule
pub fn set_50move_rule(val: bool) {
    TB_50MOVE_RULE.store(val, Ordering::Relaxed);
}

pub fn probe_depth() -> usize {
    TB_PROBE_DEPTH.load(Ordering::Relaxed)
}

/// Probe WDL for a position. Returns Some(signum): 1=win, 0=draw, -1=loss.
/// Only probes if piece count <= probe_limit.
pub fn probe_wdl(pos: &Position) -> Option<i32> {
    let max = TB_MAX_PIECES.load(Ordering::Relaxed);
    if max == 0 { return None; }

    let limit = TB_PROBE_LIMIT.load(Ordering::Relaxed);

    // Count pieces (including kings)
    let piece_count = (pos.colors[0] | pos.colors[1]).count_ones() as usize;
    if piece_count > limit { return None; }

    // Convert to shakmaty position via FEN
    let fen_str = pos.to_fen();
    let fen: Fen = match fen_str.parse() {
        Ok(f) => f,
        Err(_) => return None,
    };

    let setup = fen.into_setup();
    let chess: Chess = match setup.position(CastlingMode::Standard) {
        Ok(c) => c,
        Err(_) => return None,
    };

    // Probe WDL
    let guard = TB.lock().unwrap();
    let tb = guard.as_ref()?;

    let use_50move = TB_50MOVE_RULE.load(Ordering::Relaxed);

    if use_50move {
        match tb.probe_wdl_after_zeroing(&chess) {
            Ok(wdl) => Some(wdl.signum()),
            Err(_) => None,
        }
    } else {
        match tb.probe_wdl(&chess) {
            Ok(wdl) => Some(wdl.signum()),
            Err(_) => None,
        }
    }
}

/// Root TB probe: return the DTZ-optimal move (UCI-encoded) plus WDL signum.
/// Used to short-circuit search in TB positions instead of letting alpha-beta
/// pick a winning-but-suboptimal move that risks 50-move-rule conversion failure.
pub fn probe_root(pos: &Position) -> Option<(String, i32)> {
    let max = TB_MAX_PIECES.load(Ordering::Relaxed);
    if max == 0 { return None; }

    let limit = TB_PROBE_LIMIT.load(Ordering::Relaxed);
    let piece_count = (pos.colors[0] | pos.colors[1]).count_ones() as usize;
    if piece_count > limit { return None; }

    let fen_str = pos.to_fen();
    let fen: Fen = fen_str.parse().ok()?;
    let setup = fen.into_setup();
    let chess: Chess = setup.position(CastlingMode::Standard).ok()?;

    let guard = TB.lock().unwrap();
    let tb = guard.as_ref()?;

    let use_50move = TB_50MOVE_RULE.load(Ordering::Relaxed);
    let wdl = if use_50move {
        tb.probe_wdl_after_zeroing(&chess).ok()?.signum()
    } else {
        tb.probe_wdl(&chess).ok()?.signum()
    };

    let (best, _dtz) = tb.best_move(&chess).ok()??;
    let uci = best.to_uci(CastlingMode::Standard).to_string();
    Some((uci, wdl))
}
