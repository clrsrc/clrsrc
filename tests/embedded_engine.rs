//! Option A gates (Postfach 041/042): the stateful `EmbeddedEngine` must
//!  (1) actually consult the opening book before searching — the blind spot the
//!      bot found in `embedded_determinism` (fixed depth never hits the book), and
//!  (2) reproduce the UCI search path bit-for-bit on a book miss.

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use clrsrc::board::Position;
use clrsrc::book::ExpBook;
use clrsrc::search::{self, SearchInfo};
use clrsrc::time::{TimeControl, TimeManager};
use clrsrc::tt;
use clrsrc::types;
use clrsrc::{EmbeddedConfig, EmbeddedEngine, EmbeddedLimits};

const NNUE: &str = "clrsrc_v32_seed_b.nnue";
const BOOK: &str = "clrsrc_selfplay_pilot2_v5.book";

fn far_deadline() -> Instant { Instant::now() + Duration::from_secs(3600) }

/// (1) Book path: the engine must return the SAME move a direct book probe gives,
/// with depth==0 && nodes==0 (i.e. it short-circuited the search via the book).
#[test]
fn embedded_engine_plays_the_book() {
    clrsrc::init_tables();

    // Direct probe as ground truth (variety 0 = deterministic, ignores rng).
    let book = match ExpBook::load(BOOK) {
        Some(b) => b,
        None => { eprintln!("SKIP: {} not loadable as ExpBook", BOOK); return; }
    };
    let mut probe_pos = Position::startpos();
    let direct = book.probe_best(&mut probe_pos, 0, 0);
    let expected = match direct {
        Some(m) => m,
        None => { eprintln!("SKIP: book has no entry for startpos"); return; }
    };

    let mut engine = EmbeddedEngine::init(EmbeddedConfig {
        eval_file: Some(NNUE.to_string()),
        exp_file: Some(BOOK.to_string()),
        play_from_exp: true,
        book_variety: 0,
        ..Default::default()
    });

    let limits = EmbeddedLimits {
        tc: TimeControl { wtime: 60_000, btime: 60_000, winc: 0, binc: 0,
                          movestogo: 0, movetime: 0, depth: 0, infinite: false, nodes: 0, soft_nodes: 0 },
        max_deadline: far_deadline(),
        ponder: false, game_ply: 0, depth: 0, nodes: 0,
    };
    let cancel = AtomicBool::new(false);
    let out = engine.search_position("startpos", &[], limits, &cancel);

    assert_eq!(out.best, expected, "engine book move must equal direct probe");
    assert_eq!(out.depth, 0, "book hit must report depth 0");
    assert_eq!(out.nodes, 0, "book hit must report 0 nodes (no search ran)");
}

/// (2) Book miss: with NO book configured, the engine at fixed depth must match a
/// plain UCI-style search (same best move + node count + score).
#[test]
fn embedded_engine_matches_uci_on_book_miss() {
    clrsrc::init_tables();
    const DEPTH: i32 = 10;
    let fens = [
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    ];

    for fen in fens {
        // Reference: plain search path.
        let ttbl = tt::TTable::new_shared(64);
        let mut ref_info = SearchInfo::new(ttbl);
        let loaded = ref_info.nnue.load(NNUE).is_ok();
        ref_info.silent = true;
        let mut ref_pos = Position::from_fen(fen).expect("valid FEN");
        if loaded { ref_info.nnue.refresh(&ref_pos, &mut ref_info.acc_stack[0]); }
        ref_info.hash_history.clear();
        ref_info.hash_history.push(ref_pos.hash);
        ref_info.game_history_len = 1;
        ref_info.clear_for_search();
        let tc = TimeControl { wtime: 0, btime: 0, winc: 0, binc: 0, movestogo: 0, movetime: 0,
                              depth: DEPTH, infinite: false, nodes: 0, soft_nodes: 0 };
        let mut tm = TimeManager::new(&tc, ref_pos.side == types::WHITE, 0);
        let ref_best = search::search(&mut ref_pos, &mut ref_info, &mut tm);
        let (ref_nodes, ref_score) = (ref_info.nodes, ref_info.root_score);

        // Embedded engine, bookless, same fixed depth.
        let mut engine = EmbeddedEngine::init(EmbeddedConfig {
            eval_file: Some(NNUE.to_string()),
            ..Default::default()
        });
        let limits = EmbeddedLimits {
            tc: TimeControl { wtime: 0, btime: 0, winc: 0, binc: 0, movestogo: 0, movetime: 0,
                             depth: DEPTH, infinite: false, nodes: 0, soft_nodes: 0 },
            max_deadline: far_deadline(),
            ponder: false, game_ply: 0, depth: DEPTH, nodes: 0,
        };
        let cancel = AtomicBool::new(false);
        let out = engine.search_position(fen, &[], limits, &cancel);

        assert_eq!(out.best, ref_best, "best move diverged on {}", fen);
        assert_eq!(out.nodes, ref_nodes, "nodes diverged on {} (emb {} vs ref {})", fen, out.nodes, ref_nodes);
        assert_eq!(out.score_cp, ref_score, "score diverged on {}", fen);
    }
}
