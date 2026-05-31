//! C3 determinism gate (BOT_ENGINE_INTEGRATION_PLAN.md, Validierung Gate 2):
//! the embedded `search_embedded()` facade must produce the EXACT same result
//! as the plain UCI search path at the same limits.
//!
//! At a fixed depth the time computation is bypassed (TimeManager early-returns
//! on `tc.depth > 0`) and a far-future deadline never triggers the wall ceiling,
//! so any divergence would be the facade itself. We assert best move + node count
//! are bit-identical across a representative set of positions.

use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

use clrsrc::board::{self, Position};
use clrsrc::search::{self, SearchInfo};
use clrsrc::time::{TimeControl, TimeManager};
use clrsrc::tt;
use clrsrc::types;
use clrsrc::{search_embedded, EmbeddedLimits};

const NNUE: &str = "clrsrc_v32_seed_b.nnue";
const DEPTH: i32 = 10;

const POSITIONS: &[&str] = &[
    board::STARTPOS_FEN,
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "6k1/8/1p2R3/1Pb1p1B1/8/1r5P/6K1/8 w - - 4 45",
];

fn fresh_info(nnue_loaded: &mut bool) -> SearchInfo {
    let ttbl = tt::TTable::new_shared(64);
    let mut info = SearchInfo::new(ttbl);
    *nnue_loaded = info.nnue.load(NNUE).is_ok();
    info.silent = true;
    info
}

fn prepare(info: &mut SearchInfo, pos: &Position, nnue_loaded: bool) {
    if nnue_loaded {
        info.nnue.refresh(pos, &mut info.acc_stack[0]);
    }
    info.hash_history.clear();
    info.hash_history.push(pos.hash);
    info.game_history_len = 1;
    info.clear_for_search();
}

#[test]
fn embedded_matches_uci_path_at_fixed_depth() {
    clrsrc::zobrist::init();
    clrsrc::magic::init();
    clrsrc::attacks::init();
    search::init_lmr();

    for fen in POSITIONS {
        let is_white = Position::from_fen(fen).expect("valid FEN").side == types::WHITE;

        // --- Reference: plain UCI-style search at fixed depth ---
        let mut ref_loaded = false;
        let mut ref_info = fresh_info(&mut ref_loaded);
        let mut ref_pos = Position::from_fen(fen).expect("valid FEN");
        prepare(&mut ref_info, &ref_pos, ref_loaded);
        let ref_tc = TimeControl {
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, movetime: 0, depth: DEPTH,
            infinite: false, nodes: 0, soft_nodes: 0,
        };
        let mut ref_tm = TimeManager::new(&ref_tc, is_white, 0);
        let ref_best = search::search(&mut ref_pos, &mut ref_info, &mut ref_tm);
        let ref_nodes = ref_info.nodes;
        let ref_score = ref_info.root_score;

        // --- Embedded: same depth, far-future deadline, no ponder ---
        let mut emb_loaded = false;
        let mut emb_info = fresh_info(&mut emb_loaded);
        let mut emb_pos = Position::from_fen(fen).expect("valid FEN");
        prepare(&mut emb_info, &emb_pos, emb_loaded);
        let limits = EmbeddedLimits {
            tc: TimeControl {
                wtime: 0, btime: 0, winc: 0, binc: 0,
                movestogo: 0, movetime: 0, depth: DEPTH,
                infinite: false, nodes: 0, soft_nodes: 0,
            },
            max_deadline: Instant::now() + Duration::from_secs(3600),
            ponder: false,
            game_ply: 0,
            depth: DEPTH,
            nodes: 0,
        };
        let cancel = AtomicBool::new(false);
        let outcome = search_embedded(&mut emb_pos, &mut emb_info, limits, &cancel);

        assert_eq!(
            outcome.best, ref_best,
            "best move diverged on {} (embedded {:?} vs uci {:?})",
            fen, outcome.best, ref_best
        );
        assert_eq!(
            outcome.nodes, ref_nodes,
            "node count diverged on {} (embedded {} vs uci {})",
            fen, outcome.nodes, ref_nodes
        );
        assert_eq!(
            outcome.score_cp, ref_score,
            "score diverged on {}", fen
        );
    }
}
