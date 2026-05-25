// ---- Perft: move generation validation ----

use crate::types::*;
use crate::board::Position;
use crate::movegen;

/// Count leaf nodes at given depth
pub fn perft(pos: &mut Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut list = MoveList::new();
    movegen::generate_all(pos, &mut list);

    let mut nodes = 0u64;

    for i in 0..list.len {
        let mv = list.moves[i];
        let undo = pos.make_move(mv);

        // Legality check: king of side that just moved must not be in check
        let ksq = pos.king_sq(pos.side.flip());
        if !crate::attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side) {
            nodes += perft(pos, depth - 1);
        }

        pos.unmake_move(mv, undo);
    }

    nodes
}

/// Perft with divide: show per-move node counts at root
pub fn perft_divide(pos: &mut Position, depth: u32) -> u64 {
    let mut list = MoveList::new();
    movegen::generate_all(pos, &mut list);

    let mut total = 0u64;

    for i in 0..list.len {
        let mv = list.moves[i];
        let undo = pos.make_move(mv);

        let ksq = pos.king_sq(pos.side.flip());
        if !crate::attacks::is_attacked(&pos.pieces, &pos.colors, ksq, pos.side) {
            let nodes = if depth <= 1 { 1 } else { perft(pos, depth - 1) };
            println!("{}: {}", mv, nodes);
            total += nodes;
        }

        pos.unmake_move(mv, undo);
    }

    println!("\nTotal: {}", total);
    total
}

/// Standard perft test suite
pub fn run_suite() {
    let tests: Vec<(&str, Vec<u64>)> = vec![
        // Position 1: Starting position
        (
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            vec![20, 400, 8_902, 197_281, 4_865_609],
        ),
        // Position 2: Kiwipete
        (
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            vec![48, 2_039, 97_862, 4_085_603],
        ),
        // Position 3
        (
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            vec![14, 191, 2_812, 43_238, 674_624],
        ),
        // Position 4
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            vec![6, 264, 9_467, 422_333],
        ),
        // Position 5
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            vec![44, 1_486, 62_379, 2_103_487],
        ),
        // Position 6
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            vec![46, 2_079, 89_890, 3_894_594],
        ),
    ];

    let mut all_passed = true;

    for (i, (fen, expected)) in tests.iter().enumerate() {
        print!("Position {}: ", i + 1);
        let mut pos = Position::from_fen(fen).expect("valid FEN");

        let mut passed = true;
        for (depth, &exp) in expected.iter().enumerate() {
            let depth = (depth + 1) as u32;
            let result = perft(&mut pos, depth);
            if result != exp {
                println!("FAIL at depth {} — got {} expected {}", depth, result, exp);
                passed = false;
                all_passed = false;
                break;
            }
        }
        if passed {
            let max_depth = expected.len();
            println!("OK (depth 1-{})", max_depth);
        }
    }

    if all_passed {
        println!("\nAll perft tests PASSED!");
    } else {
        println!("\nSome perft tests FAILED!");
    }
}
