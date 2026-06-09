// ---- UCI protocol handler ----
// Supports Lazy SMP: multiple search threads share a TT.

use std::io::{self, BufRead};
use std::sync::Arc;
use std::thread;
use crate::types::*;
use crate::board::Position;
use crate::perft;
use crate::search::{self, SearchInfo};
use crate::tune;
use crate::book::{Book, ExpBook, ExpEntry, OverlayWriter, encode_poly_move, polyglot_hash,
    EXP_SOURCE_ENGINE, EXP_FLAG_VALIDATED, EXP_FLAG_MATE, EXP_SCORE_NONE};
use std::sync::Mutex;
use crate::tt::{self, SharedTT};
use crate::time::{self, TimeControl, TimeManager};

pub fn uci_loop() {
    let stdin = io::stdin();
    let mut pos = Position::startpos();
    let mut tt = tt::TTable::new_shared(64);
    let mut num_threads: usize = 1;
    let mut info = SearchInfo::new(Arc::clone(&tt));
    // Self-contained default: load the embedded NNUE so a bare binary plays with
    // NNUE eval out-of-the-box (CCRL / no EvalFile). A later `setoption EvalFile`
    // overrides this (e.g. for SPRT net-swaps).
    let _ = info.nnue.load_embedded();
    let mut game_ply: u32 = 0;

    // Book support
    let mut own_book = false;
    let mut best_book_move = true;
    let mut book: Option<Book> = None;
    // JBK2 experience/book (read-only consumer); opt-in via PlayFromExp.
    let mut exp: Option<ExpBook> = None;
    let mut play_from_exp = false;
    let mut book_variety: u8 = 0;
    // Experience-learning write path (opt-in via LearnDuringPlay; all default off → frozen behavior).
    let mut learn = false;
    let mut exp_min_depth: i32 = 16;
    let mut exp_path: Option<String> = None;
    // Created lazily once learning is on and an ExpFile path is known. Shared with the search
    // thread (1 write/move at the root, so a mutex is uncontended).
    let mut exp_writer: Option<Arc<Mutex<OverlayWriter>>> = None;
    // Advancing RNG state for book variety (seeded once from the wall clock, then incremented
    // per probe so consecutive book moves decorrelate regardless of clock granularity).
    let mut exp_rng: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0)
        | 1;

    // Search thread handle for ponder support
    let mut search_handle: Option<thread::JoinHandle<(Move, SearchInfo, i32)>> = None;

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        let tokens: Vec<&str> = line.split_whitespace().collect();
        if tokens.is_empty() {
            continue;
        }

        match tokens[0] {
            "uci" => {
                println!("id name clrsrc {}", env!("CARGO_PKG_VERSION"));
                println!("id author clrsrc contributors");
                println!("option name Hash type spin default 64 min 1 max 65536");
                println!("option name Threads type spin default 1 min 1 max 256");
                println!("option name EvalFile type string default <empty>");
                println!("option name OwnBook type check default false");
                println!("option name BestBookMove type check default true");
                println!("option name BookFile type string default <empty>");
                println!("option name ExpFile type string default <empty>");
                println!("option name PlayFromExp type check default false");
                println!("option name LearnDuringPlay type check default false");
                println!("option name ExpMinSaveDepth type spin default 16 min 1 max 100");
                println!("option name BookVariety type spin default 0 min 0 max 2");
                println!("option name SyzygyPath type string default <empty>");
                println!("option name SyzygyProbeDepth type spin default 1 min 1 max 100");
                println!("option name SyzygyProbeLimit type spin default 6 min 0 max 7");
                println!("option name Syzygy50MoveRule type check default true");
                println!("option name Ponder type check default true");
                println!("option name LongTimeNMP type check default false");
                // Only show tuning options if requested (avoid GUI clutter)
                // GUIs can still set them without seeing them
                for p in tune::all_params() {
                    println!("option name {} type spin default {} min {} max {}", p.name, p.default, p.min, p.max);
                }
                println!("uciok");
            }
            "setoption" => {
                cmd_setoption(&tokens, &mut info, &mut tt, &mut num_threads, &mut own_book, &mut best_book_move, &mut book, &mut exp, &mut play_from_exp, &mut book_variety, &mut learn, &mut exp_min_depth, &mut exp_path);
            }
            "isready" => {
                println!("readyok");
            }
            "ucinewgame" => {
                pos = Position::startpos();
                tt.clear();
                info.history.clear();
                info.killers.clear();
                info.counter_moves.clear();
                info.cont_history.clear();
                info.cont_history_2.clear();
                info.capture_history.clear();
                info.correction_history.clear();
                info.hash_history.clear();
                info.game_history_len = 0;
                game_ply = 0;
            }
            "position" => {
                game_ply = cmd_position(&tokens, &mut pos, &mut info.hash_history);
                info.game_history_len = info.hash_history.len();
            }
            "go" => {
                // Wait for any previous search to finish.
                // Ordering assumption: `info = ret_info` here restores the searching
                // thread's info and OVERWRITES anything written to `info` since it was
                // spawned. So a `position` must be preceded by a `stop`/`ponderhit` (which
                // joins first) — the non-standard `go ponder → position → go` would lose the
                // hash_history that `position` wrote onto the placeholder. GUIs follow the
                // standard order, so this is a documented coupling, not a live bug.
                if let Some(handle) = search_handle.take() {
                    time::set_stop(true);
                    if let Ok((_, ret_info, _)) = handle.join() {
                        info = ret_info;
                    } else {
                        // Search thread panicked. `info` is the placeholder, which already
                        // carries an NNUE clone (see below), so eval stays correct; warn so
                        // the panic is visible rather than silently degrading.
                        eprintln!("info string WARNING: search thread panicked; NNUE preserved, search state reset");
                    }
                }

                // Parse go parameters
                let (tc, mut is_ponder) = parse_go(&tokens);
                // Bug D fix (2026-05-16): honor `setoption name Ponder value false` —
                // if pondering disabled by GUI, treat `go ponder` as a normal `go` so the
                // PONDERING atomic never gets set and time.rs hard-cap stays effective.
                if is_ponder && !time::allow_ponder() {
                    eprintln!("info string ignoring `go ponder` (Ponder=false)");
                    is_ponder = false;
                }
                let is_white = pos.side == crate::types::WHITE;

                // Book probe (not during ponder): try the JBK2 experience book first
                // (richer: WDL-filtered), then fall back to the Polyglot book.
                if !is_ponder && !tc.infinite && tc.depth == 0 {
                    if play_from_exp {
                        if let Some(ref eb) = exp {
                            // Advancing counter mixed with the position hash → good variety even
                            // when the wall-clock seed has coarse granularity.
                            exp_rng = exp_rng.wrapping_add(0x9E3779B97F4A7C15);
                            let rng_val = exp_rng ^ pos.hash;
                            if let Some(exp_move) = eb.probe_best(&mut pos, book_variety, rng_val) {
                                println!("bestmove {}", exp_move);
                                continue;
                            }
                        }
                    }
                    if own_book {
                        if let Some(ref bk) = book {
                            let rng_val = pos.hash ^ (game_ply as u64 * 6364136223846793005);
                            if let Some(book_move) = bk.probe(&mut pos, rng_val, best_book_move) {
                                println!("bestmove {}", book_move);
                                continue;
                            }
                        }
                    }
                }

                if is_ponder {
                    time::set_pondering(true);
                } else {
                    time::set_pondering(false);
                }
                time::set_stop(false);

                // Start search in background thread (with Lazy SMP helpers)
                let fen = pos.to_fen();
                let tt = Arc::clone(&info.tt);
                let search_tc = tc;
                let smp_threads = num_threads;

                // Experience learning: lazily open the overlay writer once, then hand a cheap
                // Arc clone to the search thread. Captured copies keep the closure self-contained.
                if learn && exp_writer.is_none() {
                    if let Some(ref p) = exp_path {
                        let overlay = format!("{}.overlay", p);
                        exp_writer = Some(Arc::new(Mutex::new(OverlayWriter::open(&overlay))));
                        eprintln!("info string LearnDuringPlay writing to {}", overlay);
                    } else {
                        eprintln!("info string LearnDuringPlay enabled but no ExpFile set; nothing written");
                    }
                }
                let learn_writer = if learn { exp_writer.clone() } else { None };
                let learn_min_depth = exp_min_depth;
                let learn_is_ponder = is_ponder;

                // Move full info (including NNUE) into thread; create fresh placeholder.
                // Give the placeholder a cheap Arc-clone of the loaded NNUE so that if the
                // search thread panics (handle.join() => Err), subsequent moves still evaluate
                // with NNUE instead of silently falling back to classical eval (~-200 ELO,
                // invisible). hash_history/game_history_len are refreshed by the next `position`
                // command; learned tables rebuild — only the once-loaded NNUE is unrecoverable.
                let mut placeholder = SearchInfo::new(Arc::clone(&tt));
                placeholder.nnue = info.nnue.clone();
                let mut thread_info = std::mem::replace(&mut info, placeholder);
                search_handle = Some(thread::spawn(move || {
                    let mut search_pos = Position::from_fen(&fen).unwrap();
                    // Initialize NNUE accumulator
                    if thread_info.nnue.is_loaded() {
                        thread_info.nnue.refresh(&search_pos, &mut thread_info.acc_stack[0]);
                    }
                    let mut tm = TimeManager::new(&search_tc, is_white, game_ply);

                    // Spawn Lazy SMP helper threads (N-1 helpers)
                    let mut helpers = Vec::new();
                    for _tid in 1..smp_threads {
                        let h_tt = Arc::clone(&thread_info.tt);
                        let h_fen = fen.clone();
                        let h_nnue = thread_info.nnue.clone(); // Arc-cloned, cheap
                        let h_history = thread_info.hash_history.clone();
                        let h_hist_len = thread_info.game_history_len;
                        let h_helper_nodes = Arc::clone(&thread_info.helper_nodes);
                        let h_tb_hits = Arc::clone(&thread_info.tb_hits);

                        helpers.push(thread::spawn(move || {
                            let mut h_pos = Position::from_fen(&h_fen).unwrap();
                            let mut h_info = SearchInfo::new(h_tt);
                            h_info.helper_nodes = h_helper_nodes;
                            h_info.tb_hits = h_tb_hits;
                            h_info.nnue = h_nnue;
                            h_info.hash_history = h_history;
                            h_info.game_history_len = h_hist_len;
                            h_info.silent = true; // helpers don't print UCI output

                            if h_info.nnue.is_loaded() {
                                h_info.nnue.refresh(&h_pos, &mut h_info.acc_stack[0]);
                            }

                            // Helpers search with infinite time (main thread controls stopping)
                            let h_tc = TimeControl::infinite();
                            let mut h_tm = TimeManager::new(&h_tc, is_white, game_ply);
                            search::search(&mut h_pos, &mut h_info, &mut h_tm);
                            // Final flush of remaining unsynced nodes
                            let delta = h_info.nodes - h_info.synced_nodes;
                            if delta > 0 {
                                h_info.helper_nodes.fetch_add(
                                    delta,
                                    std::sync::atomic::Ordering::Relaxed,
                                );
                            }
                        }));
                    }

                    // Main thread search
                    let best_move = search::search(&mut search_pos, &mut thread_info, &mut tm);

                    // Stop all helper threads
                    time::set_stop(true);
                    for h in helpers {
                        let _ = h.join();
                    }

                    // Experience learning: persist this root's deep judgment, once per move.
                    // Gated to fully-completed searches at/above the save depth; never during
                    // ponder (the root is a guessed line). Flush immediately for crash-safety
                    // (the bot watcher hard-kills between games, so buffering would lose data).
                    if !learn_is_ponder && best_move != Move::NULL {
                        if let Some(ref w) = learn_writer {
                            if thread_info.completed_depth >= learn_min_depth {
                                let root_pos = Position::from_fen(&fen).unwrap();
                                let rs = thread_info.root_score;
                                let is_mate = rs.abs() >= search::MATE_IN_MAX;
                                let score16 = rs.clamp(i16::MIN as i32 + 1, i16::MAX as i32) as i16;
                                let mut flags = EXP_FLAG_VALIDATED;
                                if is_mate {
                                    flags |= EXP_FLAG_MATE;
                                }
                                let entry = ExpEntry {
                                    key: polyglot_hash(&root_pos),
                                    packed_move: encode_poly_move(best_move),
                                    score: score16,
                                    depth: thread_info.completed_depth.clamp(0, i16::MAX as i32) as i16,
                                    count: 1,
                                    source: EXP_SOURCE_ENGINE,
                                    flags,
                                    wdl_w: 0,
                                    wdl_l: 0,
                                    nnue_eval: EXP_SCORE_NONE,
                                    // jug_score is Jugernaut-exclusive (JBK2 §11) — never written
                                    // by clrsrc. clrsrc's eval goes into clrsrc_score (offset 28),
                                    // gated by the SOURCE_CLRSRC bit. `score` mirrors it.
                                    jug_score: EXP_SCORE_NONE,
                                    sf_score: EXP_SCORE_NONE,
                                    clrsrc_score: score16,
                                };
                                if let Ok(mut wl) = w.lock() {
                                    wl.push(entry);
                                    wl.flush();
                                }
                            }
                        }
                    }

                    // Format bestmove with ponder move.
                    // Verify ponder move is legal in the position AFTER best_move —
                    // PV entries can come from TT collisions or SMP races and be stale.
                    // Bug D (2026-05-16): if Ponder=false, never append "ponder Y" — GUI must not
                    // be encouraged to send `go ponder Y` (which would risk PONDERING-stuck).
                    let ponder_move = if time::allow_ponder() && thread_info.pv_len[0] > 1 {
                        let candidate = thread_info.pv[0][1];
                        let undo = search_pos.make_move(best_move);
                        let mut legal = crate::types::MoveList::new();
                        crate::movegen::generate_legal(&mut search_pos, &mut legal);
                        let ok = legal.iter().any(|m| *m == candidate);
                        search_pos.unmake_move(best_move, undo);
                        if ok { Some(candidate) } else { None }
                    } else {
                        None
                    };

                    if !thread_info.silent {
                        if let Some(pm) = ponder_move {
                            println!("bestmove {} ponder {}", best_move, pm);
                        } else {
                            println!("bestmove {}", best_move);
                        }
                    }

                    let score = thread_info.root_score;
                    (best_move, thread_info, score)
                }));
            }
            "ponderhit" => {
                // Opponent played the expected move — reset clock and enforce time limits
                time::signal_ponderhit();
            }
            "stop" => {
                time::set_pondering(false);
                time::set_stop(true);
                // Wait for search thread to finish
                if let Some(handle) = search_handle.take() {
                    if let Ok((_, ret_info, _)) = handle.join() {
                        info = ret_info;
                    } else {
                        eprintln!("info string WARNING: search thread panicked; NNUE preserved, search state reset");
                    }
                }
            }
            "d" | "display" => {
                pos.print();
                println!("  Eval: {} cp", crate::eval::evaluate(&pos));
            }
            "perft" => {
                if tokens.len() > 1 {
                    if let Ok(depth) = tokens[1].parse::<u32>() {
                        let start = std::time::Instant::now();
                        let nodes = perft::perft_divide(&mut pos, depth);
                        let elapsed = start.elapsed();
                        let nps = if elapsed.as_millis() > 0 {
                            nodes * 1000 / elapsed.as_millis() as u64
                        } else {
                            nodes
                        };
                        println!("Time: {:?}, NPS: {}", elapsed, nps);
                    }
                }
            }
            "eval" => {
                println!("Eval: {} cp", crate::eval::evaluate(&pos));
            }
            "verify" => {
                // verify_engine runs real searches over `info`, the global STOP flag, and
                // the shared TT. A live background search would collide on all three (and
                // verify would run on the placeholder info), so assert none is in flight.
                // (`eval` above needs no such guard — it is a pure static evaluate(&pos).)
                debug_assert!(search_handle.is_none(), "verify issued while a search is in flight");
                verify_engine(&mut info, &book, best_book_move);
            }
            "quit" => {
                time::set_pondering(false);
                time::set_stop(true);
                if let Some(handle) = search_handle.take() {
                    let _ = handle.join();
                }
                break;
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn cmd_setoption(tokens: &[&str], info: &mut SearchInfo, tt: &mut SharedTT, num_threads: &mut usize,
                 own_book: &mut bool, best_book: &mut bool, book: &mut Option<Book>,
                 exp: &mut Option<ExpBook>, play_from_exp: &mut bool, book_variety: &mut u8,
                 learn: &mut bool, exp_min_depth: &mut i32, exp_path: &mut Option<String>) {
    if tokens.len() >= 5 && tokens[1] == "name" {
        let val_idx = tokens.iter().position(|&t| t == "value");
        if let Some(vi) = val_idx {
            if vi + 1 >= tokens.len() { return; }
            let name = tokens[2..vi].join(" ").to_lowercase();
            match name.as_str() {
                "hash" => {
                    if let Ok(mb) = tokens[vi + 1].parse::<usize>() {
                        let new_tt = tt::TTable::new_shared(mb.max(1).min(65536));
                        info.tt = Arc::clone(&new_tt);
                        *tt = new_tt;
                        println!("info string Hash set to {} MB ({} entries)", mb, tt.entry_count());
                    }
                }
                "threads" => {
                    if let Ok(t) = tokens[vi + 1].parse::<usize>() {
                        *num_threads = t.max(1).min(256);
                    }
                }
                "evalfile" => {
                    let path = tokens[vi + 1..].join(" ");
                    if path == "<empty>" || path.is_empty() {
                        eprintln!("info string NNUE disabled");
                    } else {
                        match info.nnue.load(&path) {
                            Ok(()) => {}
                            Err(e) => eprintln!("info string NNUE load error: {}", e),
                        }
                    }
                }
                "ownbook" => {
                    let val = tokens[vi + 1].to_lowercase();
                    *own_book = val == "true" || val == "1";
                    eprintln!("info string OwnBook = {}", *own_book);
                }
                "bestbookmove" => {
                    let val = tokens[vi + 1].to_lowercase();
                    *best_book = val == "true" || val == "1";
                }
                "syzygypath" => {
                    let path = tokens[vi + 1..].join(" ");
                    if path != "<empty>" && !path.is_empty() {
                        crate::tablebase::init(&path);
                    }
                }
                "syzygyprobedepth" => {
                    if let Ok(d) = tokens[vi + 1].parse::<usize>() {
                        crate::tablebase::set_probe_depth(d.max(1).min(100));
                    }
                }
                "syzygyprobelimit" => {
                    if let Ok(l) = tokens[vi + 1].parse::<usize>() {
                        crate::tablebase::set_probe_limit(l.min(7));
                    }
                }
                "syzygy50moverule" => {
                    let val = tokens[vi + 1].to_lowercase();
                    crate::tablebase::set_50move_rule(val == "true" || val == "1");
                }
                "bookfile" => {
                    let path = tokens[vi + 1..].join(" ");
                    if path == "<empty>" || path.is_empty() {
                        *book = None;
                        eprintln!("info string Book disabled");
                    } else {
                        *book = Book::load(&path);
                        if book.is_none() {
                            eprintln!("info string Book load failed: {}", path);
                        }
                    }
                }
                "expfile" => {
                    let path = tokens[vi + 1..].join(" ");
                    if path == "<empty>" || path.is_empty() {
                        *exp = None;
                        *exp_path = None;
                        eprintln!("info string ExpFile disabled");
                    } else {
                        *exp = ExpBook::load(&path);
                        if exp.is_none() {
                            eprintln!("info string ExpFile load failed: {}", path);
                        }
                        // Remember the path so the learning write path can target <path>.overlay,
                        // even when ExpFile fails to load as a readable book (fresh learning run).
                        *exp_path = Some(path);
                    }
                }
                "playfromexp" => {
                    let val = tokens[vi + 1].to_lowercase();
                    *play_from_exp = val == "true" || val == "1";
                    eprintln!("info string PlayFromExp = {}", *play_from_exp);
                }
                "learnduringplay" => {
                    let val = tokens[vi + 1].to_lowercase();
                    *learn = val == "true" || val == "1";
                    eprintln!("info string LearnDuringPlay = {}", *learn);
                }
                "expminsavedepth" => {
                    if let Ok(d) = tokens[vi + 1].parse::<i32>() {
                        *exp_min_depth = d.clamp(1, 100);
                        eprintln!("info string ExpMinSaveDepth = {}", *exp_min_depth);
                    }
                }
                "bookvariety" => {
                    if let Ok(v) = tokens[vi + 1].parse::<u8>() {
                        *book_variety = v.min(2);
                    }
                }
                "ponder" => {
                    // UCI standard: when Ponder=false, engine must NOT advertise ponder moves
                    // and must NOT enter pondering mode on `go ponder`. Without honoring this,
                    // PONDERING state can get stuck and bypass time.rs hard-cap.
                    let val = tokens[vi + 1].to_lowercase();
                    let on = val == "true" || val == "1";
                    time::set_allow_ponder(on);
                    eprintln!("info string Ponder = {}", on);
                }
                "longtimenmp" => {
                    // Hybrid-TC: at TC >= 30+0.3, NMP_DEPTH_DIV=5 confirmed +13.9 ELO LOS 99.6% (2026-05-09).
                    // At TC 10+0.1 (Match-TC default), NMP_DEPTH_DIV=5 was -4.3 LOS 28.7%.
                    let val = tokens[vi + 1].to_lowercase();
                    let on = val == "true" || val == "1";
                    tune::set(&tune::NMP_DEPTH_DIV, if on { 5 } else { 6 });
                    eprintln!("info string LongTimeNMP = {} (NMP_DEPTH_DIV = {})", on, if on { 5 } else { 6 });
                }
                _ => {
                    // Check tunable parameters
                    for p in tune::all_params() {
                        if name == p.name.to_lowercase() {
                            if let Ok(v) = tokens[vi + 1].parse::<i32>() {
                                tune::set(p.param, v.max(p.min).min(p.max));
                            }
                            break;
                        }
                    }
                }
            }
        }
    }
}

fn cmd_position(tokens: &[&str], pos: &mut Position, hash_history: &mut Vec<u64>) -> u32 {
    let mut idx = 1;
    let mut ply_count: u32 = 0;
    hash_history.clear();

    if idx >= tokens.len() { return 0; }

    if tokens[idx] == "startpos" {
        *pos = Position::startpos();
        idx += 1;
    } else if tokens[idx] == "fen" {
        idx += 1;
        let mut fen_parts = Vec::new();
        while idx < tokens.len() && tokens[idx] != "moves" {
            fen_parts.push(tokens[idx]);
            idx += 1;
        }
        let fen = fen_parts.join(" ");
        match Position::from_fen(&fen) {
            Ok(p) => *pos = p,
            Err(e) => eprintln!("Invalid FEN: {}", e),
        }
    }

    // Store initial position hash
    hash_history.push(pos.hash);

    if idx < tokens.len() && tokens[idx] == "moves" {
        idx += 1;
        while idx < tokens.len() {
            if let Some(mv) = pos.parse_uci_move(tokens[idx]) {
                pos.make_move(mv);
                hash_history.push(pos.hash);
                ply_count += 1;
            } else {
                eprintln!("Invalid move: {}", tokens[idx]);
                break;
            }
            idx += 1;
        }
    }

    ply_count
}

/// Parse `go` command tokens into TimeControl + ponder flag
fn parse_go(tokens: &[&str]) -> (TimeControl, bool) {
    let mut tc = TimeControl {
        wtime: 0, btime: 0, winc: 0, binc: 0,
        movestogo: 0, movetime: 0, depth: 0,
        infinite: false, nodes: 0, soft_nodes: 0,
    };
    let mut is_ponder = false;

    let mut i = 1;
    while i < tokens.len() {
        match tokens[i] {
            "wtime" => { i += 1; tc.wtime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "btime" => { i += 1; tc.btime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "winc" => { i += 1; tc.winc = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "binc" => { i += 1; tc.binc = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "movestogo" => { i += 1; tc.movestogo = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "movetime" => { i += 1; tc.movetime = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "depth" => { i += 1; tc.depth = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "nodes" => { i += 1; tc.nodes = tokens.get(i).and_then(|s| s.parse().ok()).unwrap_or(0); }
            "infinite" => { tc.infinite = true; }
            "ponder" => { is_ponder = true; }
            _ => {}
        }
        i += 1;
    }

    if !tc.infinite && tc.depth == 0 && tc.nodes == 0 && tc.movetime == 0 && tc.wtime == 0 && tc.btime == 0 && !is_ponder {
        tc.depth = 10;
    }

    (tc, is_ponder)
}

/// Verify engine sanity: test book moves and search on standard positions.
/// Reports PASS/FAIL for each test.
fn verify_engine(info: &mut SearchInfo, book: &Option<Book>, best_book: bool) {
    println!("=== Engine Verification ===");
    let mut pass = 0;
    let mut fail = 0;

    // --- Book tests (if book loaded) ---
    if let Some(ref bk) = book {
        // Expected best book moves for standard openings
        let book_tests: &[(&str, &str, &[&str])] = &[
            // (description, moves, acceptable_replies)
            ("French: 1.e4 e6 2.d4", "e2e4 e7e6 d2d4", &["d7d5"]),
            ("Sicilian: 1.e4 c5", "e2e4 c7c5", &["g1f3", "b1c3", "c2c3", "f2f4", "d2d4"]),
            ("Italian: 1.e4 e5 2.Nf3 Nc6 3.Bc4", "e2e4 e7e5 g1f3 b8c6 f1c4", &["f8c5", "g8f6", "f8e7"]),
            ("QGD: 1.d4 d5 2.c4", "d2d4 d7d5 c2c4", &["e7e6", "c7c6", "d5c4"]),
            ("Startpos White", "", &["e2e4", "d2d4", "g1f3", "c2c4"]),
            ("KID: 1.d4 Nf6 2.c4 g6", "d2d4 g8f6 c2c4 g7g6", &["b1c3", "g1f3"]),
        ];

        println!("\n--- Book Move Tests ---");
        for &(desc, moves, expected) in book_tests {
            let mut pos = Position::startpos();
            if !moves.is_empty() {
                for m in moves.split_whitespace() {
                    if let Some(mv) = pos.parse_uci_move(m) {
                        pos.make_move(mv);
                    }
                }
            }
            let rng_val = pos.hash;
            if let Some(book_move) = bk.probe(&mut pos, rng_val, best_book) {
                let move_str = format!("{}", book_move);
                let ok = expected.contains(&move_str.as_str());
                if ok {
                    println!("  PASS  {}: {} (expected: {:?})", desc, move_str, expected);
                    pass += 1;
                } else {
                    println!("  FAIL  {}: {} NOT IN {:?}", desc, move_str, expected);
                    fail += 1;
                }
            } else {
                println!("  SKIP  {}: no book move", desc);
            }
        }
    } else {
        println!("  SKIP  Book tests (no book loaded)");
    }

    // --- Search tests: tactical positions ---
    let search_tests: &[(&str, &str, &[&str], i32)] = &[
        // (description, FEN, expected_moves, search_depth)
        ("Mate in 1", "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1", &["e1e8"], 4),
        ("Fork Nf7+", "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", &["h5f7"], 4),
        ("Avoid blunder", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            &["e7e5", "c7c5", "e7e6", "c7c6", "d7d5", "g8f6", "d7d6", "g7g6"], 8),
    ];

    println!("\n--- Search Tests (depth {}) ---", search_tests[0].3);
    for &(desc, fen, expected, depth) in search_tests {
        match Position::from_fen(fen) {
            Ok(mut pos) => {
                if info.nnue.is_loaded() {
                    info.nnue.refresh(&pos, &mut info.acc_stack[0]);
                }
                info.clear_for_search();
                let tc = TimeControl { depth, ..TimeControl::infinite() };
                let mut tm = TimeManager::new(&tc, pos.side == crate::types::WHITE, 0);
                time::set_stop(false);
                info.silent = true;
                let best = search::search(&mut pos, info, &mut tm);
                info.silent = false;
                let move_str = format!("{}", best);
                let ok = expected.contains(&move_str.as_str());
                let score = info.root_score;
                if ok {
                    println!("  PASS  {}: {} (score: {} cp)", desc, move_str, score);
                    pass += 1;
                } else {
                    println!("  FAIL  {}: {} NOT IN {:?} (score: {} cp)", desc, move_str, expected, score);
                    fail += 1;
                }
            }
            Err(e) => println!("  ERROR {}: {}", desc, e),
        }
    }

    // --- NPS test ---
    println!("\n--- NPS Test ---");
    {
        let mut pos = Position::startpos();
        if info.nnue.is_loaded() {
            info.nnue.refresh(&pos, &mut info.acc_stack[0]);
        }
        info.clear_for_search();
        let tc = TimeControl { depth: 12, ..TimeControl::infinite() };
        let mut tm = TimeManager::new(&tc, true, 0);
        time::set_stop(false);
        info.silent = true;
        search::search(&mut pos, info, &mut tm);
        info.silent = false;
        let elapsed = tm.elapsed_ms().max(1);
        let nps = info.nodes * 1000 / elapsed as u64;
        println!("  Depth 12: {} nodes, {} NPS, {} ms", info.nodes, nps, elapsed);
        if nps > 500_000 {
            println!("  PASS  NPS > 500K");
            pass += 1;
        } else {
            println!("  FAIL  NPS too low: {}", nps);
            fail += 1;
        }
    }

    println!("\n=== Result: {} passed, {} failed ===", pass, fail);
}
