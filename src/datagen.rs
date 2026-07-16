// ---- Data Generation: Self-play for NNUE Training ----
// Output: Bullet-compatible binary format (32 bytes per position)
// Usage: clrsrc datagen <games> <depth> <output.bin> [threads]
// Note: depth parameter is now ignored — search uses soft/hard node limits instead.

use crate::types::*;
use crate::board::Position;
use crate::movegen;
use crate::search::{self, SearchInfo, MINIMUM_MATE_SCORE};
use crate::time::{TimeControl, TimeManager};
use crate::attacks;
use crate::bitboard::pop_lsb;
use crate::book::Book;

use std::io::Write;
use std::sync::atomic::{AtomicU64, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

// ---- Bullet-compatible binary format ----
// 32 bytes per position, matching bulletformat::ChessBoard layout:
//   occ:      u64    (8 bytes)  - occupancy bitboard
//   pcs:      [u8;16](16 bytes) - 4-bit packed pieces in popcount order of occ
//   score:    i16    (2 bytes)  - centipawn eval from STM perspective
//   result:   u8     (1 byte)   - 2=white win, 1=draw, 0=black win
//   ksq:      u8     (1 byte)   - STM king square
//   opp_ksq:  u8     (1 byte)   - NSTM king square
//   extra:    u8     (1 byte)   - bit 0 = stm (0=white, 1=black)
//   _pad:     [u8;2] (2 bytes)  - padding

// Win adjudication
const WIN_ADJ_SCORE: i32 = 3000;     // |score| >= this for N moves → adjudicate win
const WIN_ADJ_COUNT: i32 = 5;        // consecutive moves above threshold

// Draw adjudication (critical for data quality)
const DRAW_ADJ_SCORE: i32 = 10;      // |score| <= this for N plies → adjudicate draw
const DRAW_ADJ_COUNT: i32 = 10;      // consecutive plies below threshold
const DRAW_ADJ_MIN_PLY: u32 = 70;    // only adjudicate draws after this ply

const MAX_GAME_MOVES: u32 = 512;
const RANDOM_OPENING_PLIES: u32 = 8;
const MIN_RECORD_PLY: u32 = 8; // skip first N plies after random opening (P4 21.05: 4→8, more book overlap tolerance)
const SCORE_FILTER: i32 = 10000; // don't record positions with |score| above this
const OPENING_VERIFY_LIMIT: i32 = 1000; // reject openings with |eval| > this

// Node-based search safety cutoff (per-position soft node target is passed in via CLI)
const HARD_NODE_LIMIT: u64 = 10_000_000; // safety cutoff

/// Pack a mailbox piece into 4-bit Bullet format
/// bits [2:0] = piece type (0=P, 1=N, 2=B, 3=R, 4=Q, 5=K)
/// bit  [3]   = color (0=white, 1=black)
#[inline]
fn pack_piece(mailbox_piece: u8) -> u8 {
    let pt = piece_type(mailbox_piece).index() as u8;
    let color = piece_color(mailbox_piece).index() as u8;
    pt | (color << 3)
}

/// Pack position into 32-byte Bullet-compatible entry
fn pack_position(pos: &Position, score: i32) -> [u8; 32] {
    let occ = pos.colors[0] | pos.colors[1];
    let mut pcs = [0u8; 16];

    // Pack pieces in popcount order (LSB first in occupancy)
    let mut occ_iter = occ;
    let mut idx = 0usize;
    while occ_iter != 0 {
        let sq = pop_lsb(&mut occ_iter);
        let piece = pos.mailbox[sq as usize];
        let packed = pack_piece(piece);
        if idx % 2 == 0 {
            pcs[idx / 2] = packed;
        } else {
            pcs[idx / 2] |= packed << 4;
        }
        idx += 1;
    }

    // Score from STM perspective (Bullet expects this)
    let stm_score = score.max(-32000).min(32000) as i16;

    let stm_king = pos.king_sq(pos.side);
    let nstm_king = pos.king_sq(pos.side.flip());

    let mut buf = [0u8; 32];
    buf[0..8].copy_from_slice(&occ.to_le_bytes());
    buf[8..24].copy_from_slice(&pcs);
    buf[24..26].copy_from_slice(&stm_score.to_le_bytes());
    // buf[26] = result — filled in later
    buf[27] = stm_king;
    buf[28] = nstm_king;
    buf[29] = pos.side.index() as u8;
    // buf[30..32] = padding (already 0)
    buf
}

// ---- Simple xoshiro256** PRNG ----

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

    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}

/// Get all legal moves for a position
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

/// Play a single self-play game, return collected positions.
/// Uses separate TTs per color and node-based search.
fn play_game(
    info_white: &mut SearchInfo,
    info_black: &mut SearchInfo,
    rng: &mut Rng,
    book: Option<&Book>,
    soft_nodes: u64,
) -> Vec<[u8; 32]> {
    let mut pos = Position::startpos();

    // Opening phase: use book if available, else random moves
    let mut book_plies = 0u32;
    if let Some(bk) = book {
        loop {
            let rng_val = rng.next_u64();
            if let Some(mv) = bk.probe(&mut pos, rng_val, false) {
                pos.make_move(mv);
                book_plies += 1;
            } else {
                break;
            }
            if book_plies >= 40 {
                break;
            }
        }
        if book_plies < 4 {
            for _ in book_plies..RANDOM_OPENING_PLIES {
                let legal = get_legal_moves(&mut pos);
                if legal.is_empty() {
                    return Vec::new();
                }
                let idx = rng.next_usize(legal.len());
                pos.make_move(legal[idx]);
            }
        }
    } else {
        for _ in 0..RANDOM_OPENING_PLIES {
            let legal = get_legal_moves(&mut pos);
            if legal.is_empty() {
                return Vec::new();
            }
            let idx = rng.next_usize(legal.len());
            pos.make_move(legal[idx]);
        }
    }

    // Verify opening position is playable
    {
        let info = if pos.side == WHITE { &mut *info_white } else { &mut *info_black };
        info.hash_history.clear();
        info.hash_history.push(pos.hash);
        info.game_history_len = 1;
        info.clear_for_search();
        crate::time::set_stop(false);
        let tc = TimeControl {
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, movetime: 0, depth: 4,
            infinite: false, nodes: 10_000, soft_nodes: 0,
        };
        let mut tm = TimeManager::new(&tc, pos.side == WHITE, 0);
        search::search(&mut pos, info, &mut tm);
        if info.root_score.abs() > OPENING_VERIFY_LIMIT {
            return Vec::new(); // reject this opening
        }
    }

    let mut game_entries: Vec<[u8; 32]> = Vec::new();
    let mut hash_history: Vec<u64> = Vec::new();
    let mut adj_white_count = 0i32;
    let mut adj_black_count = 0i32;
    let mut draw_adj_count = 0i32;
    let mut game_result: u8 = 1; // default draw
    let mut ply_count = 0u32;

    loop {
        if ply_count >= MAX_GAME_MOVES {
            break;
        }

        hash_history.push(pos.hash);

        // 3-fold repetition
        let rep_count = hash_history.iter().filter(|&&h| h == pos.hash).count();
        if rep_count >= 3 {
            break;
        }

        // 50-move rule
        if pos.halfmove_clock >= 100 {
            break;
        }

        // Check for game over
        let legal = get_legal_moves(&mut pos);
        if legal.is_empty() {
            if pos.is_in_check() {
                game_result = if pos.side == WHITE { 0 } else { 2 };
            }
            break;
        }

        // Select the correct SearchInfo (separate TT per color)
        let info = if pos.side == WHITE { &mut *info_white } else { &mut *info_black };

        // Populate search hash_history for repetition detection
        info.hash_history = hash_history.clone();
        info.game_history_len = info.hash_history.len();

        // Search with node-based limits
        info.clear_for_search();
        crate::time::set_stop(false);
        // P3a (21.05): infinite=true bypasses TimeManager's wtime=0 → 100ms-cap default,
        // so soft_nodes actually limits the search (was effectively no-op before).
        // Now 500K soft_nodes = ~1s/pos on mobile, 1M = ~2s/pos — real depth labels.
        let tc = TimeControl {
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, movetime: 0, depth: 0,
            infinite: true, nodes: HARD_NODE_LIMIT, soft_nodes,
        };
        let mut tm = TimeManager::new(&tc, pos.side == WHITE, 0);
        let best_move = search::search(&mut pos, info, &mut tm);
        let score = info.root_score;

        if best_move == Move::NULL {
            break;
        }

        // Win adjudication
        if score.abs() >= WIN_ADJ_SCORE {
            if (pos.side == WHITE && score > 0) || (pos.side == BLACK && score < 0) {
                adj_white_count += 1;
                adj_black_count = 0;
            } else {
                adj_black_count += 1;
                adj_white_count = 0;
            }
        } else {
            adj_white_count = 0;
            adj_black_count = 0;
        }

        if adj_white_count >= WIN_ADJ_COUNT {
            game_result = 2;
            break;
        }
        if adj_black_count >= WIN_ADJ_COUNT {
            game_result = 0;
            break;
        }

        // Draw adjudication (only after min ply)
        if ply_count >= DRAW_ADJ_MIN_PLY && score.abs() <= DRAW_ADJ_SCORE {
            draw_adj_count += 1;
        } else {
            draw_adj_count = 0;
        }
        if draw_adj_count >= DRAW_ADJ_COUNT {
            game_result = 1; // draw
            break;
        }

        // Record position (with filters)
        let dominated_by_mate = score.abs() >= MINIMUM_MATE_SCORE;
        let dominated_by_score = score.abs() >= SCORE_FILTER;
        let in_check = pos.is_in_check();
        let too_early = ply_count < MIN_RECORD_PLY;

        // Noisy move filtering: skip positions where best move is capture or promotion
        let best_is_capture = best_move.is_capture() || best_move.is_ep();
        let best_is_promo = best_move.is_promotion();

        // P4 (21.05): gives-check filter — exclude positions where best move puts opponent in check.
        // Forcing checks are typically "obvious" moves for NN to learn (low information content) and
        // bias the dataset toward tactical sequences. Bullet filter_v2 standard.
        // Cost: extra make+unmake per ply, negligible vs 500K-1M node search.
        let undo_peek = pos.make_move(best_move);
        let best_gives_check = pos.is_in_check();
        pos.unmake_move(best_move, undo_peek);

        if !in_check && !dominated_by_mate && !dominated_by_score && !too_early
            && !best_is_capture && !best_is_promo && !best_gives_check
        {
            let entry = pack_position(&pos, score);
            game_entries.push(entry);
        }

        pos.make_move(best_move);
        ply_count += 1;
    }

    // Fill in game result — STM-relative (bullet stores/expects result from the side-to-move's
    // perspective; see bulletformat ChessBoard::new flip for stm==1). game_result is WHITE-relative
    // (0=black win, 1=draw, 2=white win), so flip it for black-to-move positions. buf[29] = pos.side.
    // Bug fixed 2026-06-13: previously written unflipped → black-to-move WDL targets were inverted
    // (empirical decode: result==2 score>0 ratio 97% white-to-move vs 4.8% black-to-move). See Fact #61.
    for entry in &mut game_entries {
        let stm_is_white = entry[29] == 0;
        entry[26] = if stm_is_white { game_result } else { 2 - game_result };
    }

    game_entries
}

/// Run multi-threaded self-play data generation
pub fn run_datagen(num_games: u32, soft_nodes: u64, output_path: &str, num_threads: u32, book_path: Option<&str>) {
    let threads = num_threads.max(1);

    // Load opening book if specified
    let book: Option<Arc<Book>> = book_path.and_then(|path| {
        Book::load(path).map(Arc::new)
    });

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .expect("Failed to open output file");
    let writer = Arc::new(Mutex::new(std::io::BufWriter::with_capacity(
        4 * 1024 * 1024, file,
    )));

    let total_positions = Arc::new(AtomicU64::new(0));
    let games_done = Arc::new(AtomicU32::new(0));
    let start_time = std::time::Instant::now();

    let book_info = if book.is_some() { "with book" } else { "no book (random openings)" };
    eprintln!(
        "datagen: {} games, soft {} / hard {} nodes, {} threads, {}, output: {}",
        num_games, soft_nodes, HARD_NODE_LIMIT, threads, book_info, output_path
    );
    eprintln!(
        "datagen: draw adj |score|<={} for {} plies (after ply {}), noisy filtering ON, separate TTs",
        DRAW_ADJ_SCORE, DRAW_ADJ_COUNT, DRAW_ADJ_MIN_PLY
    );

    let mut handles = Vec::new();

    for thread_id in 0..threads {
        let writer = Arc::clone(&writer);
        let total_positions = Arc::clone(&total_positions);
        let games_done = Arc::clone(&games_done);
        let book = book.clone();

        // Divide games among threads
        let games_per_thread = num_games / threads;
        let extra = if thread_id < num_games % threads { 1 } else { 0 };
        let my_games = games_per_thread + extra;

        let handle = std::thread::spawn(move || {
            // Separate TTs per color (4MB each) — prevents info leaking between sides
            let tt_white = crate::tt::TTable::new_shared(4);
            let tt_black = crate::tt::TTable::new_shared(4);
            let mut info_white = SearchInfo::new(tt_white);
            let mut info_black = SearchInfo::new(tt_black);
            info_white.silent = true;
            info_black.silent = true;

            // Load NNUE: try current dir, then next to executable
            let mut nnue_found = false;
            let exe_dir = std::env::current_exe().ok()
                .and_then(|p| p.parent().map(|d| d.to_path_buf()));
            for name in &["clrsrc_v32_seed_b.nnue", "clrsrc_v16_kb4.nnue", "clrsrc_v11.nnue", "clrsrc.nnue"] {
                let candidates: Vec<std::path::PathBuf> = {
                    let mut v = vec![std::path::PathBuf::from(name)];
                    if let Some(ref dir) = exe_dir {
                        v.push(dir.join(name));
                        // Also try two levels up (target/release/../..)
                        if let Some(proj) = dir.parent().and_then(|d| d.parent()) {
                            v.push(proj.join(name));
                        }
                    }
                    v
                };
                for path in &candidates {
                    if path.exists() {
                        let ps = path.to_string_lossy().to_string();
                        if thread_id == 0 {
                            eprintln!("datagen: loading NNUE from {}", ps);
                        }
                        let _ = info_white.nnue.load(&ps);
                        let _ = info_black.nnue.load(&ps);
                        nnue_found = true;
                        break;
                    }
                }
                if nnue_found { break; }
            }
            if !nnue_found && thread_id == 0 {
                eprintln!("datagen: WARNING — no NNUE file found, using HCE eval");
            }

            // Unique seed per thread
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
                ^ ((thread_id as u64 + 1) * 0x517cc1b727220a95);
            let mut rng = Rng::new(seed);

            let mut local_positions = 0u64;
            let mut local_buf: Vec<u8> = Vec::with_capacity(64 * 1024);

            for local_game in 0..my_games {
                // Reset TTs periodically
                if local_game % 100 == 0 {
                    info_white.tt.clear();
                    info_black.tt.clear();
                }

                // Clear history tables EVERY game to prevent age() slowdown.
                for info in [&mut info_white, &mut info_black] {
                    info.history.clear();
                    info.killers.clear();
                    info.counter_moves.clear();
                    info.cont_history.clear();
                    info.capture_history.clear();
                }

                let entries = play_game(&mut info_white, &mut info_black, &mut rng, book.as_deref(), soft_nodes);
                local_positions += entries.len() as u64;

                // Buffer entries locally
                for entry in &entries {
                    local_buf.extend_from_slice(entry);
                }

                // Flush to shared writer every game — minimises buffered-data loss on
                // crash/reboot for slow single-producer datagen (1M soft_nodes). Lock
                // overhead per game is negligible vs multi-second searches.
                if local_buf.len() >= 32 {
                    if !local_buf.is_empty() {
                        let mut w = writer.lock().unwrap();
                        w.write_all(&local_buf).expect("Failed to write");
                        w.flush().expect("Failed to flush");
                        local_buf.clear();
                    }

                    let batch = if (local_game + 1) % 10 == 0 { 10 } else { (local_game + 1) % 10 };
                    let done = games_done.fetch_add(batch, Ordering::Relaxed);
                    total_positions.fetch_add(local_positions, Ordering::Relaxed);
                    let tp = total_positions.load(Ordering::Relaxed);
                    local_positions = 0;

                    // Thread 0 prints progress + writes status file
                    if thread_id == 0 && ((local_game + 1) % 10 == 0 || local_game + 1 == my_games) {
                        let actual_done = done + batch;
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let games_per_sec = actual_done as f64 / elapsed;
                        let pos_per_sec = tp as f64 / elapsed;
                        eprintln!(
                            "game ~{}/{}: {} positions total ({:.0} games/s, {:.0} pos/s, {} threads)",
                            actual_done, num_games, tp, games_per_sec, pos_per_sec, threads
                        );
                        // Write machine-readable status file
                        let _ = std::fs::write("datagen_status.txt", format!(
                            "{} {} {} {:.1} {:.1} {}",
                            tp, actual_done, num_games, pos_per_sec, games_per_sec, threads
                        ));
                    }
                }
            }
        });

        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().expect("Thread panicked");
    }

    // Final flush
    {
        let mut w = writer.lock().unwrap();
        w.flush().expect("Failed to flush output");
    }

    let elapsed = start_time.elapsed();
    let tp = total_positions.load(Ordering::Relaxed);
    let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "datagen complete: {} positions in {} games ({:.1}s, {:.1} MB, {} threads)",
        tp, num_games, elapsed.as_secs_f64(),
        file_size as f64 / (1024.0 * 1024.0),
        threads
    );
}
