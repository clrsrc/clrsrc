// ---- Stockfish Rescoring: Replace clrsrc eval with SF eval in training data ----
// Usage: clrsrc rescore <input.bin> <output.bin> [threads] [nodes] [sf_path]
//
// Reads Bullet-format .bin, sends each position to Stockfish via UCI,
// replaces the score field with SF's eval, preserves WDL result.
//
// **Resume support**: If output.bin already exists, counts completed positions
// and continues from where it left off. Safe to Ctrl-C and restart.
// Progress is checkpointed periodically so minimal work is lost on interrupt.

use crate::types::*;

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, Command, Stdio};
use std::sync::atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering};
use std::time::Instant;

const DEFAULT_SF_PATH: &str = "stockfish";
const DEFAULT_NODES: u64 = 25_000;
const DEFAULT_THREADS: u32 = 4;
const CHECKPOINT_INTERVAL_SECS: u64 = 60; // save progress every 60 seconds

// ---- FEN reconstruction from 32-byte Bullet entry ----

/// Unpack a 32-byte entry into a FEN string.
/// Returns (fen, original_score, result).
fn unpack_to_fen(buf: &[u8; 32]) -> (String, i16, u8) {
    let occ = u64::from_le_bytes(buf[0..8].try_into().unwrap());
    let pcs = &buf[8..24];
    let score = i16::from_le_bytes(buf[24..26].try_into().unwrap());
    let result = buf[26];
    let stm_black = (buf[29] & 1) != 0;

    // Reconstruct mailbox[64] from occupancy + packed pieces
    let mut mailbox = [EMPTY; 64];
    let mut occ_iter = occ;
    let mut idx = 0usize;
    while occ_iter != 0 {
        let sq = occ_iter.trailing_zeros() as usize;
        occ_iter &= occ_iter - 1;
        let nibble = if idx % 2 == 0 {
            pcs[idx / 2] & 0x0F
        } else {
            pcs[idx / 2] >> 4
        };
        let pt = nibble & 7;   // 0=P,1=N,2=B,3=R,4=Q,5=K
        let color = (nibble >> 3) & 1; // 0=white,1=black
        mailbox[sq] = color * 6 + pt;
        idx += 1;
    }

    // Build FEN piece placement (rank 8 down to rank 1)
    let mut fen = String::with_capacity(90);
    for rank in (0..8).rev() {
        let mut empty = 0u8;
        for file in 0..8 {
            let sq = rank * 8 + file;
            if mailbox[sq] == EMPTY {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(PIECE_CHARS[mailbox[sq] as usize] as char);
            }
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }

    // Side to move
    fen.push(' ');
    fen.push(if stm_black { 'b' } else { 'w' });

    // Castling rights heuristic: if king+rook on original squares, assume available
    fen.push(' ');
    let mut castling = String::new();
    if mailbox[E1 as usize] == make_piece(WHITE, KING) {
        if mailbox[H1 as usize] == make_piece(WHITE, ROOK) { castling.push('K'); }
        if mailbox[A1 as usize] == make_piece(WHITE, ROOK) { castling.push('Q'); }
    }
    if mailbox[E8 as usize] == make_piece(BLACK, KING) {
        if mailbox[H8 as usize] == make_piece(BLACK, ROOK) { castling.push('k'); }
        if mailbox[A8 as usize] == make_piece(BLACK, ROOK) { castling.push('q'); }
    }
    if castling.is_empty() {
        fen.push('-');
    } else {
        fen.push_str(&castling);
    }

    // EP=-, halfmove=0, fullmove=1
    fen.push_str(" - 0 1");

    (fen, score, result)
}

// ---- Stockfish UCI process ----

struct SfProcess {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    stdout: BufReader<std::process::ChildStdout>,
    line_buf: String,
}

impl SfProcess {
    fn new(sf_path: &str) -> Self {
        let mut child = Command::new(sf_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap_or_else(|e| panic!("Failed to start Stockfish at {}: {}", sf_path, e));

        let stdin = BufWriter::new(child.stdin.take().unwrap());
        let stdout = BufReader::new(child.stdout.take().unwrap());

        let mut sf = SfProcess { child, stdin, stdout, line_buf: String::with_capacity(256) };

        // Initialize UCI
        sf.send("uci");
        sf.wait_for("uciok");
        sf.send("setoption name Hash value 16");
        sf.send("setoption name Threads value 1");
        sf.send("isready");
        sf.wait_for("readyok");

        sf
    }

    fn send(&mut self, cmd: &str) {
        writeln!(self.stdin, "{}", cmd).unwrap();
        self.stdin.flush().unwrap();
    }

    fn read_line(&mut self) -> &str {
        self.line_buf.clear();
        self.stdout.read_line(&mut self.line_buf).unwrap();
        self.line_buf.trim_end()
    }

    fn wait_for(&mut self, token: &str) {
        loop {
            let line = self.read_line().to_string();
            if line.starts_with(token) {
                break;
            }
        }
    }

    /// Evaluate a position. Returns score in centipawns from STM perspective.
    fn eval(&mut self, fen: &str, nodes: u64) -> i16 {
        self.send(&format!("position fen {}", fen));
        self.send(&format!("go nodes {}", nodes));

        let mut last_score: i32 = 0;
        loop {
            let line = self.read_line().to_string();
            if line.starts_with("bestmove") {
                break;
            }
            // Parse "info ... score cp X ..." or "info ... score mate X ..."
            if let Some(idx) = line.find("score cp ") {
                let rest = &line[idx + 9..];
                if let Some(val) = rest.split_whitespace().next().and_then(|s| s.parse::<i32>().ok()) {
                    last_score = val;
                }
            } else if let Some(idx) = line.find("score mate ") {
                let rest = &line[idx + 11..];
                if let Some(val) = rest.split_whitespace().next().and_then(|s| s.parse::<i32>().ok()) {
                    last_score = if val > 0 { 30000 - val } else { -30000 - val };
                }
            }
        }

        last_score.clamp(-32000, 32000) as i16
    }
}

impl Drop for SfProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        let _ = self.child.wait();
    }
}

// ---- Main rescoring pipeline with resume support ----

pub fn run(args: &[String]) {
    let input = args.get(0).expect("Usage: clrsrc rescore <input.bin> <output.bin> [threads] [nodes] [sf_path]");
    let output = args.get(1).expect("Usage: clrsrc rescore <input.bin> <output.bin> [threads] [nodes] [sf_path]");
    let threads: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_THREADS);
    let nodes: u64 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_NODES);
    let sf_path = args.get(4).map(|s| s.as_str()).unwrap_or(DEFAULT_SF_PATH);

    // Read input
    let data = std::fs::read(input).unwrap_or_else(|e| panic!("Failed to read {}: {}", input, e));
    let total_entries = data.len() / 32;
    if data.len() % 32 != 0 {
        panic!("Input file size {} is not a multiple of 32", data.len());
    }

    // Check for existing output (resume support)
    let already_done = if std::path::Path::new(output).exists() {
        let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        let done = (out_size / 32) as usize;
        if out_size % 32 != 0 {
            panic!("Output file {} has invalid size {} (not a multiple of 32). Delete it to restart.", output, out_size);
        }
        if done >= total_entries {
            println!("Output {} already complete ({} positions). Nothing to do.", output, done);
            return;
        }
        done
    } else {
        0
    };

    let remaining = total_entries - already_done;

    println!("Stockfish rescoring:");
    println!("  Input:     {} ({} positions)", input, total_entries);
    println!("  Output:    {}", output);
    println!("  SF:        {}", sf_path);
    println!("  Threads:   {}", threads);
    println!("  Nodes:     {}", nodes);
    if already_done > 0 {
        println!("  Resuming:  {} done, {} remaining", already_done, remaining);
    }
    println!();

    // Work-stealing model: atomic counter distributes positions sequentially.
    // done_flags[i] tracks whether position i (relative to remaining) is complete.
    // Periodic checkpoint finds longest contiguous prefix and appends to output.
    let next_work = AtomicUsize::new(0);
    let done_flags: Vec<AtomicBool> = (0..remaining).map(|_| AtomicBool::new(false)).collect();
    let done_count = &AtomicU64::new(0);
    let stop = &AtomicBool::new(false);
    let start = Instant::now();

    // Result buffer: copy input data for remaining entries, overwrite scores in-place
    let mut result_buf: Vec<u8> = Vec::with_capacity(remaining * 32);
    result_buf.extend_from_slice(&data[already_done * 32..]);
    let result_ptr = result_buf.as_mut_ptr();

    struct SendPtr(*mut u8);
    unsafe impl Send for SendPtr {}
    unsafe impl Sync for SendPtr {}
    let shared_ptr = SendPtr(result_ptr);

    // Checkpoint state: how many positions have been flushed to disk
    let flushed = AtomicUsize::new(0);

    std::thread::scope(|s| {
        // Worker threads
        for _t in 0..threads as usize {
            let data_ref = &data;
            let next_ref = &next_work;
            let flags_ref = &done_flags;
            let done_ref = done_count;
            let stop_ref = stop;
            let sptr = &shared_ptr;

            s.spawn(move || {
                let mut sf = SfProcess::new(sf_path);

                loop {
                    if stop_ref.load(Ordering::Relaxed) {
                        break;
                    }

                    let i = next_ref.fetch_add(1, Ordering::Relaxed);
                    if i >= remaining {
                        break;
                    }

                    let input_idx = already_done + i;
                    let offset = input_idx * 32;
                    let entry: &[u8; 32] = data_ref[offset..offset + 32].try_into().unwrap();

                    // Skip all-zero entries (corrupted/padding) — mark done without SF eval
                    if entry.iter().all(|&b| b == 0) {
                        flags_ref[i].store(true, Ordering::Release);
                        done_ref.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let (fen, _old_score, _result) = unpack_to_fen(entry);
                    let new_score = sf.eval(&fen, nodes);

                    // Write new score into result buffer
                    unsafe {
                        let ptr = sptr.0.add(i * 32 + 24);
                        let score_bytes = new_score.to_le_bytes();
                        *ptr = score_bytes[0];
                        *ptr.add(1) = score_bytes[1];
                    }

                    flags_ref[i].store(true, Ordering::Release);
                    let count = done_ref.fetch_add(1, Ordering::Relaxed) + 1;

                    if count % 1000 == 0 {
                        let elapsed = start.elapsed().as_secs_f64();
                        let rate = count as f64 / elapsed;
                        let eta = (remaining as u64 - count) as f64 / rate;
                        let eta_h = (eta / 3600.0) as u64;
                        let eta_m = ((eta % 3600.0) / 60.0) as u64;
                        eprint!("\r  {}/{} ({:.1}%) | {:.0} pos/s | ETA {}h {:02}m   ",
                            already_done as u64 + count, total_entries,
                            (already_done as u64 + count) as f64 / total_entries as f64 * 100.0,
                            rate, eta_h, eta_m);
                    }
                }
            });
        }

        // Checkpoint thread: periodically flush contiguous completed prefix to disk
        let flags_ref = &done_flags;
        let flushed_ref = &flushed;
        let stop_ref = stop;
        let done_ref2 = done_count;
        let sptr = &shared_ptr;

        s.spawn(move || {
            let mut last_checkpoint = Instant::now();

            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));

                let all_done = done_ref2.load(Ordering::Relaxed) as usize >= remaining;
                let should_checkpoint = last_checkpoint.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECS
                    || stop_ref.load(Ordering::Relaxed)
                    || all_done;

                if !should_checkpoint {
                    continue;
                }

                // Find contiguous completed prefix starting from flushed position
                let current_flushed = flushed_ref.load(Ordering::Relaxed);
                let mut new_end = current_flushed;
                while new_end < remaining && flags_ref[new_end].load(Ordering::Acquire) {
                    new_end += 1;
                }

                if new_end > current_flushed {
                    // Append newly completed positions to output
                    let bytes_start = current_flushed * 32;
                    let bytes_end = new_end * 32;
                    let slice = unsafe {
                        std::slice::from_raw_parts(sptr.0.add(bytes_start), bytes_end - bytes_start)
                    };

                    use std::fs::OpenOptions;
                    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(output) {
                        if file.write_all(slice).is_ok() {
                            flushed_ref.store(new_end, Ordering::Relaxed);
                            let total_flushed = already_done + new_end;
                            eprintln!("\r  [checkpoint] {}/{} saved ({:.1}%)                    ",
                                total_flushed, total_entries,
                                total_flushed as f64 / total_entries as f64 * 100.0);
                        }
                    }

                    last_checkpoint = Instant::now();
                }

                if all_done || stop_ref.load(Ordering::Relaxed) {
                    break;
                }
            }
        });

    });

    // Final checkpoint: flush any remaining completed positions
    let current_flushed = flushed.load(Ordering::Relaxed);
    let mut final_end = current_flushed;
    while final_end < remaining && done_flags[final_end].load(Ordering::Acquire) {
        final_end += 1;
    }

    if final_end > current_flushed {
        let bytes_start = current_flushed * 32;
        let bytes_end = final_end * 32;
        let slice = &result_buf[bytes_start..bytes_end];

        use std::fs::OpenOptions;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(output)
            .unwrap_or_else(|e| panic!("Failed to open {}: {}", output, e));
        file.write_all(slice)
            .unwrap_or_else(|e| panic!("Failed to write {}: {}", output, e));
    }

    let total_saved = already_done + final_end;
    let completed = done_count.load(Ordering::Relaxed) as usize;
    let elapsed = start.elapsed();
    let rate = if elapsed.as_secs_f64() > 0.0 { completed as f64 / elapsed.as_secs_f64() } else { 0.0 };

    eprintln!();
    println!("  Completed: {} positions in {:.1}s ({:.0} pos/s)", completed, elapsed.as_secs_f64(), rate);
    println!("  Saved:     {}/{} total ({:.1}%)",
        total_saved, total_entries,
        total_saved as f64 / total_entries as f64 * 100.0);
    if total_saved < total_entries {
        println!("  Remaining: {} — restart to continue", total_entries - total_saved);
    } else {
        println!("  COMPLETE!");
    }
}
