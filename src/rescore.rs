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
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::time::{Duration, Instant};

const DEFAULT_SF_PATH: &str = "stockfish";
const DEFAULT_NODES: u64 = 25_000;
const DEFAULT_THREADS: u32 = 4;
const CHECKPOINT_INTERVAL_SECS: u64 = 20; // small WUs: short checkpoint, cheap on append (was 60)
const EVAL_TIMEOUT_SECS: u64 = 10;        // per-position SF watchdog: kill+restart+skip on hang

/// Decrements a live-worker counter on drop (any exit path: return/break). Lets the
/// checkpoint thread terminate even when every worker bails without progress (e.g. SF
/// init failure) — previously that path panic-aborted, so the hang never surfaced.
struct ActiveGuard<'a>(&'a AtomicUsize);
impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) { self.0.fetch_sub(1, Ordering::Relaxed); }
}

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

// ---- Stockfish UCI process (crash-isolated, per-eval timeout) ----
//
// A dedicated reader thread pumps SF stdout lines into a channel; the worker reads
// with `recv_timeout`. This makes both failure modes recoverable WITHOUT panicking
// (critical under `panic = "abort"`): a crashed SF closes the channel (-> Dead), a
// hung SF trips the deadline (-> Timeout). The worker kills+restarts SF and skips
// the offending position instead of aborting the whole run.

enum SfErr { Timeout, Dead }

struct SfProcess {
    child: Child,
    stdin: BufWriter<std::process::ChildStdin>,
    rx: Receiver<String>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl SfProcess {
    fn new(sf_path: &str) -> Result<Self, String> {
        let mut child = Command::new(sf_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("Failed to start Stockfish at {}: {}", sf_path, e))?;

        let stdin = BufWriter::new(child.stdin.take().unwrap());
        let stdout = child.stdout.take().unwrap();
        let (tx, rx) = std::sync::mpsc::channel::<String>();

        // Reader thread: stdout lines -> channel. Exits on EOF/err or when the
        // receiver is dropped. Never panics (read errors just end the thread).
        let reader = std::thread::spawn(move || {
            let mut br = BufReader::new(stdout);
            let mut line = String::with_capacity(256);
            loop {
                line.clear();
                match br.read_line(&mut line) {
                    Ok(0) => break,                      // EOF: SF exited
                    Ok(_) => {
                        if tx.send(line.trim_end().to_string()).is_err() {
                            break;                       // receiver gone
                        }
                    }
                    Err(_) => break,                     // read error: SF crashed
                }
            }
        });

        let mut sf = SfProcess { child, stdin, rx, reader: Some(reader) };

        // Handshake with a generous startup timeout.
        sf.send("uci").map_err(|_| "SF stdin closed during init".to_string())?;
        sf.wait_for("uciok", Duration::from_secs(10)).map_err(|_| "SF: no uciok".to_string())?;
        let _ = sf.send("setoption name Hash value 16");
        let _ = sf.send("setoption name Threads value 1");
        sf.send("isready").map_err(|_| "SF stdin closed during init".to_string())?;
        sf.wait_for("readyok", Duration::from_secs(10)).map_err(|_| "SF: no readyok".to_string())?;
        Ok(sf)
    }

    fn send(&mut self, cmd: &str) -> Result<(), ()> {
        writeln!(self.stdin, "{}", cmd).map_err(|_| ())?;
        self.stdin.flush().map_err(|_| ())
    }

    fn wait_for(&mut self, token: &str, timeout: Duration) -> Result<(), ()> {
        let deadline = Instant::now() + timeout;
        loop {
            let now = Instant::now();
            if now >= deadline { return Err(()); }
            match self.rx.recv_timeout(deadline - now) {
                Ok(line) => { if line.starts_with(token) { return Ok(()); } }
                Err(_) => return Err(()),
            }
        }
    }

    /// Evaluate a position. Returns score (cp, STM-relative) or SfErr on timeout/crash.
    fn eval(&mut self, fen: &str, nodes: u64, timeout: Duration) -> Result<i16, SfErr> {
        self.send(&format!("position fen {}", fen)).map_err(|_| SfErr::Dead)?;
        self.send(&format!("go nodes {}", nodes)).map_err(|_| SfErr::Dead)?;

        let deadline = Instant::now() + timeout;
        let mut last_score: i32 = 0;
        loop {
            let now = Instant::now();
            if now >= deadline { return Err(SfErr::Timeout); }
            let line = match self.rx.recv_timeout(deadline - now) {
                Ok(l) => l,
                Err(RecvTimeoutError::Timeout) => return Err(SfErr::Timeout),
                Err(RecvTimeoutError::Disconnected) => return Err(SfErr::Dead),
            };
            if line.starts_with("bestmove") {
                return Ok(last_score.clamp(-32000, 32000) as i16);
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
    }

    fn kill(&mut self) {
        let _ = self.child.kill();
    }
}

impl Drop for SfProcess {
    fn drop(&mut self) {
        let _ = writeln!(self.stdin, "quit");
        let _ = self.stdin.flush();
        // Ensure the child is gone so the reader thread hits EOF and can be joined.
        let _ = self.child.kill();
        let _ = self.child.wait();
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

// ---- TB rescoring: replace eval with Syzygy-perfect labels for endgame positions ----
// Usage: clrsrc tbrescore <input.bin> <output.bin> <syzygy_path> [win_score=2000] [probe_limit=6]
//
// For every position with <= probe_limit pieces, the score field is overwritten with a
// tablebase-perfect value (win = +win_score, draw = 0, loss = -win_score, STM-relative —
// matching the SF rescorer's STM convention). The WDL result byte is PRESERVED (same
// contract as `run`). Positions above the probe limit are copied through UNCHANGED, so a
// TB pass can be layered on top of an SF-rescored / endgame-weighted set: it only sharpens
// the <=6-man labels to ground truth. Single-threaded (the TB probe serialises on a global
// mutex anyway); the endgame subset is small relative to the full corpus.
pub fn run_tb(args: &[String]) {
    let input = args.get(0).expect("Usage: clrsrc tbrescore <input.bin> <output.bin> <syzygy_path> [win_score=2000] [probe_limit=6]");
    let output = args.get(1).expect("Usage: clrsrc tbrescore <input.bin> <output.bin> <syzygy_path> [win_score=2000] [probe_limit=6]");
    let syzygy = args.get(2).expect("syzygy_path required (';'-separated Syzygy dirs)");
    let win_score: i16 = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);
    let probe_limit: usize = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(6);

    crate::tablebase::init(syzygy);
    crate::tablebase::set_probe_limit(probe_limit);
    crate::tablebase::set_50move_rule(false); // theoretical WDL (entries store halfmove=0 anyway)

    let data = std::fs::read(input).unwrap_or_else(|e| panic!("Failed to read {}: {}", input, e));
    if data.len() % 32 != 0 {
        panic!("Input file size {} is not a multiple of 32", data.len());
    }
    let total = data.len() / 32;

    println!("TB rescoring:");
    println!("  Input:        {} ({} positions)", input, total);
    println!("  Output:       {}", output);
    println!("  Syzygy:       {}", syzygy);
    println!("  win_score:    {}cp   probe_limit: {} pieces", win_score, probe_limit);
    println!();

    let mut out = data.clone(); // overwrite scores in place; >limit positions stay unchanged
    let start = Instant::now();
    let (mut tb_win, mut tb_draw, mut tb_loss, mut above_limit, mut not_found, mut empty) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);

    for i in 0..total {
        let off = i * 32;
        let entry: &[u8; 32] = data[off..off + 32].try_into().unwrap();
        if entry.iter().all(|&b| b == 0) { empty += 1; continue; }

        // Piece-count gate from occupancy bitboard — avoids FEN-parse for the >6-man majority.
        let occ = u64::from_le_bytes(entry[0..8].try_into().unwrap());
        if occ.count_ones() as usize > probe_limit { above_limit += 1; continue; }

        let (fen, _old, _res) = unpack_to_fen(entry);
        let pos = match crate::board::Position::from_fen(&fen) {
            Ok(p) => p,
            Err(_) => { not_found += 1; continue; }
        };
        match crate::tablebase::probe_wdl(&pos) {
            Some(wdl) => {
                let score = (wdl as i16) * win_score; // -win_score / 0 / +win_score (STM)
                let sb = score.to_le_bytes();
                out[off + 24] = sb[0];
                out[off + 25] = sb[1];
                match wdl { 1 => tb_win += 1, -1 => tb_loss += 1, _ => tb_draw += 1 }
            }
            None => { not_found += 1; }
        }

        if i % 200_000 == 0 && i > 0 {
            let done = (tb_win + tb_draw + tb_loss) as f64;
            eprint!("\r  {}/{} scanned | TB-relabelled {:.0} | {:.0} pos/s   ",
                i, total, done, i as f64 / start.elapsed().as_secs_f64());
        }
    }

    std::fs::write(output, &out).unwrap_or_else(|e| panic!("Failed to write {}: {}", output, e));
    eprintln!();
    let relabelled = tb_win + tb_draw + tb_loss;
    println!("  TB-relabelled: {} positions (W {} / D {} / L {})", relabelled, tb_win, tb_draw, tb_loss);
    println!("  Unchanged:     {} above {}-piece limit, {} not-found/unparseable, {} empty",
        above_limit, probe_limit, not_found, empty);
    println!("  Time:          {:.1}s", start.elapsed().as_secs_f64());
    println!("  Output:        {} ({} positions, identical layout)", output, total);
}

// ---- Endgame subset filter: keep only positions with <= max_pieces (for endgame weighting) ----
// Usage: clrsrc egfilter <input.bin> <output.bin> [max_pieces=6]
// No tablebase needed — piece count comes straight from the occupancy bitboard. Produces the
// endgame slice that gets TB-relabelled (`tbrescore`) and oversampled into the training mix (lever 1).
pub fn run_egfilter(args: &[String]) {
    let input = args.get(0).expect("Usage: clrsrc egfilter <input.bin> <output.bin> [max_pieces=6]");
    let output = args.get(1).expect("Usage: clrsrc egfilter <input.bin> <output.bin> [max_pieces=6]");
    let max_pieces: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(6);
    // Optional 4th arg "above" inverts the gate: keep positions with > max_pieces
    // (the non-endgame complement). Default "below" keeps <= max_pieces.
    let above = args.get(3).map(|s| s.as_str()) == Some("above");

    let data = std::fs::read(input).unwrap_or_else(|e| panic!("Failed to read {}: {}", input, e));
    if data.len() % 32 != 0 {
        panic!("Input file size {} is not a multiple of 32", data.len());
    }
    let total = data.len() / 32;
    let mut out: Vec<u8> = Vec::new();
    let mut kept = 0u64;
    for i in 0..total {
        let off = i * 32;
        let entry = &data[off..off + 32];
        if entry.iter().all(|&b| b == 0) { continue; }
        let occ = u64::from_le_bytes(entry[0..8].try_into().unwrap());
        let n = occ.count_ones();
        let keep = if above { n > max_pieces } else { n <= max_pieces };
        if keep {
            out.extend_from_slice(entry);
            kept += 1;
        }
    }
    std::fs::write(output, &out).unwrap_or_else(|e| panic!("Failed to write {}: {}", output, e));
    println!("egfilter: {} / {} positions ({} {} pieces) kept -> {} ({:.1}%)",
        kept, total, if above { ">" } else { "<=" }, max_pieces, output, kept as f64 / total.max(1) as f64 * 100.0);
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

    // Check for existing output (resume support).
    // Fix #1 (self-healing): append is NOT atomic — a crash/disk-full mid-`write_all`
    // can leave a dangling sub-entry, making the size not a multiple of 32. Instead of
    // demanding a manual delete (= whole-WU loss), truncate the partial tail to the last
    // 32-byte boundary and resume. Loss is bounded to the single in-flight position.
    let already_done = if std::path::Path::new(output).exists() {
        let out_size = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0);
        let valid = out_size - (out_size % 32);
        if out_size % 32 != 0 {
            eprintln!("  [recover] {} size {} not a multiple of 32 — truncating to {} ({} complete positions) and resuming",
                output, out_size, valid, valid / 32);
            std::fs::OpenOptions::new().write(true).open(output)
                .and_then(|f| f.set_len(valid))
                .unwrap_or_else(|e| panic!("Failed to truncate {} to {}: {}", output, valid, e));
        }
        let done = (valid / 32) as usize;
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
    let skipped = &AtomicU64::new(0); // positions kept at original score after SF hang/crash
    let active = &AtomicUsize::new(threads as usize); // live workers; 0 => checkpoint thread can stop
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
            let skip_ref = skipped;
            let active_ref = active;
            let stop_ref = stop;
            let sptr = &shared_ptr;

            s.spawn(move || {
                let _active = ActiveGuard(active_ref); // decrement live-worker count on any exit
                let timeout = Duration::from_secs(EVAL_TIMEOUT_SECS);
                let mut sf = match SfProcess::new(sf_path) {
                    Ok(s) => s,
                    Err(e) => { eprintln!("\n  [worker] SF init failed: {} — worker exiting (resume will retry)", e); return; }
                };

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

                    // Fix #3/#4: crash-isolated, timeout-bounded eval. On SF hang/crash,
                    // kill+restart SF and retry once; on persistent failure, SKIP — the
                    // position keeps its ORIGINAL score (result_buf is a copy of the input),
                    // so a poison FEN degrades one label instead of aborting the run.
                    let mut new_score: Option<i16> = None;
                    let mut attempts = 0u32;
                    loop {
                        match sf.eval(&fen, nodes, timeout) {
                            Ok(sc) => { new_score = Some(sc); break; }
                            Err(_) => {
                                attempts += 1;
                                sf.kill();
                                match SfProcess::new(sf_path) {
                                    Ok(ns) => sf = ns,
                                    Err(_) => { skip_ref.fetch_add(1, Ordering::Relaxed); break; }
                                }
                                if attempts >= 2 { skip_ref.fetch_add(1, Ordering::Relaxed); break; }
                            }
                        }
                    }

                    // Write new score into result buffer (skip => keep original score)
                    if let Some(sc) = new_score {
                        unsafe {
                            let ptr = sptr.0.add(i * 32 + 24);
                            let score_bytes = sc.to_le_bytes();
                            *ptr = score_bytes[0];
                            *ptr.add(1) = score_bytes[1];
                        }
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
        let active_ref2 = active;
        let sptr = &shared_ptr;

        s.spawn(move || {
            let mut last_checkpoint = Instant::now();

            loop {
                std::thread::sleep(std::time::Duration::from_secs(5));

                let all_done = done_ref2.load(Ordering::Relaxed) as usize >= remaining;
                let workers_done = active_ref2.load(Ordering::Relaxed) == 0; // every worker exited
                let should_checkpoint = last_checkpoint.elapsed().as_secs() >= CHECKPOINT_INTERVAL_SECS
                    || stop_ref.load(Ordering::Relaxed)
                    || all_done
                    || workers_done;

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

                if all_done || workers_done || stop_ref.load(Ordering::Relaxed) {
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

    let skip_n = skipped.load(Ordering::Relaxed);
    eprintln!();
    println!("  Completed: {} positions in {:.1}s ({:.0} pos/s)", completed, elapsed.as_secs_f64(), rate);
    if skip_n > 0 {
        println!("  Skipped:   {} positions kept at original score (SF hang/crash)", skip_n);
    }
    println!("  Saved:     {}/{} total ({:.1}%)",
        total_saved, total_entries,
        total_saved as f64 / total_entries as f64 * 100.0);
    if total_saved < total_entries {
        println!("  Remaining: {} — restart to continue", total_entries - total_saved);
    } else {
        println!("  COMPLETE!");
    }
}
