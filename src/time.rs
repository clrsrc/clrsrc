// ---- Time Management ----
// Phase-aware with stability-based scaling.

use std::time::Instant;
use std::sync::atomic::{AtomicBool, Ordering};

pub static STOP: AtomicBool = AtomicBool::new(false);
pub static PONDERING: AtomicBool = AtomicBool::new(false);
/// Whether the GUI has enabled pondering via UCI `setoption name Ponder value <bool>`.
/// When false: engine ignores `go ponder` commands (treats as normal `go`) and does NOT
/// append "ponder X" to its `bestmove` output. Default true (standard UCI behavior).
static ALLOW_PONDER: AtomicBool = AtomicBool::new(true);
/// Signals TimeManager to reset its clock on ponderhit
static PONDERHIT_RESET: AtomicBool = AtomicBool::new(false);

pub fn signal_ponderhit() {
    PONDERHIT_RESET.store(true, Ordering::Relaxed);
    PONDERING.store(false, Ordering::Relaxed);
}

pub fn check_ponderhit_reset() -> bool {
    PONDERHIT_RESET.compare_exchange(true, false, Ordering::Relaxed, Ordering::Relaxed).is_ok()
}

pub fn should_stop() -> bool {
    STOP.load(Ordering::Relaxed)
}

pub fn set_stop(val: bool) {
    STOP.store(val, Ordering::Relaxed);
}

pub fn is_pondering() -> bool {
    PONDERING.load(Ordering::Relaxed)
}

pub fn set_pondering(val: bool) {
    PONDERING.store(val, Ordering::Relaxed);
}

pub fn allow_ponder() -> bool {
    ALLOW_PONDER.load(Ordering::Relaxed)
}

pub fn set_allow_ponder(val: bool) {
    ALLOW_PONDER.store(val, Ordering::Relaxed);
}

/// Time control limits parsed from UCI "go" command
pub struct TimeControl {
    pub wtime: i64,       // ms
    pub btime: i64,
    pub winc: i64,
    pub binc: i64,
    pub movestogo: u32,
    pub movetime: i64,    // fixed time per move
    pub depth: i32,       // fixed depth
    pub infinite: bool,
    pub nodes: u64,       // hard node count (0 = unlimited)
    pub soft_nodes: u64,  // soft node limit — stop at ID boundary (0 = unlimited)
}

impl TimeControl {
    pub fn infinite() -> Self {
        TimeControl {
            wtime: 0, btime: 0, winc: 0, binc: 0,
            movestogo: 0, movetime: 0, depth: 0,
            infinite: true, nodes: 0, soft_nodes: 0,
        }
    }
}

pub struct TimeManager {
    start: Instant,
    base_soft_limit_ms: i64,  // original soft limit (before factor scaling)
    base_hard_limit_ms: i64,  // original hard limit (post patch-A-v2 cap) — upper bound for shrink-only hard scaling
    soft_limit_ms: i64,
    hard_limit_ms: i64,
    infinite: bool,
    max_depth: i32,
    max_nodes: u64,
    soft_nodes: u64,      // soft node limit (stop at ID boundary)
    stable_count: u32,
    last_best_move: u16,
    prev_score: i32,
    stability_factor: f64,  // Stash table [2.50,1.20,0.90,0.80,0.75] indexed by min(stable_count,4)
    score_factor: f64,      // Stash 2^(-Δscore/100), per-iter (NOT accumulated)
    node_tm_factor: f64,    // node-based factor (existing)
    forced_factor: f64,     // Patch B Phase 2: 1 legal=0.01, strong-forced=0.39, weak-forced=0.63, else 1.0
}

impl TimeManager {
    pub fn new(tc: &TimeControl, is_white: bool, game_ply: u32) -> Self {
        let mut tm = TimeManager {
            start: Instant::now(),
            base_soft_limit_ms: i64::MAX,
            base_hard_limit_ms: i64::MAX,
            soft_limit_ms: i64::MAX,
            hard_limit_ms: i64::MAX,
            infinite: tc.infinite,
            max_depth: if tc.depth > 0 { tc.depth } else { 100 },
            max_nodes: tc.nodes,
            soft_nodes: tc.soft_nodes,
            stable_count: 0,
            last_best_move: 0,
            prev_score: 0,
            stability_factor: 1.0,
            score_factor: 1.0,
            node_tm_factor: 1.0,
            forced_factor: 1.0,
        };

        if tc.infinite || tc.depth > 0 {
            return tm;
        }

        if tc.movetime > 0 {
            tm.soft_limit_ms = tc.movetime;
            tm.hard_limit_ms = tc.movetime;
            return tm;
        }

        let time = if is_white { tc.wtime } else { tc.btime };
        let inc = if is_white { tc.winc } else { tc.binc };

        if time <= 0 {
            tm.soft_limit_ms = 100;
            tm.hard_limit_ms = 100;
            return tm;
        }

        // Stockfish-style time manager init (Modus B port from chess_engines/top/stockfish/src/src/timeman.cpp).
        // Log-scaled constants → caps grow with log10(time): tight at TC 10+0.1, generous at TC 60+0.6 / Lichess-Rapid.
        // Move overhead: 30 ms safety buffer (hardcoded; future: UCI MoveOverhead option).
        let move_overhead: f64 = 30.0;

        // Move horizon: ≤50 moves until next TC. Below 1 s scale mtg down so timeLeft stays positive.
        let mut mtg: i64 = if tc.movestogo > 0 {
            (tc.movestogo as i64).min(50)
        } else {
            50
        };
        if time < 1000 {
            mtg = ((time as f64) * 0.05) as i64;
            if mtg < 1 { mtg = 1; }
        }

        // Effective time budget across remaining horizon (incl. inc, minus per-move overhead).
        let time_left = (time as f64 + (inc as f64) * (mtg - 1) as f64
                         - move_overhead * (mtg + 2) as f64).max(1.0);

        let (optimum, maximum): (f64, f64);

        if tc.movestogo == 0 {
            // ----- x basetime (+ inc) mode -----
            let log_time = (time as f64 / 1000.0).log10();
            let opt_const = (0.0029869 + 0.00033554 * log_time).min(0.004905);
            let max_const = (3.3744 + 3.0608 * log_time).max(3.1441);

            // TC-adaptive adjust: more time per move when total game time is bigger.
            let time_adjust = (0.3272 * (time_left).log10() - 0.4141).max(0.0);

            let opt_scale = ((0.012112 + (game_ply as f64 + 3.22713).powf(0.46866) * opt_const)
                             .min(0.19404 * time as f64 / time_left))
                            * time_adjust;
            let max_scale = (max_const + game_ply as f64 / 12.352).min(6.873);

            optimum = (opt_scale * time_left).max(1.0);
            maximum = ((0.8097 * time as f64 - move_overhead).min(max_scale * optimum)).max(optimum);
        } else {
            // ----- x moves in y seconds (+ inc) mode -----
            let opt_scale = ((0.88 + game_ply as f64 / 116.4) / mtg as f64)
                            .min(0.88 * time as f64 / time_left);
            let max_scale = 1.3 + 0.11 * mtg as f64;

            optimum = (opt_scale * time_left).max(1.0);
            maximum = ((0.8097 * time as f64 - move_overhead).min(max_scale * optimum)).max(optimum);
        }

        tm.soft_limit_ms = optimum as i64;
        tm.hard_limit_ms = maximum as i64;

        // TC-adaptive per-move cap (Patch A v2, 19.05): Stockfish-Modus-B caps
        // `maximum` via max_scale.min(6.873) which at Blitz 5+2 mid-game allowed
        // up to 51% of remaining time on one move (Lichess game WEWtf22r M19:
        // hard_limit 72s on 141s remaining → engine spent ~70s, mate at M61).
        //
        // v1 (initial 19.05) applied a `log10(time)`-scaled cap at all TCs.
        // SPRT 351g @ Blitz 3+2 = −23.9 ±14.4 ELO LOS 0.1%: regressed badly at
        // mid/late Blitz where remaining time < 60s and the cap cut natural
        // think-time by 50-80% (factor 0.15 too tight, no inc-amortization).
        //
        // v2 only activates above 60 s remaining — preserves Blitz/Bullet
        // behaviour (natural max already tight there) while keeping the
        // 35 s cap for the 141 s WEWtf22r-pattern and longer TCs.
        //   time =  30s → no cap (matches B6 behaviour)
        //   time =  60s → no cap (boundary)
        //   time = 141s → 25% (cap = 35s, was 127s)
        //   time = 300s → 28% (cap = 84s)
        //   time = 3600s → 35% (cap = 1260s, natural < cap → no effect)
        if time > 60_000 {
            let log_time = (time as f64 / 1000.0).log10().max(0.5);
            let per_move_cap = (0.10 + 0.07 * log_time).max(0.15).min(0.40);
            tm.hard_limit_ms = tm.hard_limit_ms.min((time as f64 * per_move_cap) as i64);
            tm.soft_limit_ms = tm.soft_limit_ms.min(tm.hard_limit_ms);
        }

        // Capture bases for Patch B factor recomputation (post patch-A-v2 cap).
        tm.base_soft_limit_ms = tm.soft_limit_ms;
        tm.base_hard_limit_ms = tm.hard_limit_ms;

        tm
    }

    pub fn elapsed_ms(&self) -> i64 {
        self.start.elapsed().as_millis() as i64
    }

    /// Should we start a new iteration?
    pub fn can_start_iteration(&self) -> bool {
        if is_pondering() { return true; }
        if self.infinite || self.soft_limit_ms == i64::MAX { return true; }
        // Don't start if we've used > 60% of soft limit
        self.elapsed_ms() < self.soft_limit_ms * 60 / 100
    }

    /// Should we stop searching NOW? (checked during search)
    pub fn should_stop_hard(&mut self, nodes: u64) -> bool {
        // On ponderhit, reset the clock so time starts from now
        if check_ponderhit_reset() {
            self.start = Instant::now();
        }
        if is_pondering() { return false; }
        if self.max_nodes > 0 && nodes >= self.max_nodes {
            return true;
        }
        if self.infinite || self.hard_limit_ms == i64::MAX { return false; }
        self.elapsed_ms() >= self.hard_limit_ms
    }

    /// Apply combined stability × score × node × forced factor to soft/hard limits.
    /// hard can only SHRINK from base (preserves patch-A-v2 cap as upper bound).
    /// soft is recomputed from base × total, then clamped to hard.
    fn apply_factors(&mut self) {
        if self.infinite || self.base_hard_limit_ms == i64::MAX { return; }
        let total = self.stability_factor * self.score_factor * self.node_tm_factor * self.forced_factor;
        let scaled_hard = (self.base_hard_limit_ms as f64 * total) as i64;
        self.hard_limit_ms = scaled_hard.min(self.base_hard_limit_ms).max(1);
        let scaled_soft = (self.base_soft_limit_ms as f64 * total) as i64;
        self.soft_limit_ms = scaled_soft.min(self.hard_limit_ms).max(1);
    }

    /// Update forced-move factor after each ID iteration.
    /// Patch B Phase 2 (20.05.2026): Stash/Viridithas forced-move detection.
    /// root_moves_count = 1 (only legal move) → 0.01× (essentially instant).
    /// eval_diff (best - second_best, in centipawns):
    ///   strong-forced (>400cp = ~4 pawns) → 0.39× (very dominant move)
    ///   weak-forced (>170cp = ~1.7 pawns) → 0.63× (clear best)
    ///   else → 1.0× (no shrink)
    /// Combined multiplicatively with stability × score × node factors.
    pub fn update_forced(&mut self, root_moves_count: u32, eval_diff_cp: i32) {
        if self.infinite || self.base_hard_limit_ms == i64::MAX { return; }
        self.forced_factor = if root_moves_count == 1 {
            0.01
        } else if eval_diff_cp > 400 {
            0.39
        } else if eval_diff_cp > 170 {
            0.63
        } else {
            1.0
        };
        self.apply_factors();
    }

    /// Update stability tracking after each ID iteration.
    /// Patch B (20.05.2026): Stash/Viridithas table indexed by min(stable_count, 4).
    /// stable_count = 0 (BM just changed) → 2.50× more time.
    /// stable_count ≥ 4 (5 consecutive iters same BM) → 0.75× less time.
    /// Replaces previous heuristic [1.15, 1.0, 0.8, 0.6] which didn't shrink hard_limit
    /// (root cause of WEWtf22r-pattern mid-game time-dumps).
    pub fn update_best_move(&mut self, best_move: u16) {
        if best_move == self.last_best_move {
            self.stable_count += 1;
        } else {
            self.stable_count = 0;
            self.last_best_move = best_move;
        }
        const STAB_TABLE: [f64; 5] = [2.50, 1.20, 0.90, 0.80, 0.75];
        let idx = (self.stable_count as usize).min(4);
        self.stability_factor = STAB_TABLE[idx];
        self.apply_factors();
    }

    /// Update score-progression factor after each ID iteration.
    /// Patch B (20.05.2026): Stash `timeman_scale_score_diff` formula —
    /// score_factor = 2^(-Δscore / 100), clamped Δ ∈ [-100, 100].
    /// Δscore = score_now - score_prev_iter.
    /// score falls 100cp → 2.00× (search longer, look for rescue).
    /// score rises 100cp → 0.50× (search shorter, safe enough).
    /// NOT accumulated across iterations — replaces previous additive score_drop_factor.
    pub fn update_score(&mut self, score: i32, depth: i32) {
        if depth <= 4 || self.infinite || self.base_hard_limit_ms == i64::MAX {
            self.prev_score = score;
            return;
        }
        let delta = (score - self.prev_score).clamp(-100, 100) as f64;
        self.score_factor = 2.0_f64.powf(-delta / 100.0);
        self.prev_score = score;
        self.apply_factors();
    }

    /// Update node-based time management factor (v02 form — stepped heuristic).
    /// Viridithas continuous formula (Phase 3) was tested 20.05 and REJECTED at Blitz
    /// (−10.4 ±15.9 ELO LOS 10%, 400g). Extender-bias eats clock budget without ELO return.
    /// Possible future Phase 3b: best_move_nodes-fix isolated, or dampened Viridithas.
    pub fn update_node_fraction(&mut self, best_move_nodes: u64, total_nodes: u64) {
        if total_nodes == 0 || self.infinite || self.base_hard_limit_ms == i64::MAX {
            return;
        }
        let fraction = best_move_nodes as f64 / total_nodes as f64;
        self.node_tm_factor = if fraction > 0.90 { 0.5 }
            else if fraction > 0.80 { 0.6 }
            else if fraction > 0.60 { 0.8 }
            else if fraction < 0.25 { 1.5 }
            else if fraction < 0.35 { 1.2 }
            else { 1.0 };
        self.apply_factors();
    }

    pub fn max_depth(&self) -> i32 {
        self.max_depth
    }

    /// Should we stop starting new iterations? (soft node limit reached)
    pub fn should_stop_soft(&self, nodes: u64) -> bool {
        self.soft_nodes > 0 && nodes >= self.soft_nodes
    }
}
