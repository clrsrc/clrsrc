// ---- Time Management ----
// Phase-aware with stability-based scaling.

use std::time::Instant;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

pub static STOP: AtomicBool = AtomicBool::new(false);
pub static PONDERING: AtomicBool = AtomicBool::new(false);
/// Whether the GUI has enabled pondering via UCI `setoption name Ponder value <bool>`.
/// When false: engine ignores `go ponder` commands (treats as normal `go`) and does NOT
/// append "ponder X" to its `bestmove` output. Default true (standard UCI behavior).
static ALLOW_PONDER: AtomicBool = AtomicBool::new(true);

/// Initial base time (ms) of the current game, captured at ply ≤ 1 (≈ full clock).
/// 0 = not yet captured this game. Used solely to detect a genuine *bullet game*
/// so the bullet-aware `time_adjust` floor is NOT applied to blitz/rapid low-time
/// endgames (which share the same low *remaining* time but are SPRT-validated).
/// Re-captured every game because ply resets to 0/1 at game start (UCI + embedded).
static GAME_BASE_MS: AtomicI64 = AtomicI64::new(0);

/// Bullet-aware floor on the per-move `time_adjust` term (see build()). The legacy
/// `time_adjust = 0.3272·log10(time_left) − 0.4141` collapses toward ~0 at short TC
/// (≈0.07 at 30 s, ≈0.17 at 60 s vs ≈0.32 at blitz 180 s), starving bullet of search
/// depth: clrsrc banks the clock / increment and is mated on the board with 50–105 %
/// of its clock unused (Lichess bullet ~124 Elo < blitz, 71/71 board losses, 0 flags,
/// sub-peer losses to −200 Elo). Flooring `time_adjust` to a blitz-like level makes
/// bullet spend a healthy fraction of its clock → deeper search. Applied ONLY when the
/// captured base time classifies the game as bullet. 0.0 = legacy behaviour.
/// SPRT-tune in [0.35, 0.70]; MUST be confirmed non-regressing at blitz before deploy.
/// 0.0 here (validated baseline) for the public v1.2.0 cut — the unvalidated A/B value is
/// NOT shipped publicly. A LIVE bullet A/B at 0.50 runs independently on the VServer bot
/// (separate embedded binary, since 22.06; self-play-unvalidatable, so only the live run can
/// decide; data-starved → re-check ~04.07). If that A/B validates, set 0.50 + rebuild + a
/// follow-up release. Blitz/rapid/bench provably inert via the is_bullet gate either way
/// (game_base + 40·inc ≤ 179 s); bench unchanged (1527458).
const BULLET_TA_FLOOR: f64 = 0.0;

pub fn signal_ponderhit() {
    // End pondering. We deliberately do NOT reset the search clock: the time already spent
    // pondering (on the opponent's clock) counts against this move's budget, so the soft/hard
    // limits — measured from `go ponder` — prevent over-allocating our own clock after the hit.
    // Resetting here (old behavior) granted a full fresh budget post-hit while the search was
    // already deep, so one giant iteration blew the clock → ponderhit time-forfeits (fixed 27.05).
    PONDERING.store(false, Ordering::Relaxed);
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
    max_deadline: Option<Instant>, // C2 (embedded): absolute wall-clock ceiling from the bot.
                            // None for the UCI path (behaviour unchanged). When Some, it caps the
                            // computed limits AND is enforced directly in should_stop_hard().
}

impl TimeManager {
    /// UCI path — unchanged behaviour (30 ms hardcoded overhead, no deadline).
    pub fn new(tc: &TimeControl, is_white: bool, game_ply: u32) -> Self {
        Self::build(tc, is_white, game_ply, 30.0, None)
    }

    /// Embedded path (C2): same form-aware budgeting as `new()`, but the bot's
    /// absolute wall-clock `max_deadline` is the authoritative hard ceiling, and
    /// the engine's 30 ms overhead is dropped — the bot already subtracted RTT +
    /// move_overhead when computing the deadline (single source of truth).
    #[allow(dead_code)] // used by the embedded lib facade (lib.rs), not the UCI binary
    pub fn with_deadline(tc: &TimeControl, is_white: bool, game_ply: u32, max_deadline: Instant) -> Self {
        Self::build(tc, is_white, game_ply, 0.0, Some(max_deadline))
    }

    fn build(tc: &TimeControl, is_white: bool, game_ply: u32, move_overhead_in: f64, max_deadline: Option<Instant>) -> Self {
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
            max_deadline,
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

        // Bullet-aware TM (21.06): capture the game's initial base time at ply ≤ 1
        // (clock still ≈ full) so we can lift the per-move floor ONLY for genuine
        // bullet games. A fresh game always re-enters ply 0/1 → self-resetting.
        if game_ply <= 1 && time > 0 {
            GAME_BASE_MS.store(time, Ordering::Relaxed);
        }
        let game_base = GAME_BASE_MS.load(Ordering::Relaxed);
        // Lichess speed classifier: estimated game duration = base + 40·inc.
        // Bullet (incl. ultrabullet) < 180 s → boost; blitz (≥ 180 s) untouched.
        let is_bullet = game_base > 0 && game_base + 40 * inc <= 179_000;

        if time <= 0 {
            tm.soft_limit_ms = 100;
            tm.hard_limit_ms = 100;
            return tm;
        }

        // Stockfish-style time manager init (Modus B port from chess_engines/top/stockfish/src/src/timeman.cpp).
        // Log-scaled constants → caps grow with log10(time): tight at TC 10+0.1, generous at TC 60+0.6 / Lichess-Rapid.
        // Move overhead: UCI path passes 30 ms safety buffer; embedded path passes 0
        // (the bot's deadline already accounts for RTT + overhead).
        let move_overhead: f64 = move_overhead_in;

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
            // Bullet-aware floor (21.06): in genuine bullet games this term otherwise
            // collapses (~0.07 at 30 s) and starves search → board losses with the clock
            // unused. Floor it to a blitz-like level so bullet actually spends its time.
            // Blitz/rapid (incl. low-time endgames) keep floor 0.0 = legacy tuning.
            let ta_floor = if is_bullet { BULLET_TA_FLOOR } else { 0.0 };
            let time_adjust = (0.3272 * (time_left).log10() - 0.4141).max(ta_floor);

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

        // Low-time cap (Patch C, 28.05): below 60 s remaining the natural max still
        // lets the engine spend > 50 % of the clock on one move (game AgHuraju vs
        // matmoi 2013, 3+2: 28 s of a 47 s budget → outoftime).
        // Sister regime of patch-A-v2 (cap above 60 s).
        // Cap shape: `time * 0.15 + inc`. The `+ inc` term fixes v1's (19.05)
        // SPRT-reject reason "factor 0.15 too tight, no inc-amortization":
        //   time =  5s, inc=2s →  2.75 s cap
        //   time = 30s, inc=2s →  6.50 s cap
        //   time = 60s, inc=2s → 11.00 s cap
        // REQUIRES SPRT before VServer-deploy: v1 full-range 0.15 lost
        // −23.9 ELO at Blitz 3+2. If Blitz mid-game regresses, narrow the
        // threshold (e.g. time ≤ 30 000) or raise the factor (0.20–0.25).
        if time > 0 && time <= 60_000 {
            let low_time_cap = ((time as f64) * 0.15 + (inc as f64)) as i64;
            tm.hard_limit_ms = tm.hard_limit_ms.min(low_time_cap.max(50));
            tm.soft_limit_ms = tm.soft_limit_ms.min(tm.hard_limit_ms);
        }

        // Capture bases for Patch B factor recomputation (post patch-A-v2 cap).
        tm.base_soft_limit_ms = tm.soft_limit_ms;
        tm.base_hard_limit_ms = tm.hard_limit_ms;

        // C2 (embedded): cap the form-aware budget by the bot's absolute deadline.
        // "min(clrsrc_computed_hard, max_deadline - start)" — clrsrc keeps inc-amortisation,
        // stability & phase logic, but the bot's knowledge of the real remaining time is the
        // upper bound. Cap the BASE so the factor system (apply_factors) never exceeds it.
        if let Some(dl) = tm.max_deadline {
            let budget = dl.saturating_duration_since(tm.start).as_millis() as i64;
            let cap = budget.max(1);
            tm.hard_limit_ms = tm.hard_limit_ms.min(cap);
            tm.soft_limit_ms = tm.soft_limit_ms.min(tm.hard_limit_ms);
            tm.base_hard_limit_ms = tm.hard_limit_ms;
            tm.base_soft_limit_ms = tm.soft_limit_ms;
        }

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
        // While pondering, never stop on time (it's the opponent's clock). After ponderhit,
        // `is_pondering` is false and the clock keeps running from `go ponder`, so ponder time
        // counts against the budget — total wall time is bounded by hard_limit, never over-allocating.
        if is_pondering() { return false; }
        // C2 (embedded): absolute wall-clock ceiling. Enforced directly so it also bounds
        // infinite/depth-mode embedded searches, independent of the computed hard_limit_ms.
        if let Some(dl) = self.max_deadline {
            if Instant::now() >= dl { return true; }
        }
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
        // Cap how far the SOFT limit (the "start a new iteration" gate) may inflate above its
        // base. stability_factor (up to 2.50× while the best move keeps changing) is meant to
        // grant more time when a *better* move was just found — but in dead-flat positions
        // (eval ≈ 0.0, many near-equal root moves) the best move oscillates as noise, pinning
        // stability at 2.50× and ballooning soft from ~40s toward the ~158s hard cap. On slower
        // hardware the engine is still in that oscillation phase when the inflated budget lets it
        // launch one more very deep iteration → 126s on a single quiet move (Lichess EqGn5ie6,
        // Rapid 600+5). HARD is left untouched (a running iteration may finish); only the
        // start-new-iteration budget is bounded. Invisible at fast TC / fast HW (the BM
        // stabilises before the budget binds), so this is an SPRT-at-Rapid-TC change. Tunable.
        // Flat-Defense-Fix v1 (17.06, bot #186 + chess_engines #48): 1.50 deckelte die
        // Kritikalitäts-Eskalation (score_factor bis 2.0× bei Eval-Kippen, stability bis 2.5×
        // bei BM-Wechsel) auf 1.5× base_soft → defensiv-schlechte Stellungen bekamen flach zu
        // wenig Zeit (kein fail-low-Extend sichtbar). 1.50→2.50 lässt den Verteidigungs-Extend
        // atmen; der Forfeit-Schutz bleibt orthogonal (bot-max_deadline-Polling + hard-Clamp),
        // analog Berserk/Viri-Vorbild (soft atmet, separate Hard-Decke fängt Runaway).
        const SOFT_INFLATION_CAP: f64 = 2.50;
        let soft_ceiling = (self.base_soft_limit_ms as f64 * SOFT_INFLATION_CAP) as i64;
        let scaled_soft = (self.base_soft_limit_ms as f64 * total) as i64;
        self.soft_limit_ms = scaled_soft.min(soft_ceiling).min(self.hard_limit_ms).max(1);
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
