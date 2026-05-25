// ---- Tunable Search Parameters ----
// Adjustable via UCI "setoption name X value Y" for SPSA tuning.
// All values are stored in a global struct protected by atomic operations.

use std::sync::atomic::{AtomicI32, Ordering};

macro_rules! tunable {
    ($name:ident, $default:expr, $min:expr, $max:expr) => {
        pub static $name: AtomicI32 = AtomicI32::new($default);
    };
}

// RFP (Reverse Futility Pruning)
tunable!(RFP_MARGIN_IMP, 80, 30, 120);    // SPSA v1: 70→80
tunable!(RFP_MARGIN_NIMP, 58, 20, 100);   // SPSA v1: 50→58
tunable!(RFP_DEPTH, 7, 4, 12);            // SPSA v2: 8→7 (-52.5 ELO, LOS 99.4%)

// NMP (Null Move Pruning)
tunable!(NMP_BASE, 3, 2, 5);
tunable!(NMP_DEPTH_DIV, 6, 3, 12);
tunable!(NMP_EVAL_DIV, 230, 100, 400);     // SPSA v2: 200→230 (+29 ELO, LOS 93.4%)
tunable!(NMP_EVAL_MAX, 3, 1, 6);

// Razoring
tunable!(RAZOR_MARGIN, 210, 100, 400);   // SPSA v2: 240→210 (-40.7 ELO, LOS 95.5%)
tunable!(RAZOR_DEPTH, 3, 1, 5);

// Futility Pruning
tunable!(FP_MARGIN_MUL, 68, 40, 150);     // SPSA v2: 80→68 (-23.2 ELO)
tunable!(FP_MARGIN_ADD, 115, 50, 200);    // SPSA v2: 100→115 (+17.4 ELO)
tunable!(FP_DEPTH, 8, 4, 12);

// LMP (Late Move Pruning)
tunable!(LMP_BASE, 3, 1, 6);

// History Pruning
tunable!(HIST_PRUNE_DEPTH, 5, 2, 8);
tunable!(HIST_PRUNE_MARGIN, 1024, 256, 4096);

// SEE Pruning
tunable!(SEE_DEPTH, 8, 4, 12);
tunable!(SEE_QUIET_MUL, 42, 20, 100);     // SPSA v2: 50→42 (-29.0 ELO)
tunable!(SEE_CAP_MUL, 100, 50, 200);

// LMR
tunable!(LMR_HIST_DIV, 5120, 2048, 10240);
tunable!(LMR_DIVISOR, 175, 125, 300);     // scaled x100; base divisor in ln(d)*ln(m)/div formula

// Aspiration Window
tunable!(ASP_DELTA, 21, 10, 50);          // SPSA v2: 25→21 (-34.9 ELO)

// ProbCut
tunable!(PROBCUT_MARGIN, 80, 50, 200);   // SPSA: 100→80
tunable!(PROBCUT_DEPTH, 5, 3, 8);

// Singular Extension
tunable!(SE_DEPTH, 8, 5, 12);
tunable!(SE_MARGIN_MUL, 3, 1, 4);         // SPSA v2: 2→3 (+17.4 ELO, LOS 81.7%)

#[inline]
pub fn get(param: &AtomicI32) -> i32 {
    param.load(Ordering::Relaxed)
}

pub fn set(param: &AtomicI32, val: i32) {
    param.store(val, Ordering::Relaxed);
}

/// All tunable parameters with their names (for UCI and SPSA)
pub struct TuneParam {
    pub name: &'static str,
    pub param: &'static AtomicI32,
    pub default: i32,
    pub min: i32,
    pub max: i32,
}

pub fn all_params() -> Vec<TuneParam> {
    vec![
        TuneParam { name: "RFP_MARGIN_IMP", param: &RFP_MARGIN_IMP, default: 80, min: 30, max: 120 },
        TuneParam { name: "RFP_MARGIN_NIMP", param: &RFP_MARGIN_NIMP, default: 58, min: 20, max: 100 },
        TuneParam { name: "RFP_DEPTH", param: &RFP_DEPTH, default: 7, min: 4, max: 12 },
        TuneParam { name: "NMP_BASE", param: &NMP_BASE, default: 3, min: 2, max: 5 },
        TuneParam { name: "NMP_DEPTH_DIV", param: &NMP_DEPTH_DIV, default: 6, min: 3, max: 12 },
        TuneParam { name: "NMP_EVAL_DIV", param: &NMP_EVAL_DIV, default: 230, min: 100, max: 400 },
        TuneParam { name: "NMP_EVAL_MAX", param: &NMP_EVAL_MAX, default: 3, min: 1, max: 6 },
        TuneParam { name: "RAZOR_MARGIN", param: &RAZOR_MARGIN, default: 210, min: 100, max: 400 },
        TuneParam { name: "RAZOR_DEPTH", param: &RAZOR_DEPTH, default: 3, min: 1, max: 5 },
        TuneParam { name: "FP_MARGIN_MUL", param: &FP_MARGIN_MUL, default: 68, min: 40, max: 150 },
        TuneParam { name: "FP_MARGIN_ADD", param: &FP_MARGIN_ADD, default: 115, min: 50, max: 200 },
        TuneParam { name: "FP_DEPTH", param: &FP_DEPTH, default: 8, min: 4, max: 12 },
        TuneParam { name: "LMP_BASE", param: &LMP_BASE, default: 3, min: 1, max: 6 },
        TuneParam { name: "HIST_PRUNE_DEPTH", param: &HIST_PRUNE_DEPTH, default: 5, min: 2, max: 8 },
        TuneParam { name: "HIST_PRUNE_MARGIN", param: &HIST_PRUNE_MARGIN, default: 1024, min: 256, max: 4096 },
        TuneParam { name: "SEE_DEPTH", param: &SEE_DEPTH, default: 8, min: 4, max: 12 },
        TuneParam { name: "SEE_QUIET_MUL", param: &SEE_QUIET_MUL, default: 42, min: 20, max: 100 },
        TuneParam { name: "SEE_CAP_MUL", param: &SEE_CAP_MUL, default: 100, min: 50, max: 200 },
        TuneParam { name: "LMR_HIST_DIV", param: &LMR_HIST_DIV, default: 5120, min: 2048, max: 10240 },
        TuneParam { name: "LMR_DIVISOR", param: &LMR_DIVISOR, default: 175, min: 125, max: 300 },
        TuneParam { name: "ASP_DELTA", param: &ASP_DELTA, default: 21, min: 10, max: 50 },
        TuneParam { name: "PROBCUT_MARGIN", param: &PROBCUT_MARGIN, default: 80, min: 50, max: 200 },
        TuneParam { name: "PROBCUT_DEPTH", param: &PROBCUT_DEPTH, default: 5, min: 3, max: 8 },
        TuneParam { name: "SE_DEPTH", param: &SE_DEPTH, default: 8, min: 5, max: 12 },
        TuneParam { name: "SE_MARGIN_MUL", param: &SE_MARGIN_MUL, default: 3, min: 1, max: 4 },
    ]
}
