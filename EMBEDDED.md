# Embedded Library API

clrsrc ships as **both** a UCI binary (`[[bin]]`) and a Rust library (`[lib]`). The
library exposes an embeddable search facade so a host process — e.g. a Lichess bot —
can run clrsrc **in-process**, handing it the host's authoritative wall-clock as the
search deadline, instead of driving it over the UCI subprocess protocol.

The UCI binary is **unchanged**; the embedded path is purely additive. The reference
consumer is **[LiRu-Bot](https://github.com/liru-bot/liru-bot)** (built with
`--features embedded`).

---

## License — read this before linking clrsrc

clrsrc is **GPL-3.0-or-later**. Linking it into another program produces a combined
work. If you link it into an **AGPL-3.0** host (as LiRu-Bot does):

> Building with the embedded feature links clrsrc (GPL-3.0-or-later) into the host
> (AGPL-3.0-or-later) as a single combined work, permitted by **AGPLv3 §13 / GPLv3 §13**.
> Each part keeps its own license; the clrsrc portion remains GPL-3.0-or-later.
> Operating the combined binary as a **network service** triggers the AGPL §13
> obligation to offer the Corresponding Source of the whole to its users. A host's
> **default (subprocess) build does not link clrsrc** and is unaffected.

When you distribute a binary built against a pinned clrsrc, its Corresponding Source
is clrsrc at that exact tag/commit — pin the dependency accordingly (e.g.
`clrsrc = { git = "https://github.com/clrsrc/clrsrc", tag = "v1.1.0" }`).

---

## Configuration — `EmbeddedConfig`

One-time engine setup. The fields are **1:1 the UCI option keys** of the subprocess
path (`EvalFile`, `Hash`, `SyzygyPath`, …), so embedded and subprocess configure the
engine identically. A bare `EmbeddedConfig::default()` is a valid classical-eval,
bookless engine.

| Field | Type | Default | UCI option | Notes |
|-------|------|---------|------------|-------|
| `eval_file` | `Option<String>` | `None` | `EvalFile` | `None`/empty ⇒ classical eval |
| `hash_mb` | `usize` | `64` | `Hash` | clamped to `[1, 65536]` |
| `syzygy_path` | `Option<String>` | `None` | `SyzygyPath` | `;`-separated dirs |
| `syzygy_probe_limit` | `Option<usize>` | `None` | `SyzygyProbeLimit` | clamped to `min(7)` |
| `syzygy_probe_depth` | `Option<usize>` | `None` | `SyzygyProbeDepth` | clamped to `[1, 100]` |
| `syzygy_50move` | `Option<bool>` | `None` | `Syzygy50MoveRule` | |
| `exp_file` | `Option<String>` | `None` | `ExpFile` | JBK2 experience/opening book |
| `play_from_exp` | `bool` | `false` | `PlayFromExp` | probe the JBK2 book before search |
| `book_variety` | `u8` | `0` | `BookVariety` | 0 = best, 1/2 = wider pool |
| `own_book` | `bool` | `false` | `OwnBook` | probe the Polyglot fallback book |
| `book_file` | `Option<String>` | `None` | `BookFile` | Polyglot book path |
| `best_book_move` | `bool` | `true` | `BestBookMove` | best vs weighted-random |

## Engine — `EmbeddedEngine`

```rust
let engine = clrsrc::EmbeddedEngine::init(config);   // loads NNUE/TT/Syzygy/books once
assert!(engine.nnue_loaded() || config.eval_file.is_none()); // fail-fast, see below
let outcome = engine.search_position(start_fen, &moves, limits, &cancel);
```

- **`init(config) -> EmbeddedEngine`** — loads NNUE / TT / Syzygy / books **once**.
  Global lookup tables are initialised here too (idempotent); you need not call
  anything else first. Hold one engine **per game** so TT/History/NNUE stay warm.
- **`nnue_loaded() -> bool`** — the **fail-fast contract**. If you passed an
  `eval_file` but this returns `false`, refuse to start: a silently unloaded net
  plays classical eval (~−200 Elo) with no other symptom. Check it after `init`.
- **`search_position(start_fen, &[moves], limits, &cancel) -> SearchOutcome`** —
  the engine rebuilds the position **and the repetition history** internally from
  `start_fen` ("startpos" accepted) + the UCI move list; the caller never touches
  `SearchInfo`. It probes the opening book before searching (same precedence/gate as
  the UCI `go` handler).

## Per-move input/output

```rust
pub struct EmbeddedLimits {
    pub tc: TimeControl,        // wtime/btime/winc/binc — clrsrc derives soft/hard from this
    pub max_deadline: Instant,  // ABSOLUTE wall-clock ceiling (see below)
    pub ponder: bool,           // pondering on opponent's clock (no time-stop while true)
    pub game_ply: u32,          // plies from game start — feeds the TC scaling formula
    pub depth: i32,             // fixed-depth override (0 = unlimited)
    pub nodes: u64,             // fixed-node override (0 = unlimited)
}

pub struct SearchOutcome {
    pub best: Move,             // .to_uci()
    pub ponder: Option<Move>,
    pub pv: Vec<Move>,          // .to_uci() per move
    pub score_cp: i32,
    pub depth: i32,
    pub nodes: u64,
    pub mate: Option<i32>,      // mate-in-N (signed; negative = being mated)
}
```

`TimeControl` carries `wtime, btime, winc, binc, movestogo, movetime, depth, infinite,
nodes, soft_nodes`; for the embedded path you fill the raw clocks (the engine zeroes
the rest internally).

### Deadline semantics — the point of the embedded path

The reason to embed rather than spawn a subprocess: **one** authoritative absolute
wall-clock deadline instead of two stacked time approximations (host overhead +
engine overhead). clrsrc keeps its form-aware budgeting (increment amortisation,
stability factors, game-phase logic), but `max_deadline` is the hard ceiling:

```
TimeManager.hard_limit = min(clrsrc_computed_hard, max_deadline − start)
```

Raw clocks pass through untouched; the **single** overhead lives in the gap between
"now" and `max_deadline`. The host computes that deadline from the real remaining
clock minus its measured network round-trip (≈ half-RTT) and a small buffer — the
engine receives only the opaque `Instant` and never reasons about the network. The
engine's own hardcoded 30 ms UCI overhead is dropped on the embedded path.

### Book-hit invariant

A book hit returns `SearchOutcome { best = book move, depth = 0, nodes = 0 }`. The
caller does **not** have to special-case book vs search — this is a guaranteed
invariant.

### `silent` / `print_info` boundary

The engine sets its own output flags: `silent = false` (root TB probe active,
UCI-identical behaviour) and `print_info = false` (no stdout spam). **The caller does
not configure these.**

## Caveats

- **Standard chess only.** clrsrc's `Position::from_fen` does not handle Chess960
  castling; select the embedded path for standard games only.
- **Cancellation** is a `&AtomicBool` bridged to clrsrc's process-global `STOP`,
  which is sufficient for `concurrency: 1`. A live per-search cancel (for running
  multiple in-process games concurrently) is not yet implemented.

## Determinism

For a fixed depth and identical inputs the embedded path returns the **same** move,
score and node count as the UCI binary — verified by `tests/embedded_determinism.rs`
and `tests/embedded_engine.rs` (which also cover the book-hit path and a book-miss
UCI match).
