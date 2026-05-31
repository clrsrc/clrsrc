# Changelog

All notable changes to clrsrc are documented here.

## [1.1.0] - 2026-05-31

### Added
- **Embedded library API (`[lib]` target).** clrsrc now builds as both the UCI binary and a
  Rust library, exposing an in-process search facade so a host (e.g. a Lichess bot) can run
  clrsrc without the UCI subprocess protocol and hand it the host's authoritative wall-clock as
  the search deadline. New: `EmbeddedEngine` (`init`, `nnue_loaded`, `search_position`),
  `EmbeddedConfig`, `EmbeddedLimits`, `SearchOutcome`, and `TimeManager::with_deadline`. The
  UCI binary and its behaviour are **unchanged** — the embedded path is purely additive. See
  [`EMBEDDED.md`](EMBEDDED.md). Reference consumer: the AGPL-3.0 LiRu-Bot (`--features embedded`).

### Performance
- **Incremental NNUE update fusion.** `update_inc` now folds `parent − removed (+ removed2) + added`
  into a single pass (`vec_add_sub` / `vec_add_sub_sub`) instead of copy-then-subtract-then-add.
  Bit-identical evaluation (i16-wrapping is associative), so the `bench` node count is unchanged
  (2,846,610) — this is free search speed (~+6–9% NPS), not a strength change.

### Fixed
- **`go nodes <N>` ignored the node limit.** The default-depth-10 fallback did not check
  `tc.nodes`, so `go nodes N` without a time control was capped at depth 10 and the node limit
  was silently dropped. Node-limited searches now run to the requested node count.
- **Search-thread panic dropped the NNUE.** If the search thread panicked, the join handler
  installed a placeholder `SearchInfo` with no NNUE, so the rest of the game ran on classical
  eval (~−200 Elo) with no visible error. The placeholder now retains the NNUE and the panic is
  reported on stderr.

### Internal
- Removed an unused `CorrectionHistory` update path (material/continuation tables were updated
  every quiet node but never read) — pure CPU/cache savings, `bench` node count unchanged.

## [1.0.2] - 2026-05-28

### Fixed
- **Polyglot hash en-passant gating.** `book::polyglot_hash` mixed the ep-file key whenever
  `pos.ep_square != NO_SQ`. `Position::make_move` sets `ep_square` on every double-pawn-push
  without checking whether a pawn of the side to move can actually capture there, so clrsrc's
  hash diverged from any Polyglot/JBK2 builder that follows the spec (e.g. the bundled
  `jugernaut_v4.book`). Effect: after any 2-step pawn push on the main line, the engine
  silently failed to look up its own book — the experience/book features only worked from
  positions without a "fresh" ep-square. The hash is now gated on capturable ep, matching
  the Polyglot specification. `bench` node count is unchanged (2,846,610); 7/7 tests pass
  including the `polyglot_hash_startpos_vector` regression vector.

## [1.0.1] - 2026-05-27

### Fixed
- **Pondering time management.** After a `ponderhit`, the engine reset its search clock to
  zero, granting a full fresh time budget *on top of* the (free) ponder time. Because the
  search was already at a deep ply from pondering, a single further iteration could consume
  the entire budget, occasionally flagging in increment games. The clock is no longer reset on
  `ponderhit`: time spent pondering now counts against the move budget, so total search time
  stays bounded by the normal per-move limit. Long ponders play instantly (banking the clock);
  short ponders think normally. `bench` node count is unchanged.

### Added
- **JBK2 experience / opening-book support (opt-in, all options default off).** These features
  do not touch the search; the `bench` node count is unaffected whether or not they are enabled.
  - Read a JBK2 book/experience file: UCI options `ExpFile`, `PlayFromExp`, `BookVariety`
    (strict / ±15cp / ±30cp tolerance with a WDL filter). CLI: `clrsrc exp <file> [FEN]`.
  - Persisted learning during play: UCI options `LearnDuringPlay` and `ExpMinSaveDepth`
    (default 16) write the engine's deep root judgments to an append-only `<ExpFile>.overlay`.
  - Offline consolidation: `clrsrc expmerge <book> <overlay> <out> [--clrsrc-mirror]` merges an
    overlay into a sorted main book per the multi-source field policy.
  - Interoperates byte-for-byte with the Jugernaut JBK2 v2 format.
- **Bundled opening/experience book `jugernaut_v4.book`** (~192k positions, JBK2 v2 format),
  generated entirely from self-play and freely redistributable under this project's license.
  Published as a release asset; enable it with `ExpFile` + `PlayFromExp` (off by default).

[1.0.2]: https://github.com/clrsrc/clrsrc/releases/tag/v1.0.2
[1.0.1]: https://github.com/clrsrc/clrsrc/releases/tag/v1.0.1
[1.0.0]: https://github.com/clrsrc/clrsrc/releases/tag/v1.0.0
