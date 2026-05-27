# Changelog

All notable changes to clrsrc are documented here.

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

[1.0.1]: https://github.com/clrsrc/clrsrc/releases/tag/v1.0.1
[1.0.0]: https://github.com/clrsrc/clrsrc/releases/tag/v1.0.0
