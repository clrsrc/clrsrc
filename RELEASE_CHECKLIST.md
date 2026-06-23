# Release checklist (clrsrc)

A public release must be **portable, self-contained, and free of personal/internal
information**. The `tools/prerelease_check.py` guardrail enforces the mechanical parts and
runs automatically as a `pre-push` hook — but go through this list before tagging.

## One-time setup (per clone)

```sh
git config core.hooksPath hooks      # activates hooks/pre-push (runs the guardrail)
```

## Automated guardrail (`tools/prerelease_check.py`)

Runs on every `git push` and can be run manually:

```sh
python tools/prerelease_check.py                  # scan tracked files
python tools/prerelease_check.py --notes notes.md # ALSO hard-scan a release-notes file
```

**Hard-fails the push** on:
- **PII / dev paths** — `P:/Projekte…`, `C:/Users…`, `/home/skamm…`, the maintainer's name/email.
- **Internal references** — `netcup`, `interinstanz` (inter-instance bus / private infra).
- **Non-portable build** — `target-cpu=native` in `.cargo/config.toml` (must stay `x86-64-v2`;
  AVX2/AVX-512 are selected at *runtime*, so `native` only risks an illegal-instruction crash
  on other people's machines).

**Warns** (eyeball, does not block) on any `… Elo` / `CCRL` mention in tracked files —
internal self-play SPRT/SPSA deltas in code comments are fine, but anything **absolute /
CCRL** must not appear in public-facing text.

Vetted false positives go in `tools/.release_allow` (`<path-suffix>:<substring>`), e.g. the
"Stefan Meyer-Kahlen" UCI-protocol attribution in `CREDITS.md`.

## Manual checks before tagging

- [ ] `prerelease_check.py` is clean (and you've eyeballed the Elo/CCRL warnings).
- [ ] **Release notes** scanned: `python tools/prerelease_check.py --notes <notes-file>` — public
      notes must have **no** Elo/CCRL figures and **no** PII.
- [ ] Version bumped in `Cargo.toml` + `Cargo.lock`; `CHANGELOG.md` entry added.
- [ ] **Working tree == live engine** — the released source must match what actually runs
      (verify any new search/eval term is deployed/validated, not WIP; cross-check the live
      source if in doubt). Don't ship unvalidated experiments (e.g. an A/B-only constant).
- [ ] Build is **self-contained**: the embedded NNUE loads with no external `.nnue` present.
- [ ] **Bench gate**: `clrsrc bench` matches the expected node count for this build.
- [ ] Embedded net is the intended one (check the `info string NNUE loaded` line).

## Release

```sh
git push origin main          # guardrail runs here
git push origin vX.Y.Z
gh release create vX.Y.Z --title "…" --notes-file <clean-notes> <assets…>
```
