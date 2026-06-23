#!/usr/bin/env python3
"""Pre-release / pre-push guardrail for the PUBLIC clrsrc repositories.

Why this exists
---------------
The v0.3.0 release process leaked two classes of problem into a public artifact:
  1. PII / internal references in tracked source (dev paths like ``P:/Projekte/...``,
     personal names, internal hostnames) — and a non-portable ``target-cpu=native``
     build flag that crashes on other people's CPUs.
  2. An internal Elo figure in the public release notes.

This script makes those un-shippable: it scans the git-tracked files (and,
optionally, a release-notes file) and FAILS the push/release if it finds a hard
violation. Wire it as a ``pre-push`` hook so it runs automatically — a headless
autopilot or an interactive session both get stopped before anything goes public.

Usage
-----
  python tools/prerelease_check.py                 # scan tracked files (hook mode)
  python tools/prerelease_check.py --notes FILE    # also hard-scan a release-notes file
  python tools/prerelease_check.py --strict-elo    # treat Elo/CCRL in tracked files as hard fail too

Exit code 0 = clean (warnings allowed), 1 = hard violation (block).

Categories
----------
HARD FAIL (always block):
  * dev-path        Windows working-tree paths (P:/Projekte, C:/Users, ...)
  * personal-name   stefan / kammann / skamm
  * user-email      the maintainer's email
  * internal-host   netcup / interinstanz / the inter-instance bus
  * target-cpu      `target-cpu=native` in .cargo/config.toml (must stay portable)

WARN (print, do not block tracked files — historical CHANGELOG dev-deltas are ok):
  * elo / ccrl      any Elo/CCRL mention. In a --notes file these become HARD FAIL
                    (public release notes must be free of internal/absolute rating claims).

An explicit allow-list lives in ``tools/.release_allow`` (one ``path:substring``
per line) for vetted false positives (e.g. CREDITS attributions).
"""
import argparse
import os
import re
import subprocess
import sys

# (name, compiled-regex). Matched against every tracked text file.
HARD_PATTERNS = [
    ("dev-path",      re.compile(r"[A-Za-z]:[\\/]+(?:Projekte|Users)\b")),
    ("dev-path",      re.compile(r"/home/skamm\b")),
    ("personal-name", re.compile(r"(?i)\b(stefan|kammann|skamm)\b")),
    ("user-email",    re.compile(r"(?i)skammann@|@dm2tim\b")),
    ("internal-host", re.compile(r"(?i)\b(netcup|interinstanz)\b")),
]
# target-cpu=native is only a problem in the shipped cargo config.
CARGO_CFG = os.path.join(".cargo", "config.toml")
ELO_PATTERNS = [
    ("elo",  re.compile(r"(?i)[+-]?\s*\d+(?:\.\d+)?\s*(?:±\s*\d+(?:\.\d+)?\s*)?Elo\b")),
    ("ccrl", re.compile(r"(?i)\bCCRL\b")),
]
# Text extensions worth scanning (skip binaries / nnue / books).
TEXT_EXT = {".rs", ".toml", ".md", ".txt", ".lock", ".yml", ".yaml", ".cfg", ".json", ".sh", ".py"}

# The guardrail's own files legitimately CONTAIN the patterns (the regexes, the allow-list,
# the checklist documenting them) — never scan them, or they self-trip.
SELF_EXCLUDE = {
    "tools/prerelease_check.py",
    "tools/.release_allow",
    "RELEASE_CHECKLIST.md",
    "hooks/pre-push",
}


def tracked_files(repo):
    out = subprocess.run(["git", "-C", repo, "ls-files"], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines()
            if os.path.splitext(f)[1].lower() in TEXT_EXT
            and f.replace("\\", "/") not in SELF_EXCLUDE]


def load_allow(repo):
    allow = []
    p = os.path.join(repo, "tools", ".release_allow")
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                path, sub = line.split(":", 1)
                allow.append((path.strip(), sub.strip()))
    return allow


def allowed(allow, relpath, line):
    return any(relpath.replace("\\", "/").endswith(ap) and sub in line for ap, sub in allow)


def scan_file(repo, relpath, patterns):
    hits = []
    full = os.path.join(repo, relpath)
    try:
        lines = open(full, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return hits
    for n, line in enumerate(lines, 1):
        for name, rx in patterns:
            if rx.search(line):
                hits.append((name, relpath, n, line.strip()[:120]))
    return hits


def main():
    # Windows consoles default to cp1252 and choke on the ± / − in CHANGELOG lines.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--notes", help="release-notes file to hard-scan (Elo/CCRL + PII = fail)")
    ap.add_argument("--strict-elo", action="store_true", help="Elo/CCRL in tracked files is a hard fail too")
    args = ap.parse_args()
    repo = os.path.abspath(args.repo)
    allow = load_allow(repo)

    hard, warn = [], []
    for f in tracked_files(repo):
        for hit in scan_file(repo, f, HARD_PATTERNS):
            (hard if not allowed(allow, hit[1], hit[3]) else warn).append(hit)
        for hit in scan_file(repo, f, ELO_PATTERNS):
            if allowed(allow, hit[1], hit[3]):
                continue
            (hard if args.strict_elo else warn).append(hit)

    # target-cpu=native in the shipped cargo config
    cfg = os.path.join(repo, CARGO_CFG)
    if os.path.exists(cfg):
        for n, line in enumerate(open(cfg, encoding="utf-8").read().splitlines(), 1):
            if re.search(r"target-cpu\s*=\s*[\"']?native", line):
                hard.append(("target-cpu", CARGO_CFG, n, line.strip()))

    # release notes: PII + Elo/CCRL all hard
    if args.notes and os.path.exists(args.notes):
        for n, line in enumerate(open(args.notes, encoding="utf-8").read().splitlines(), 1):
            for name, rx in HARD_PATTERNS + ELO_PATTERNS:
                if rx.search(line):
                    hard.append((f"notes:{name}", os.path.basename(args.notes), n, line.strip()[:120]))

    def show(title, items):
        if not items:
            return
        print(f"\n{title}")
        for name, path, n, line in items:
            print(f"  [{name}] {path}:{n}: {line}")

    show("WARN (review before public release):", warn)
    show("HARD FAIL (blocking — must be removed before public release):", hard)

    if hard:
        print(f"\n✗ prerelease_check: {len(hard)} blocking violation(s). Push/release aborted.")
        print("  Fix them, or add a vetted false-positive to tools/.release_allow (path:substring).")
        return 1
    print(f"\n✓ prerelease_check: clean ({len(warn)} warning(s) to eyeball). OK to release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
