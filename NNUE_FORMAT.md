# clrsrc NNUE format & requirements

What an NNUE file must satisfy to be usable by clrsrc. Authoritative source:
`src/nnue.rs` (the `load()` auto-detection + the architecture constants).

clrsrc loads a network from a file with `Nnue::load(path)`. The loader **auto-detects
the format and the architecture parameters from the file's byte size** (and a header
for the CLNN format). If nothing matches it returns `Unknown NNUE format (size: N bytes)`.

---

## 1. Hard requirement — the feature encoding must match clrsrc

This is the part that size alone does **not** catch: a net with the wrong feature set
loads fine but produces a garbage evaluation. The net **must be trained with clrsrc's
exact input encoding**:

- **768 inputs per king-bucket** — `INPUT_SIZE = 768` = 6 piece types × 2 colors × 64 squares.
- **Perspective network** — two accumulators (white/black), side-to-move-relative; the
  black perspective is mirrored. See `feature_index(piece_color, pt, sq, perspective)`.
- **King-bucket indexing** per clrsrc's scheme (`feature_index_bucketed` / `king_bucket_info`):
  `feature = king_bucket(king_square) * 768 + piece_feature`. The live net uses **kb4**
  (4 king buckets).

In practice: train with **clrsrc's `bullet` configuration** (the same harness used for the
training nets, e.g. `clrsrc_25ktest.rs`). An arbitrary NNUE (a Stockfish net, or any other
feature scheme / input order) will **not** work even if its size happens to match.

## 2. Quantization (must match)

- **i16 weights**, **SCReLU** activation.
- `QA = 255` (feature/clamp quantization), `QB = 64` (output-weight quantization).

## 3. Architecture parameters (auto-detected; must be in the supported set)

| Constant | Value | Meaning |
|---|---|---|
| `INPUT_SIZE` | 768 | inputs per king-bucket (fixed) |
| `MAX_HIDDEN` | 1024 | max hidden size H |
| `MAX_BUCKETS` | 16 | max king buckets B |
| `NUM_OUTPUT_BUCKETS` | 8 | output buckets (MaterialCount<8> layout) |
| `L2_SIZE` | 16 | second hidden layer (L2 format only) |

- **Hidden size H** is auto-detected from `{256, 384, 512, 640, 768, 896, 1024}` (≤ `MAX_HIDDEN`).
- **King buckets B** from `{1, 2, 4, 8, 10, 16}` (≤ `MAX_BUCKETS`). Live = 4.
- The file size must **exactly** equal the layout size for (H, B); a 64-byte-aligned padded
  size is also accepted (`padded = (expected + 63) & !63`). Any other size → `Unknown NNUE format`.

## 4. Supported formats and their byte sizes

The loader tries these in order (a net must be exactly **one** of them). `H` = hidden,
`B` = king buckets, `OB` = `NUM_OUTPUT_BUCKETS` = 8.

| Format | Size (bytes, before 64-byte padding) |
|---|---|
| CLNN (clrsrc header `"CLNN"`) | per-header |
| Single-layer `768 → H → 1` | `H * 1542 + 2` |
| King-bucketed `768×B → H → 1` | `H * (1536*B + 6) + 2` |
| King-bucketed + output buckets `768×B → H → OB` | `H * (1536*B + 34) + 16` |
| Threats-bucketed (V0): 896 feat/bucket (768 piece + 128 threat) | `H * (1792*B + 6) + 2` |
| Two-layer (L2) `768 → H → L2(16) → 1` | `768*H*2 + H*2 + H*2*16*2 + 16*2 + 16*2 + 2` |

(`1542 = 768*2 + 2 + 4`; `1536 = 768*2`; `1792 = 896*2`. The `+2` / `+16` are bias blocks.)

## 5. The live reference network

`clrsrc_v32_seed_b.nnue` (internal id `2bff5e05`), **4,723,264 bytes**:
king-bucketed, **B = 4**, **H = 768** → `768*(1536*4 + 6) + 2 = 4,723,202`, padded to
4,723,264 (next multiple of 64). The engine reports it at load as:
`NNUE loaded (bucketed 4): ... (768x4x768→1)`.

Since **v1.1.1** this net is **embedded** in the binary (`include_bytes!`), so the bare
`.exe` plays with it by default.

## 6. How clrsrc selects the net

In priority order:
1. `setoption name EvalFile value <path>` (UCI) — explicit override, loads from file.
2. The **embedded** default net (v1.1.1+) — used when no `EvalFile` is given.
3. CLI: `clrsrc bench <depth> <path>` loads `<path>` (falls back to embedded if absent).

(For SPRT net A/B testing, swap `EvalFile` per engine — see `.claude/skills/sprt`.)

## 7. Verifying a net loads

```bash
# Bench with an explicit net (prints the detected layout + node count):
clrsrc bench 11 path/to/net.nnue
# or via UCI:
printf 'setoption name EvalFile value P:/abs/path/net.nnue\nisready\nquit\n' | clrsrc.exe
```
Look for `NNUE loaded (... ): <name> (768x B x H → …)`. A correct, encoding-matched net
also reproduces the expected `bench` node count for its architecture.

## 8. Why bigger-H experiments need no rebuild

A net with a different **hidden size** in the supported set loads on the existing binary
because the accumulator is sized to `MAX_HIDDEN` (`Accumulator { white: [i16; MAX_HIDDEN], … }`)
and H is read from the file. Example: the kb4 × **1024** headroom-test net (same scheme as
v32, only `H = 768 → 1024`) loads and plays on the current binary — `1024 ≤ MAX_HIDDEN`.
A larger king-bucket count, output-bucket count, L2 size, or a changed feature encoding,
however, **would** require code/constant changes (and a rebuild).
