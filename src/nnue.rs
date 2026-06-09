// ---- NNUE: Efficiently Updatable Neural Network ----
// Architecture: (768 → H) × 2 → 1  (single-layer)
//           or: (768 → H) × 2 → L2 → 1  (L2 architecture)
//           or: (768*B → H) × 2 → 1  (king-bucketed, horizontally mirrored)
// Input: 768 features = 2 colors × 6 piece types × 64 squares, per perspective
// Bucketed: each perspective selects a bucket based on own king square
// Accumulator: incrementally updated i16[H] per perspective
// Output: SCReLU activation → i32 dot product → centipawns
// Quantization: feature layer QA=255, output layer QB=64

use crate::types::*;
use crate::bitboard::*;
use crate::board::Position;
use std::sync::Arc;

pub const INPUT_SIZE: usize = 768;
pub const MAX_HIDDEN: usize = 1024; // max supported hidden size
pub const L2_SIZE: usize = 16;
pub const MAX_BUCKETS: usize = 16; // max supported king buckets
pub const NUM_OUTPUT_BUCKETS: usize = 8; // bullet MaterialCount<8> layout

/// Output-bucket index from total piece count, matching bullet MaterialCount<N=8>:
///     divisor = ceil(32 / N) = 4
///     bucket  = (popcount - 2) / divisor
#[inline]
pub fn output_bucket_for_occupied(occupied_count: u32) -> usize {
    let divisor = 32usize.div_ceil(NUM_OUTPUT_BUCKETS);
    (occupied_count.saturating_sub(2) as usize) / divisor
}
const QA: i32 = 255; // SCReLU clamp value (feature quantization)
const QB: i32 = 64;  // output weight quantization
const NNUE_SCALE: i32 = 400; // network output → centipawn scaling

// ---- King bucket layout (horizontally mirrored) ----
// 32 entries for queenside (files a-d), mirrored to kingside (files e-h)
// Must match the layout in bullet/examples/clrsrc_bucketed.rs!
#[rustfmt::skip]
const BUCKET_LAYOUT_QS: [usize; 32] = [
    0, 0, 1, 1,  // rank 1
    0, 0, 1, 1,  // rank 2
    2, 2, 3, 3,  // rank 3
    2, 2, 3, 3,  // rank 4
    2, 2, 3, 3,  // rank 5
    2, 2, 3, 3,  // rank 6
    2, 2, 3, 3,  // rank 7
    2, 2, 3, 3,  // rank 8
];

// Expanded to 64 squares (with horizontal mirroring applied)
const BUCKET_LAYOUT: [usize; 64] = expand_bucket_layout();

const fn expand_bucket_layout() -> [usize; 64] {
    let mirror_file: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];
    let mut out = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        let rank = i / 8;
        let file = i % 8;
        out[i] = BUCKET_LAYOUT_QS[rank * 4 + mirror_file[file]];
        i += 1;
    }
    out
}

// ---- Feature indexing ----
// Unbucketed: features in [0..768), index = relative_color * 384 + piece_type * 64 + rel_sq
// Bucketed: features in [0..768*B), index = bucket_offset + (base_index ^ h_flip)

#[inline]
pub fn feature_index(piece_color: Color, pt: PieceType, sq: Square, perspective: Color) -> usize {
    let rel_color = if piece_color == perspective { 0 } else { 1 };
    let rel_sq = if perspective == WHITE { sq } else { mirror_sq(sq) };
    rel_color * 384 + pt.index() * 64 + rel_sq as usize
}

/// Compute bucket offset and horizontal flip for a perspective's king square.
/// Uses raw (absolute) king square for bucket lookup — must match Bullet trainer
/// (ChessBucketsMirrored), which does NOT vertically mirror king squares.
/// Only horizontal mirroring (h_flip) is applied to features within the bucket.
#[inline]
pub fn king_bucket_info(ksq: Square, _perspective: Color) -> (usize, usize) {
    let file = ksq as usize % 8;
    let h_flip = if file > 3 { 7 } else { 0 }; // XOR with 7 flips file bits
    let bucket_offset = INPUT_SIZE * BUCKET_LAYOUT[ksq as usize];
    (bucket_offset, h_flip)
}

/// Bucketed feature index: bucket_offset + (base_feature ^ h_flip)
#[inline]
pub fn feature_index_bucketed(
    piece_color: Color, pt: PieceType, sq: Square,
    perspective: Color, bucket_offset: usize, h_flip: usize,
) -> usize {
    let base = feature_index(piece_color, pt, sq, perspective);
    bucket_offset + (base ^ h_flip)
}

/// king_bucket_info with explicit stride (uses 896 for threats layout, 768 for plain).
#[inline]
pub fn king_bucket_info_with_stride(ksq: Square, stride: usize) -> (usize, usize) {
    let file = ksq as usize % 8;
    let h_flip = if file > 3 { 7 } else { 0 };
    let bucket_offset = stride * BUCKET_LAYOUT[ksq as usize];
    (bucket_offset, h_flip)
}

/// Threat feature index in the 896-feature-per-bucket layout (V0 threats).
/// "attacker_color attacks square sq" → index 768 + own/opp_offset(0/64) + sq_relative
#[inline]
pub fn threat_feature_index_bucketed(
    attacker_color: Color, sq: Square,
    perspective: Color, bucket_offset: usize, h_flip: usize,
) -> usize {
    let rel_color = if attacker_color == perspective { 0 } else { 1 };
    let rel_sq = if perspective == WHITE { sq } else { mirror_sq(sq) };
    let base = 768 + rel_color * 64 + rel_sq as usize;
    bucket_offset + (base ^ h_flip)
}

/// SCReLU (Squared Clipped ReLU): clamp(x, 0, QA)²
#[inline]
fn screlu(x: i16) -> i32 {
    let y = (x as i32).clamp(0, QA);
    y * y
}


// ---- Accumulator ----
// align(64) so white[..] and black[..] start on 64-byte boundaries.
// black is at offset MAX_HIDDEN * 2 bytes = 2048, also a multiple of 64.
// This lets AVX-512 use aligned 512-bit loads (`_mm512_load_si512`).

#[derive(Clone)]
#[repr(C, align(64))]
pub struct Accumulator {
    pub white: [i16; MAX_HIDDEN],
    pub black: [i16; MAX_HIDDEN],
    pub computed: bool,
    /// King bucket info per perspective (bucket_offset, h_flip) — only used for bucketed nets
    pub white_kb: (usize, usize),
    pub black_kb: (usize, usize),
}

impl Accumulator {
    pub fn new() -> Self {
        Accumulator {
            white: [0; MAX_HIDDEN],
            black: [0; MAX_HIDDEN],
            computed: false,
            white_kb: (0, 0),
            black_kb: (0, 0),
        }
    }
}

// ---- Feature delta for incremental updates ----

pub struct FeatureDelta {
    added: [(usize, usize); 4],   // (white_perspective_idx, black_perspective_idx)
    removed: [(usize, usize); 4],
    n_added: usize,
    n_removed: usize,
    /// True iff a king move actually crossed a king-bucket boundary (or the
    /// horizontal-mirror line). Only then does a bucketed net need a full
    /// refresh; ~70% of king moves stay in the same bucket and can use
    /// incremental update.
    pub king_bucket_changed: bool,
}

impl FeatureDelta {
    fn new() -> Self {
        FeatureDelta {
            added: [(0, 0); 4],
            removed: [(0, 0); 4],
            n_added: 0,
            n_removed: 0,
            king_bucket_changed: false,
        }
    }

    #[inline]
    fn add_feature(&mut self, color: Color, pt: PieceType, sq: Square) {
        let w = feature_index(color, pt, sq, WHITE);
        let b = feature_index(color, pt, sq, BLACK);
        self.added[self.n_added] = (w, b);
        self.n_added += 1;
    }

    #[inline]
    fn remove_feature(&mut self, color: Color, pt: PieceType, sq: Square) {
        let w = feature_index(color, pt, sq, WHITE);
        let b = feature_index(color, pt, sq, BLACK);
        self.removed[self.n_removed] = (w, b);
        self.n_removed += 1;
    }
}

/// Compute the feature delta for a move. Call BEFORE make_move.
pub fn compute_delta(pos: &Position, mv: Move) -> FeatureDelta {
    let mut d = FeatureDelta::new();
    let us = pos.side;
    let them = us.flip();
    let from = mv.from();
    let to = mv.to();
    let moving_pt = piece_type(pos.mailbox[from as usize]);

    // For bucketed nets: a refresh is only needed when the moving king crosses
    // a bucket boundary (BUCKET_LAYOUT) or the horizontal-mirror line (h_flip).
    // ~70% of king moves stay in the same bucket → save a full refresh.
    // For non-bucketed nets the caller's is_bucketed() guard makes this moot.
    if moving_pt == KING {
        let info_from = king_bucket_info(from, us);
        let info_to = king_bucket_info(to, us);
        d.king_bucket_changed = info_from != info_to;
    }

    match mv.flags() {
        Move::FLAG_QUIET | Move::FLAG_DOUBLE_PAWN => {
            d.remove_feature(us, moving_pt, from);
            d.add_feature(us, moving_pt, to);
        }
        Move::FLAG_CAPTURE => {
            let cap_pt = piece_type(pos.mailbox[to as usize]);
            d.remove_feature(us, moving_pt, from);
            d.remove_feature(them, cap_pt, to);
            d.add_feature(us, moving_pt, to);
        }
        Move::FLAG_EP => {
            let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
            d.remove_feature(us, PAWN, from);
            d.remove_feature(them, PAWN, cap_sq);
            d.add_feature(us, PAWN, to);
        }
        Move::FLAG_KING_CASTLE => {
            d.remove_feature(us, KING, from);
            d.add_feature(us, KING, to);
            let (rf, rt) = if us == WHITE { (H1, F1) } else { (H8, F8) };
            d.remove_feature(us, ROOK, rf);
            d.add_feature(us, ROOK, rt);
        }
        Move::FLAG_QUEEN_CASTLE => {
            d.remove_feature(us, KING, from);
            d.add_feature(us, KING, to);
            let (rf, rt) = if us == WHITE { (A1, D1) } else { (A8, D8) };
            d.remove_feature(us, ROOK, rf);
            d.add_feature(us, ROOK, rt);
        }
        _ if mv.is_promotion() => {
            d.remove_feature(us, PAWN, from);
            d.add_feature(us, mv.promo_piece_type(), to);
            if mv.is_capture() {
                let cap_pt = piece_type(pos.mailbox[to as usize]);
                d.remove_feature(them, cap_pt, to);
            }
        }
        _ => {
            // Fallback (shouldn't happen with valid moves)
            d.remove_feature(us, moving_pt, from);
            d.add_feature(us, moving_pt, to);
        }
    }

    d
}

// ---- Network architecture variants ----

enum NnueArch {
    /// (768→H)×2→1 — single output layer
    SingleLayer {
        output_weights: Box<[i16]>,  // [hidden_size * 2]
        output_bias: i16,
    },
    /// (768*B→H)×2→NUM_OUTPUT_BUCKETS — output-bucketed single layer (bullet transposed save)
    SingleLayerOutBucket {
        output_weights: Box<[i16]>,  // [NUM_OUTPUT_BUCKETS][hidden_size * 2], bucket-contiguous
        output_biases: [i16; NUM_OUTPUT_BUCKETS],
    },
    /// (768→512)×2→L2→1 — two output layers
    L2Layer {
        l1_weights: Vec<[i16; L2_SIZE]>,   // [2*HIDDEN_SIZE][L2_SIZE]
        l1_biases: [i16; L2_SIZE],
        l2_weights: [i16; L2_SIZE],
        l2_bias: i16,
    },
}

// ---- Network parameters ----

pub struct NnueParams {
    feature_weights: Vec<[i16; MAX_HIDDEN]>, // (features_per_bucket * num_buckets) × MAX_HIDDEN
    feature_biases: [i16; MAX_HIDDEN],       // only first hidden_size used
    hidden_size: usize,
    num_buckets: usize,                      // 1 = unbucketed, >1 = king-bucketed
    features_per_bucket: usize,              // 768 (plain) or 896 (with threats)
    has_threats: bool,                       // if true: extra 128 features/bucket for threats
    arch: NnueArch,
}

// ---- Runtime SIMD dispatch ----
// Detected once at construction; cached so hot paths just match on an enum.
#[derive(Copy, Clone, PartialEq, Eq)]
enum SimdImpl {
    Scalar,
    Avx2,
    Avx512,
}

#[inline]
fn detect_simd() -> SimdImpl {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx512bw") && std::is_x86_feature_detected!("avx512f") {
            return SimdImpl::Avx512;
        }
        if std::is_x86_feature_detected!("avx2") {
            return SimdImpl::Avx2;
        }
    }
    SimdImpl::Scalar
}

// ---- NNUE engine ----

pub struct Nnue {
    params: Option<Arc<NnueParams>>,
    simd: SimdImpl,
}

impl Clone for Nnue {
    fn clone(&self) -> Self {
        Nnue { params: self.params.clone(), simd: self.simd }
    }
}

impl Nnue {
    pub fn new() -> Self {
        Nnue { params: None, simd: detect_simd() }
    }

    pub fn is_loaded(&self) -> bool {
        self.params.is_some()
    }

    /// Load network — auto-detects format and hidden size by file size:
    /// Single-layer: size = H * (768*2 + 2 + 4) + 2 = H * 1542 + 2
    /// L2: size = H * (768*2 + 2 + 2*L2*2 + 2*L2/H + 2/H) ... (complex)
    pub fn load(&mut self, path: &str) -> Result<(), String> {
        let data = std::fs::read(path).map_err(|e| format!("Failed to read {}: {}", path, e))?;
        self.load_bytes(data.as_slice(), path)
    }

    /// Load the embedded default network (clrsrc_v32_seed_b) so the release binary
    /// is self-contained (no external EvalFile needed; EvalFile still overrides).
    pub fn load_embedded(&mut self) -> Result<(), String> {
        static EMBEDDED_NNUE: &[u8] = include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/clrsrc_v32_seed_b.nnue"));
        self.load_bytes(EMBEDDED_NNUE, "<embedded clrsrc_v32_seed_b>")
    }

    /// Parse + load a network from raw bytes (format auto-detected by header/size).
    fn load_bytes(&mut self, data: &[u8], path: &str) -> Result<(), String> {

        // Try CLNN header first
        if data.len() >= 4 && &data[0..4] == b"CLNN" {
            return self.load_clnn(data, path);
        }

        // Auto-detect hidden size for single-layer: size = H * 1542 + 2 (+ padding)
        // Formula: INPUT_SIZE * H * 2 + H * 2 + H * 2 * 2 + 2 = H * (768*2 + 2 + 4) + 2
        let per_hidden = INPUT_SIZE * 2 + 2 + 4; // 1542
        for &h in &[256usize, 384, 512, 640, 768, 896, 1024] {
            let expected = h * per_hidden + 2;
            let padded = (expected + 63) & !63;
            if data.len() == expected || data.len() == padded {
                if h > MAX_HIDDEN {
                    return Err(format!("Hidden size {} exceeds MAX_HIDDEN {}", h, MAX_HIDDEN));
                }
                return self.load_bullet_single_dynamic(data, path, h);
            }
        }

        // Try bucketed single-layer: size = H * (768*B*2 + 2 + 4) + 2
        for &h in &[256usize, 384, 512, 640, 768, 896, 1024] {
            for &b in &[2usize, 4, 8, 10, 16] {
                let per_hidden_b = INPUT_SIZE * b * 2 + 2 + 4;
                let expected = h * per_hidden_b + 2;
                let padded = (expected + 63) & !63;
                if data.len() == expected || data.len() == padded {
                    if h > MAX_HIDDEN || b > MAX_BUCKETS {
                        return Err(format!("H={} or B={} exceeds limits", h, b));
                    }
                    return self.load_bullet_bucketed(data, path, h, b);
                }
            }
        }

        // Try bucketed single-layer with NUM_OUTPUT_BUCKETS output buckets:
        //   features: INPUT_SIZE*B*H*i16 + H*i16
        //   l1:       NUM_OUTPUT_BUCKETS * 2*H * i16 + NUM_OUTPUT_BUCKETS * i16
        //   total = H*(2*INPUT_SIZE*B + 2 + 4*NUM_OUTPUT_BUCKETS) + 2*NUM_OUTPUT_BUCKETS
        for &h in &[256usize, 384, 512, 640, 768, 896, 1024] {
            for &b in &[1usize, 2, 4, 8, 10, 16] {
                let expected = h * (2 * INPUT_SIZE * b + 2 + 4 * NUM_OUTPUT_BUCKETS) + 2 * NUM_OUTPUT_BUCKETS;
                let padded = (expected + 63) & !63;
                if data.len() == expected || data.len() == padded {
                    if h > MAX_HIDDEN || b > MAX_BUCKETS {
                        return Err(format!("H={} or B={} exceeds limits", h, b));
                    }
                    return self.load_bullet_outbucket(data, path, h, b);
                }
            }
        }

        // Try bucketed single-layer WITH THREATS (V0): 896 features per bucket (768 piece + 128 threat)
        //   total = H*(2*896*B + 2 + 4) + 2 = H*(1792*B + 6) + 2
        for &h in &[256usize, 384, 512, 640, 768, 896, 1024] {
            for &b in &[1usize, 2, 4, 8, 10, 16] {
                let expected = h * (1792 * b + 6) + 2;
                let padded = (expected + 63) & !63;
                if data.len() == expected || data.len() == padded {
                    if h > MAX_HIDDEN || b > MAX_BUCKETS {
                        return Err(format!("H={} or B={} exceeds limits", h, b));
                    }
                    return self.load_bullet_threats_bucketed(data, path, h, b);
                }
            }
        }

        // Try L2 with various hidden sizes
        for &h in &[256usize, 384, 512, 640, 768, 896, 1024] {
            // L2 file size: feature_weights(768*H*2) + feature_biases(H*2)
            //             + l1_weights(2H*L2*2) + l1_biases(L2*2) + l2_weights(L2*2) + l2_bias(2)
            let l2_file_size = INPUT_SIZE * h * 2 + h * 2 + h * 2 * L2_SIZE * 2 + L2_SIZE * 2 + L2_SIZE * 2 + 2;
            let l2_padded = (l2_file_size + 63) & !63;
            if data.len() == l2_file_size || data.len() == l2_padded {
                if h > MAX_HIDDEN {
                    return Err(format!("Hidden size {} exceeds MAX_HIDDEN {}", h, MAX_HIDDEN));
                }
                return self.load_bullet_l2(data, path, h);
            }
        }

        Err(format!("Unknown NNUE format (size: {} bytes)", data.len()))
    }

    /// Load Bullet single-layer with auto-detected hidden size
    fn load_bullet_single_dynamic(&mut self, data: &[u8], path: &str, h: usize) -> Result<(), String> {
        let mut off = 0;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            for j in 0..h {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..h {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut output_weights = vec![0i16; h * 2];
        for j in 0..h * 2 {
            output_weights[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let output_bias = i16::from_le_bytes([data[off], data[off + 1]]);

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: h,
            num_buckets: 1,
            features_per_bucket: 768,
            has_threats: false,
            arch: NnueArch::SingleLayer { output_weights: output_weights.into_boxed_slice(), output_bias },
        }));

        eprintln!("info string NNUE loaded (single-layer): {} ({}x{}→1)", path, INPUT_SIZE, h);
        Ok(())
    }

    /// Load Bullet bucketed single-layer: (768*B→H)×2→1
    fn load_bullet_bucketed(&mut self, data: &[u8], path: &str, h: usize, b: usize) -> Result<(), String> {
        let mut off = 0;
        let input_size = INPUT_SIZE * b;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; input_size];
        for i in 0..input_size {
            for j in 0..h {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..h {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut output_weights = vec![0i16; h * 2];
        for j in 0..h * 2 {
            output_weights[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let output_bias = i16::from_le_bytes([data[off], data[off + 1]]);

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: h,
            num_buckets: b,
            features_per_bucket: 768,
            has_threats: false,
            arch: NnueArch::SingleLayer { output_weights: output_weights.into_boxed_slice(), output_bias },
        }));

        eprintln!("info string NNUE loaded (bucketed {}): {} ({}x{}x{}→1)", b, path, INPUT_SIZE, b, h);
        Ok(())
    }

    /// Load Bullet output-bucketed: (768*B→H)×2→NUM_OUTPUT_BUCKETS, bullet `.transpose()` save (per-bucket contiguous).
    fn load_bullet_outbucket(&mut self, data: &[u8], path: &str, h: usize, b: usize) -> Result<(), String> {
        let mut off = 0;
        let input_size = INPUT_SIZE * b;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; input_size];
        for i in 0..input_size {
            for j in 0..h {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..h {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        // Output weights: bullet transpose() saves as [NUM_OUTPUT_BUCKETS][2*h] (per-bucket contiguous).
        let mut output_weights = vec![0i16; NUM_OUTPUT_BUCKETS * 2 * h];
        for i in 0..NUM_OUTPUT_BUCKETS * 2 * h {
            output_weights[i] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut output_biases = [0i16; NUM_OUTPUT_BUCKETS];
        for i in 0..NUM_OUTPUT_BUCKETS {
            output_biases[i] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: h,
            num_buckets: b,
            features_per_bucket: 768,
            has_threats: false,
            arch: NnueArch::SingleLayerOutBucket {
                output_weights: output_weights.into_boxed_slice(),
                output_biases,
            },
        }));

        eprintln!("info string NNUE loaded (bucketed {}, out-buckets {}): {} ({}x{}x{}→{})",
                  b, NUM_OUTPUT_BUCKETS, path, INPUT_SIZE, b, h, NUM_OUTPUT_BUCKETS);
        Ok(())
    }

    /// Load Bullet bucketed-with-threats V0: (768+128)*B → H × 2 → 1.
    /// 896 features per bucket = 768 piece-square + 128 threat (2 colors × 64 squares).
    fn load_bullet_threats_bucketed(&mut self, data: &[u8], path: &str, h: usize, b: usize) -> Result<(), String> {
        let mut off = 0;
        let features_per_bucket = 896;
        let total_features = features_per_bucket * b;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; total_features];
        for i in 0..total_features {
            for j in 0..h {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..h {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut output_weights = vec![0i16; h * 2];
        for j in 0..h * 2 {
            output_weights[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let output_bias = i16::from_le_bytes([data[off], data[off + 1]]);

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: h,
            num_buckets: b,
            features_per_bucket,
            has_threats: true,
            arch: NnueArch::SingleLayer { output_weights: output_weights.into_boxed_slice(), output_bias },
        }));

        eprintln!("info string NNUE loaded (bucketed {}, threats V0): {} (({}+128)x{}x{}→1)",
                  b, path, INPUT_SIZE, b, h);
        Ok(())
    }

    /// Load Bullet L2: (768→H)×2→L2_SIZE→1
    fn load_bullet_l2(&mut self, data: &[u8], path: &str, h: usize) -> Result<(), String> {
        let mut off = 0;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            for j in 0..h {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..h {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let n_l1_inputs = h * 2;
        let mut l1_weights = vec![[0i16; L2_SIZE]; n_l1_inputs];
        for i in 0..n_l1_inputs {
            for j in 0..L2_SIZE {
                l1_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut l1_biases = [0i16; L2_SIZE];
        for j in 0..L2_SIZE {
            l1_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut l2_weights = [0i16; L2_SIZE];
        for j in 0..L2_SIZE {
            l2_weights[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let l2_bias = i16::from_le_bytes([data[off], data[off + 1]]);

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: h,
            num_buckets: 1,
            features_per_bucket: 768,
            has_threats: false,
            arch: NnueArch::L2Layer { l1_weights, l1_biases, l2_weights, l2_bias },
        }));

        eprintln!("info string NNUE loaded (L2): {} ({}x{}→{}→1)", path, INPUT_SIZE, h, L2_SIZE);
        Ok(())
    }

    /// Load CLNN format (legacy, header + data)
    fn load_clnn(&mut self, data: &[u8], path: &str) -> Result<(), String> {
        if data.len() < 12 {
            return Err("CLNN file too small".to_string());
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let hidden = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;

        if version != 1 || hidden > MAX_HIDDEN {
            return Err(format!("Unsupported version {} or hidden size {}", version, hidden));
        }

        let mut off = 12;

        let mut feature_weights = vec![[0i16; MAX_HIDDEN]; INPUT_SIZE];
        for i in 0..INPUT_SIZE {
            for j in 0..hidden {
                feature_weights[i][j] = i16::from_le_bytes([data[off], data[off + 1]]);
                off += 2;
            }
        }

        let mut feature_biases = [0i16; MAX_HIDDEN];
        for j in 0..hidden {
            feature_biases[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        let mut output_weights = vec![0i16; hidden * 2];
        for j in 0..hidden * 2 {
            output_weights[j] = i16::from_le_bytes([data[off], data[off + 1]]);
            off += 2;
        }

        // CLNN stores output_bias as i32, truncate to i16
        let bias_i32 = i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let output_bias = bias_i32.clamp(-32768, 32767) as i16;

        self.params = Some(Arc::new(NnueParams {
            feature_weights,
            feature_biases,
            hidden_size: hidden,
            num_buckets: 1,
            features_per_bucket: 768,
            has_threats: false,
            arch: NnueArch::SingleLayer { output_weights: output_weights.into_boxed_slice(), output_bias },
        }));

        eprintln!("info string NNUE loaded (CLNN format): {} ({}x{})", path, INPUT_SIZE, hidden);
        Ok(())
    }

    /// Compute accumulator from scratch for a position
    pub fn refresh(&self, pos: &Position, acc: &mut Accumulator) {
        let params = self.params.as_ref().unwrap();
        let h = params.hidden_size;
        let simd = self.simd;

        if params.num_buckets > 1 {
            // Bucketed: compute king bucket info for each perspective (stride depends on layout)
            let stride = params.features_per_bucket;
            let wk = pos.king_sq(WHITE);
            let bk = pos.king_sq(BLACK);
            let (w_boff, w_flip) = king_bucket_info_with_stride(wk, stride);
            let (b_boff, b_flip) = king_bucket_info_with_stride(bk, stride);
            acc.white_kb = (w_boff, w_flip);
            acc.black_kb = (b_boff, b_flip);

            // Collect feature indices for both perspectives, then apply
            // chunk-major fan-in (bias→register→add N→store once per H-chunk).
            // Bound: 32 pieces + up to 128 threat squares per perspective = 160.
            const MAX_FEATURES: usize = 192;
            let mut w_indices = [0usize; MAX_FEATURES];
            let mut b_indices = [0usize; MAX_FEATURES];
            let mut n = 0;

            for color_idx in 0..2 {
                let color = if color_idx == 0 { WHITE } else { BLACK };
                for pt_idx in 0..6 {
                    let pt = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING][pt_idx];
                    let mut pieces = pos.pieces[pt_idx] & pos.colors[color_idx];
                    while pieces != 0 {
                        let sq = pop_lsb(&mut pieces);
                        w_indices[n] = feature_index_bucketed(color, pt, sq, WHITE, w_boff, w_flip);
                        b_indices[n] = feature_index_bucketed(color, pt, sq, BLACK, b_boff, b_flip);
                        n += 1;
                    }
                }
            }

            // V0 Threats: per-color attack bitboards → up to 128 threat features per perspective.
            if params.has_threats {
                let w_atk = crate::attacks::attacks_by_color(&pos.pieces, &pos.colors, WHITE);
                let b_atk = crate::attacks::attacks_by_color(&pos.pieces, &pos.colors, BLACK);
                for (atk_bb, atk_color) in [(w_atk, WHITE), (b_atk, BLACK)] {
                    let mut a = atk_bb;
                    while a != 0 {
                        let sq = pop_lsb(&mut a);
                        w_indices[n] = threat_feature_index_bucketed(atk_color, sq, WHITE, w_boff, w_flip);
                        b_indices[n] = threat_feature_index_bucketed(atk_color, sq, BLACK, b_boff, b_flip);
                        n += 1;
                    }
                }
            }

            refresh_fanin(&mut acc.white, &params.feature_biases, &params.feature_weights, &w_indices[..n], h, simd);
            refresh_fanin(&mut acc.black, &params.feature_biases, &params.feature_weights, &b_indices[..n], h, simd);
        } else {
            acc.white[..h].copy_from_slice(&params.feature_biases[..h]);
            acc.black[..h].copy_from_slice(&params.feature_biases[..h]);
            // Unbucketed: original path
            for color_idx in 0..2 {
                let color = if color_idx == 0 { WHITE } else { BLACK };
                for pt_idx in 0..6 {
                    let pt = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING][pt_idx];
                    let mut pieces = pos.pieces[pt_idx] & pos.colors[color_idx];
                    while pieces != 0 {
                        let sq = pop_lsb(&mut pieces);
                        let w_idx = feature_index(color, pt, sq, WHITE);
                        let b_idx = feature_index(color, pt, sq, BLACK);
                        let fw = &params.feature_weights[w_idx];
                        let fb = &params.feature_weights[b_idx];
                        vec_add_i16(&mut acc.white, fw, h, simd);
                        vec_add_i16(&mut acc.black, fb, h, simd);
                    }
                }
            }
        }

        acc.computed = true;
    }

    /// Returns true if this is a bucketed network
    #[inline]
    pub fn is_bucketed(&self) -> bool {
        self.params.as_ref().map_or(false, |p| p.num_buckets > 1)
    }

    /// V0 Threats: any move can change attack bitboards (discovered/blocked sliders, moved-piece attacks).
    /// For correctness we full-refresh every position; incremental update is a V1+ task.
    #[inline]
    pub fn needs_full_refresh_per_move(&self) -> bool {
        self.params.as_ref().map_or(false, |p| p.has_threats)
    }

    /// Incrementally update accumulator from parent using a delta.
    /// For bucketed nets with king moves, caller must use refresh() instead.
    pub fn update_inc(&self, parent: &Accumulator, child: &mut Accumulator, delta: &FeatureDelta) {
        let params = self.params.as_ref().unwrap();
        let h = params.hidden_size;
        let simd = self.simd;

        // Fast path: a simple quiet move (1 feature removed, 1 added) fuses the
        // copy + sub + add 3-pass sequence into a single pass per half. Bit-identical
        // (same per-lane op order). Quiet moves dominate the tree, so this is the hot case.
        if delta.n_removed == 1 && delta.n_added == 1 {
            let (w_rm, b_rm, w_add, b_add) = if params.num_buckets > 1 {
                let (w_boff, w_flip) = parent.white_kb;
                let (b_boff, b_flip) = parent.black_kb;
                let (rw, rb) = delta.removed[0];
                let (aw, ab) = delta.added[0];
                (w_boff + (rw ^ w_flip), b_boff + (rb ^ b_flip),
                 w_boff + (aw ^ w_flip), b_boff + (ab ^ b_flip))
            } else {
                let (rw, rb) = delta.removed[0];
                let (aw, ab) = delta.added[0];
                (rw, rb, aw, ab)
            };
            child.white_kb = parent.white_kb;
            child.black_kb = parent.black_kb;
            vec_add_sub_i16(&mut child.white, &parent.white, &params.feature_weights[w_add], &params.feature_weights[w_rm], h, simd);
            vec_add_sub_i16(&mut child.black, &parent.black, &params.feature_weights[b_add], &params.feature_weights[b_rm], h, simd);
            child.computed = true;
            return;
        }

        // Fast path: a capture (2 removed, 1 added — moving piece's from-square and the
        // captured piece both leave, the moving piece arrives). Covers plain, en-passant,
        // and promotion captures. Fuses copy+sub+sub+add into one pass. Bit-identical.
        if delta.n_removed == 2 && delta.n_added == 1 {
            let (w_rm0, b_rm0, w_rm1, b_rm1, w_add, b_add) = if params.num_buckets > 1 {
                let (w_boff, w_flip) = parent.white_kb;
                let (b_boff, b_flip) = parent.black_kb;
                let (rw0, rb0) = delta.removed[0];
                let (rw1, rb1) = delta.removed[1];
                let (aw, ab) = delta.added[0];
                (w_boff + (rw0 ^ w_flip), b_boff + (rb0 ^ b_flip),
                 w_boff + (rw1 ^ w_flip), b_boff + (rb1 ^ b_flip),
                 w_boff + (aw ^ w_flip), b_boff + (ab ^ b_flip))
            } else {
                let (rw0, rb0) = delta.removed[0];
                let (rw1, rb1) = delta.removed[1];
                let (aw, ab) = delta.added[0];
                (rw0, rb0, rw1, rb1, aw, ab)
            };
            child.white_kb = parent.white_kb;
            child.black_kb = parent.black_kb;
            vec_add_sub_sub_i16(&mut child.white, &parent.white, &params.feature_weights[w_add], &params.feature_weights[w_rm0], &params.feature_weights[w_rm1], h, simd);
            vec_add_sub_sub_i16(&mut child.black, &parent.black, &params.feature_weights[b_add], &params.feature_weights[b_rm0], &params.feature_weights[b_rm1], h, simd);
            child.computed = true;
            return;
        }

        // General path (castling, double-add cases): copy then apply each feature.
        child.white[..h].copy_from_slice(&parent.white[..h]);
        child.black[..h].copy_from_slice(&parent.black[..h]);

        if params.num_buckets > 1 {
            // Bucketed incremental update: use parent's king bucket info
            let (w_boff, w_flip) = parent.white_kb;
            let (b_boff, b_flip) = parent.black_kb;
            child.white_kb = parent.white_kb;
            child.black_kb = parent.black_kb;

            for k in 0..delta.n_removed {
                let (base_w, base_b) = delta.removed[k];
                let w_idx = w_boff + (base_w ^ w_flip);
                let b_idx = b_boff + (base_b ^ b_flip);
                let fw = &params.feature_weights[w_idx];
                let fb = &params.feature_weights[b_idx];
                vec_sub_i16(&mut child.white, fw, h, simd);
                vec_sub_i16(&mut child.black, fb, h, simd);
            }

            for k in 0..delta.n_added {
                let (base_w, base_b) = delta.added[k];
                let w_idx = w_boff + (base_w ^ w_flip);
                let b_idx = b_boff + (base_b ^ b_flip);
                let fw = &params.feature_weights[w_idx];
                let fb = &params.feature_weights[b_idx];
                vec_add_i16(&mut child.white, fw, h, simd);
                vec_add_i16(&mut child.black, fb, h, simd);
            }
        } else {
            // Unbucketed: original path
            for k in 0..delta.n_removed {
                let (w_idx, b_idx) = delta.removed[k];
                let fw = &params.feature_weights[w_idx];
                let fb = &params.feature_weights[b_idx];
                vec_sub_i16(&mut child.white, fw, h, simd);
                vec_sub_i16(&mut child.black, fb, h, simd);
            }

            for k in 0..delta.n_added {
                let (w_idx, b_idx) = delta.added[k];
                let fw = &params.feature_weights[w_idx];
                let fb = &params.feature_weights[b_idx];
                vec_add_i16(&mut child.white, fw, h, simd);
                vec_add_i16(&mut child.black, fb, h, simd);
            }
        }

        child.computed = true;
    }

    /// Copy parent accumulator (for null moves). Only copies the active hidden
    /// range (`hidden_size`), leaving the unused tail of MAX_HIDDEN untouched —
    /// 33% less memory traffic for H=768 nets.
    pub fn copy_acc(&self, parent: &Accumulator, child: &mut Accumulator) {
        let h = self.params.as_ref().unwrap().hidden_size;
        child.white[..h].copy_from_slice(&parent.white[..h]);
        child.black[..h].copy_from_slice(&parent.black[..h]);
        child.white_kb = parent.white_kb;
        child.black_kb = parent.black_kb;
        child.computed = parent.computed;
    }

    /// Forward pass — dispatches to single-layer or L2 architecture.
    /// Bullet convention: l1 "STM" weights trained with white=[0,384) features (= acc.white),
    /// l1 "NTM" weights trained with black=[0,384) mirrored features (= acc.black).
    /// Always pass (acc.white, acc.black) to match training order; negate for black STM.
    pub fn evaluate(&self, acc: &Accumulator, side: Color, output_bucket: usize) -> i32 {
        let params = self.params.as_ref().unwrap();
        let h = params.hidden_size;

        // Always white-first to match Bullet's fixed STM/NTM encoding
        let white_half = &acc.white[..h];
        let black_half = &acc.black[..h];

        let raw = match &params.arch {
            NnueArch::SingleLayer { output_weights, output_bias } => {
                self.eval_single(white_half, black_half, output_weights, *output_bias, h)
            }
            NnueArch::SingleLayerOutBucket { output_weights, output_biases } => {
                let bucket = output_bucket.min(NUM_OUTPUT_BUCKETS - 1);
                let off = bucket * 2 * h;
                let w = &output_weights[off..off + 2 * h];
                self.eval_single(white_half, black_half, w, output_biases[bucket], h)
            }
            NnueArch::L2Layer { l1_weights, l1_biases, l2_weights, l2_bias } => {
                self.eval_l2(white_half, black_half, l1_weights, l1_biases, l2_weights, *l2_bias)
            }
        };

        // Network always computes white-relative score; negate for black STM
        if side == WHITE { raw } else { -raw }
    }

    /// Single-layer: SCReLU(acc) → dot product → centipawns
    #[inline]
    fn eval_single(&self, stm: &[i16], nstm: &[i16],
                   output_weights: &[i16], output_bias: i16, h: usize) -> i32 {
        let mut output = screlu_dot(stm, nstm, output_weights, h, self.simd);

        // SCReLU output is QA² scale, weights are QB scale → product at QA²*QB
        // Divide by QA → QA*QB scale
        output /= QA as i64;
        output += output_bias as i64;

        ((output * NNUE_SCALE as i64) / (QA as i64 * QB as i64)) as i32
    }

    /// L2 architecture: SCReLU(acc) → L1(1024→32) → SCReLU → L2(32→1) → centipawns
    ///
    /// Quantization flow:
    ///   L0 acc:       i16 at QA scale
    ///   SCReLU(L0):   i32 at QA² scale
    ///   L1 dot:       i64 at QA²*QB scale → /QA → QA*QB scale → +bias(QA*QB)
    ///   Rescale:      /QB → QA scale (i16 range)
    ///   SCReLU(L1):   i32 at QA² scale (same pattern as L0!)
    ///   L2 dot:       i64 at QA²*QB scale → /QA → QA*QB scale → +bias(QA*QB)
    ///   Final:        * SCALE / (QA*QB) → centipawns
    #[inline]
    fn eval_l2(&self, stm: &[i16], nstm: &[i16],
               l1_weights: &[[i16; L2_SIZE]], l1_biases: &[i16; L2_SIZE],
               l2_weights: &[i16; L2_SIZE], l2_bias: i16) -> i32 {
        let _ = self; // L2 path stays scalar for now (not used by current bucketed nets)

        // ---- L1: (1024 → L2_SIZE) ----
        // Input-major loop: compute SCReLU ONCE per input, accumulate across all outputs.
        // This is 32x fewer SCReLU calls and cache-friendly for l1_weights[i][0..L2_SIZE].
        let mut l1_sums = [0i64; L2_SIZE];

        // STM half
        let h = stm.len();
        for i in 0..h {
            let s = screlu(stm[i]) as i64;
            let w = &l1_weights[i];
            for j in 0..L2_SIZE {
                l1_sums[j] += s * w[j] as i64;
            }
        }
        // NSTM half
        for i in 0..h {
            let s = screlu(nstm[i]) as i64;
            let w = &l1_weights[h + i];
            for j in 0..L2_SIZE {
                l1_sums[j] += s * w[j] as i64;
            }
        }

        // ---- L2: (L2_SIZE → 1) ----
        // Normalize L1, apply SCReLU, and dot with L2 weights in one pass
        let mut output: i64 = 0;
        for j in 0..L2_SIZE {
            // QA²*QB → /QA → QA*QB, + bias(QA*QB), /QB → QA scale
            let sum = l1_sums[j] / QA as i64 + l1_biases[j] as i64;
            let rescaled = (sum / QB as i64).clamp(-32768, 32767) as i16;
            output += screlu(rescaled) as i64 * l2_weights[j] as i64;
        }

        // QA²*QB → /QA → QA*QB, + bias at QA*QB, scale to centipawns
        output /= QA as i64;
        output += l2_bias as i64;

        ((output * NNUE_SCALE as i64) / (QA as i64 * QB as i64)) as i32
    }
}

// =============================================================================
// SIMD: AVX-512 / AVX2 fast paths for the NNUE hot loops.
//
// Bit-identical to the scalar implementation: integer arithmetic, same
// operation order per output lane. The horizontal sum at the end of
// `screlu_dot` is the only place reduction order matters, and since
// integer addition is associative no value diverges from the scalar path.
// =============================================================================

#[inline(always)]
fn vec_add_i16(dst: &mut [i16], src: &[i16], h: usize, simd: SimdImpl) {
    debug_assert!(dst.len() >= h && src.len() >= h);
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe { vec_add_i16_avx512(dst, src, h) },
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx2 => unsafe { vec_add_i16_avx2(dst, src, h) },
        _ => {
            for j in 0..h { dst[j] += src[j]; }
        }
    }
}

#[inline(always)]
fn vec_sub_i16(dst: &mut [i16], src: &[i16], h: usize, simd: SimdImpl) {
    debug_assert!(dst.len() >= h && src.len() >= h);
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe { vec_sub_i16_avx512(dst, src, h) },
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx2 => unsafe { vec_sub_i16_avx2(dst, src, h) },
        _ => {
            for j in 0..h { dst[j] -= src[j]; }
        }
    }
}

/// Fused `dst = parent - rm + add` in a single pass (no copy, no child reload).
/// Replaces the copy+sub+add 3-pass sequence for a simple quiet move. The op order
/// (sub then add) mirrors the unfused path's intermediate exactly, so the result —
/// and any i16 wrapping — is bit-identical.
#[inline(always)]
fn vec_add_sub_i16(dst: &mut [i16], parent: &[i16], add: &[i16], rm: &[i16], h: usize, simd: SimdImpl) {
    debug_assert!(dst.len() >= h && parent.len() >= h && add.len() >= h && rm.len() >= h);
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe { vec_add_sub_i16_avx512(dst, parent, add, rm, h) },
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx2 => unsafe { vec_add_sub_i16_avx2(dst, parent, add, rm, h) },
        _ => {
            for j in 0..h { dst[j] = parent[j] - rm[j] + add[j]; }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_add_sub_i16_avx512(dst: &mut [i16], parent: &[i16], add: &[i16], rm: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 32;
    for c in 0..chunks {
        let off = c * 32;
        let p = _mm512_loadu_si512(parent.as_ptr().add(off) as *const _);
        let r = _mm512_loadu_si512(rm.as_ptr().add(off) as *const _);
        let a = _mm512_loadu_si512(add.as_ptr().add(off) as *const _);
        let res = _mm512_add_epi16(_mm512_sub_epi16(p, r), a);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut _, res);
    }
    for j in (chunks * 32)..h { dst[j] = parent[j] - rm[j] + add[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_sub_i16_avx2(dst: &mut [i16], parent: &[i16], add: &[i16], rm: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 16;
    for c in 0..chunks {
        let off = c * 16;
        let p = _mm256_loadu_si256(parent.as_ptr().add(off) as *const _);
        let r = _mm256_loadu_si256(rm.as_ptr().add(off) as *const _);
        let a = _mm256_loadu_si256(add.as_ptr().add(off) as *const _);
        let res = _mm256_add_epi16(_mm256_sub_epi16(p, r), a);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut _, res);
    }
    for j in (chunks * 16)..h { dst[j] = parent[j] - rm[j] + add[j]; }
}

/// Fused `dst = parent - rm0 - rm1 + add` in a single pass (capture: moving piece's
/// from-square + the captured piece both removed, moving piece's to-square added).
/// Same op order as the unfused copy+sub+sub+add path, so bit-identical.
#[inline(always)]
fn vec_add_sub_sub_i16(dst: &mut [i16], parent: &[i16], add: &[i16], rm0: &[i16], rm1: &[i16], h: usize, simd: SimdImpl) {
    debug_assert!(dst.len() >= h && parent.len() >= h && add.len() >= h && rm0.len() >= h && rm1.len() >= h);
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe { vec_add_sub_sub_i16_avx512(dst, parent, add, rm0, rm1, h) },
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx2 => unsafe { vec_add_sub_sub_i16_avx2(dst, parent, add, rm0, rm1, h) },
        _ => {
            for j in 0..h { dst[j] = parent[j] - rm0[j] - rm1[j] + add[j]; }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_add_sub_sub_i16_avx512(dst: &mut [i16], parent: &[i16], add: &[i16], rm0: &[i16], rm1: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 32;
    for c in 0..chunks {
        let off = c * 32;
        let p = _mm512_loadu_si512(parent.as_ptr().add(off) as *const _);
        let r0 = _mm512_loadu_si512(rm0.as_ptr().add(off) as *const _);
        let r1 = _mm512_loadu_si512(rm1.as_ptr().add(off) as *const _);
        let a = _mm512_loadu_si512(add.as_ptr().add(off) as *const _);
        let res = _mm512_add_epi16(_mm512_sub_epi16(_mm512_sub_epi16(p, r0), r1), a);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut _, res);
    }
    for j in (chunks * 32)..h { dst[j] = parent[j] - rm0[j] - rm1[j] + add[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_sub_sub_i16_avx2(dst: &mut [i16], parent: &[i16], add: &[i16], rm0: &[i16], rm1: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 16;
    for c in 0..chunks {
        let off = c * 16;
        let p = _mm256_loadu_si256(parent.as_ptr().add(off) as *const _);
        let r0 = _mm256_loadu_si256(rm0.as_ptr().add(off) as *const _);
        let r1 = _mm256_loadu_si256(rm1.as_ptr().add(off) as *const _);
        let a = _mm256_loadu_si256(add.as_ptr().add(off) as *const _);
        let res = _mm256_add_epi16(_mm256_sub_epi16(_mm256_sub_epi16(p, r0), r1), a);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut _, res);
    }
    for j in (chunks * 16)..h { dst[j] = parent[j] - rm0[j] - rm1[j] + add[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_add_i16_avx512(dst: &mut [i16], src: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 32;
    for c in 0..chunks {
        let off = c * 32;
        let a = _mm512_loadu_si512(dst.as_ptr().add(off) as *const _);
        let b = _mm512_loadu_si512(src.as_ptr().add(off) as *const _);
        let r = _mm512_add_epi16(a, b);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut _, r);
    }
    for j in (chunks * 32)..h { dst[j] += src[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn vec_sub_i16_avx512(dst: &mut [i16], src: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 32;
    for c in 0..chunks {
        let off = c * 32;
        let a = _mm512_loadu_si512(dst.as_ptr().add(off) as *const _);
        let b = _mm512_loadu_si512(src.as_ptr().add(off) as *const _);
        let r = _mm512_sub_epi16(a, b);
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut _, r);
    }
    for j in (chunks * 32)..h { dst[j] -= src[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_i16_avx2(dst: &mut [i16], src: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = _mm256_loadu_si256(dst.as_ptr().add(off) as *const _);
        let b = _mm256_loadu_si256(src.as_ptr().add(off) as *const _);
        let r = _mm256_add_epi16(a, b);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut _, r);
    }
    for j in (chunks * 16)..h { dst[j] += src[j]; }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_sub_i16_avx2(dst: &mut [i16], src: &[i16], h: usize) {
    use std::arch::x86_64::*;
    let chunks = h / 16;
    for c in 0..chunks {
        let off = c * 16;
        let a = _mm256_loadu_si256(dst.as_ptr().add(off) as *const _);
        let b = _mm256_loadu_si256(src.as_ptr().add(off) as *const _);
        let r = _mm256_sub_epi16(a, b);
        _mm256_storeu_si256(dst.as_mut_ptr().add(off) as *mut _, r);
    }
    for j in (chunks * 16)..h { dst[j] -= src[j]; }
}

// Refresh fan-in (chunk-major): load biases into a register, accumulate N
// features for that chunk, store dst once per chunk. Saves N-1 dst loads and
// stores compared to the per-feature `vec_add` pattern. Function-preserving:
// integer add is associative, order does not matter.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn refresh_fanin_avx512(
    dst: &mut [i16],
    biases: &[i16],
    weights: &[[i16; MAX_HIDDEN]],
    indices: &[usize],
    h: usize,
) {
    use std::arch::x86_64::*;
    let chunks = h / 32;
    for c in 0..chunks {
        let off = c * 32;
        let mut acc = _mm512_loadu_si512(biases.as_ptr().add(off) as *const _);
        for &i in indices {
            let v = _mm512_loadu_si512(weights[i].as_ptr().add(off) as *const _);
            acc = _mm512_add_epi16(acc, v);
        }
        _mm512_storeu_si512(dst.as_mut_ptr().add(off) as *mut _, acc);
    }
    // Tail (h not divisible by 32) — scalar.
    let tail_start = chunks * 32;
    if tail_start < h {
        dst[tail_start..h].copy_from_slice(&biases[tail_start..h]);
        for &i in indices {
            for j in tail_start..h { dst[j] += weights[i][j]; }
        }
    }
}

#[inline]
fn refresh_fanin(
    dst: &mut [i16],
    biases: &[i16],
    weights: &[[i16; MAX_HIDDEN]],
    indices: &[usize],
    h: usize,
    simd: SimdImpl,
) {
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe {
            refresh_fanin_avx512(dst, biases, weights, indices, h)
        },
        _ => {
            // Generic fallback: bias, then add each feature. Matches the
            // existing scalar / AVX-2 refresh path bit-for-bit (assoc. add).
            dst[..h].copy_from_slice(&biases[..h]);
            for &i in indices {
                vec_add_i16(dst, &weights[i], h, simd);
            }
        }
    }
}

// ---- SCReLU dot product ----
//
// Computes sum_i (clamp(stm[i], 0, QA))^2 * w[i] + sum_i (clamp(nstm[i], 0, QA))^2 * w[h+i]
// as one i64. Used by `eval_single` / `eval_l2`.
//
// Range analysis (matches scalar `screlu` + `output_weights` i16):
//   clip in [0, QA=255]                       → fits i16
//   clip^2 in [0, 65025]                      → needs i32
//   clip^2 * w in ~[-2.13e9, 2.13e9]           → fits i32 (just under 2^31)
//   sum of 2*MAX_HIDDEN such terms in i64
//
// We expand to i32, multiply with mullo_epi32, then widen to i64 to accumulate
// — this is the safest path given clrsrc's QA=255 / QB=64 quantization.

#[inline]
fn screlu_dot(stm: &[i16], nstm: &[i16], weights: &[i16], h: usize, simd: SimdImpl) -> i64 {
    match simd {
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx512 => unsafe { screlu_dot_avx512(stm, nstm, weights, h) },
        #[cfg(target_arch = "x86_64")]
        SimdImpl::Avx2 => unsafe { screlu_dot_avx2(stm, nstm, weights, h) },
        _ => {
            let mut output: i64 = 0;
            for i in 0..h {
                output += screlu(stm[i]) as i64 * weights[i] as i64;
                output += screlu(nstm[i]) as i64 * weights[h + i] as i64;
            }
            output
        }
    }
}

// Stockfish-style "madd" SCReLU dot product, valid when clip*weight fits in i16.
// For this network the analysis (max(|w|)=127, clip<=255 -> clip*w<=32385) is well
// inside i16 range, so the result is bit-identical to the i32-expansion path.
//
// Pattern per pair (i, i+1):
//   prod_16 = mullo_epi16(clip, w)          // i16 (no overflow)
//   pair_i32 = madd_epi16(clip, prod_16)    // i32: clip[i]*prod[i] + clip[i+1]*prod[i+1]
//                                            //      = clip[i]²·w[i] + clip[i+1]²·w[i+1]
//
// We use two i32 accumulators (one per half) to break the dependency chain,
// then widen to i64 at the end.

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512bw")]
unsafe fn screlu_dot_avx512(stm: &[i16], nstm: &[i16], weights: &[i16], h: usize) -> i64 {
    use std::arch::x86_64::*;

    let zero = _mm512_setzero_si512();
    let qa = _mm512_set1_epi16(QA as i16);

    let chunks = h / 32;

    // Two i32 accumulators (16 lanes each) — break dependency between STM/NSTM
    // and between odd/even chunks. Each i32 lane accumulates up to ~chunks*4.26M
    // pair sums; for h<=1024 chunks<=32 and the max sum stays well below 2^31.
    let mut acc_stm = _mm512_setzero_si512();
    let mut acc_nstm = _mm512_setzero_si512();

    for c in 0..chunks {
        let off = c * 32;
        // STM
        let v = _mm512_loadu_si512(stm.as_ptr().add(off) as *const _);
        let w = _mm512_loadu_si512(weights.as_ptr().add(off) as *const _);
        let clip = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
        let prod16 = _mm512_mullo_epi16(clip, w);
        let pair = _mm512_madd_epi16(clip, prod16);
        acc_stm = _mm512_add_epi32(acc_stm, pair);

        // NSTM
        let v = _mm512_loadu_si512(nstm.as_ptr().add(off) as *const _);
        let w = _mm512_loadu_si512(weights.as_ptr().add(h + off) as *const _);
        let clip = _mm512_min_epi16(_mm512_max_epi16(v, zero), qa);
        let prod16 = _mm512_mullo_epi16(clip, w);
        let pair = _mm512_madd_epi16(clip, prod16);
        acc_nstm = _mm512_add_epi32(acc_nstm, pair);
    }

    // Widen i32 -> i64 and reduce. Must widen first: 16 i32 lanes each up to
    // ~2^31 could sum to ~2^35, overflowing i32 reduce. Single i64 reduce
    // intrinsic stays in registers (no stack spill / iter().sum()).
    let acc_total_lo = _mm512_add_epi32(acc_stm, acc_nstm);
    let total_lo = _mm512_cvtepi32_epi64(_mm512_castsi512_si256(acc_total_lo));
    let total_hi = _mm512_cvtepi32_epi64(_mm512_extracti64x4_epi64::<1>(acc_total_lo));
    let total = _mm512_add_epi64(total_lo, total_hi);
    let mut out: i64 = _mm512_reduce_add_epi64(total);

    // Scalar tail
    for j in (chunks * 32)..h {
        out += screlu(stm[j]) as i64 * weights[j] as i64;
        out += screlu(nstm[j]) as i64 * weights[h + j] as i64;
    }
    out
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn screlu_dot_avx2(stm: &[i16], nstm: &[i16], weights: &[i16], h: usize) -> i64 {
    use std::arch::x86_64::*;

    let zero = _mm256_setzero_si256();
    let qa = _mm256_set1_epi16(QA as i16);
    let mut acc_stm = _mm256_setzero_si256();  // 8 i32 lanes
    let mut acc_nstm = _mm256_setzero_si256();

    let chunks = h / 16;

    for c in 0..chunks {
        let off = c * 16;
        // STM
        let v = _mm256_loadu_si256(stm.as_ptr().add(off) as *const _);
        let w = _mm256_loadu_si256(weights.as_ptr().add(off) as *const _);
        let clip = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        let prod16 = _mm256_mullo_epi16(clip, w);
        let pair = _mm256_madd_epi16(clip, prod16);
        acc_stm = _mm256_add_epi32(acc_stm, pair);

        // NSTM
        let v = _mm256_loadu_si256(nstm.as_ptr().add(off) as *const _);
        let w = _mm256_loadu_si256(weights.as_ptr().add(h + off) as *const _);
        let clip = _mm256_min_epi16(_mm256_max_epi16(v, zero), qa);
        let prod16 = _mm256_mullo_epi16(clip, w);
        let pair = _mm256_madd_epi16(clip, prod16);
        acc_nstm = _mm256_add_epi32(acc_nstm, pair);
    }

    // Widen and sum
    let acc_total = _mm256_add_epi32(acc_stm, acc_nstm);
    let total_lo = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(acc_total));
    let total_hi = _mm256_cvtepi32_epi64(_mm256_extracti128_si256::<1>(acc_total));
    let total = _mm256_add_epi64(total_lo, total_hi);

    let mut tmp = [0i64; 4];
    _mm256_storeu_si256(tmp.as_mut_ptr() as *mut _, total);
    let mut out: i64 = tmp.iter().sum();

    for j in (chunks * 16)..h {
        out += screlu(stm[j]) as i64 * weights[j] as i64;
        out += screlu(nstm[j]) as i64 * weights[h + j] as i64;
    }
    out
}
