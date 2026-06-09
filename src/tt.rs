// ---- Transposition Table ----
// 16-byte entries in 2-way buckets (32 bytes/bucket fits one cache line).
// Aging via generation counter. Lock-free shared access for Lazy SMP.
// Benign data races on entry reads/writes are tolerated — key16 verification
// detects corrupted reads. This is the standard approach in chess engines.

use crate::types::Move;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering as AtomicOrd};

// TT entry flags — low 2 bits = type, bit 2 = was_pv (node was/is in PV path)
pub const FLAG_NONE: u8 = 0;
pub const FLAG_EXACT: u8 = 1;  // PV node, exact score
pub const FLAG_LOWER: u8 = 2;  // Cut node, score >= beta (fail high)
pub const FLAG_UPPER: u8 = 3;  // All node, score <= alpha (fail low)
pub const FLAG_TYPE_MASK: u8 = 0x03;
pub const FLAG_WAS_PV: u8 = 0x04;

const BUCKET_SIZE: usize = 2;

// Depth penalty applied per generation of age during probe.
// Prevents stale TT entries (from earlier ply analyses stored in the TT)
// from triggering false cutoffs when the same position appears as root.
// Tuned empirically; SPRT-candidate before any value change.
// penalty=4 → -15 Elo, penalty=1 → -7 Elo (both regress: tax the common age=1 carry-over).
// Threshold variant: only age >= TT_AGING_THRESHOLD is aged (age=1 = previous move = bulk-neutral).
// Aging inert gestellt (age<=255 < 9999) -> verhaltens-clean = Prod 3D8AC150.
// Threshold-Variante (36f8e198) ist als Binary archiviert, Stefan-geparkt.
const TT_AGING_THRESHOLD: i32 = 9999;
const TT_AGING_DEPTH_PENALTY: i32 = 2;

/// Packed TT entry: 16 bytes
#[derive(Copy, Clone)]
#[repr(C)]
pub struct TTEntry {
    pub key16: u16,
    pub depth: i8,
    pub flag: u8,
    pub score: i16,
    pub eval: i16,
    pub best_move: Move,
    pub gen: u8,
    _padding: [u8; 5],
}

impl TTEntry {
    pub const fn empty() -> Self {
        TTEntry {
            key16: 0,
            depth: -1,
            flag: FLAG_NONE,
            score: 0,
            eval: 0,
            best_move: Move::NULL,
            gen: 0,
            _padding: [0; 5],
        }
    }
}

/// Bucket of 2 entries (32 bytes, fits in one cache line)
#[derive(Copy, Clone)]
#[repr(C)]
struct TTBucket {
    entries: [TTEntry; BUCKET_SIZE],
}

impl TTBucket {
    const fn empty() -> Self {
        TTBucket {
            entries: [TTEntry::empty(); BUCKET_SIZE],
        }
    }
}

/// Thread-safe transposition table using interior mutability.
/// Shared across search threads via Arc<TTable>.
pub struct TTable {
    ptr: *mut TTBucket,
    mask: usize,
    _storage: Vec<TTBucket>,
    gen: AtomicU8,
}

// Safety: TT entries are read/written via raw pointers without locks.
// Benign races are handled by key16 verification on read.
// This is standard practice in all major chess engines.
unsafe impl Send for TTable {}
unsafe impl Sync for TTable {}

pub type SharedTT = Arc<TTable>;

impl TTable {
    /// Create a TT with the given size in megabytes
    pub fn new(mb: usize) -> Self {
        let bytes = mb * 1024 * 1024;
        let bucket_bytes = std::mem::size_of::<TTBucket>(); // 32
        let raw_buckets = bytes / bucket_bytes;
        let num_buckets = if raw_buckets.is_power_of_two() {
            raw_buckets
        } else {
            // Bucket count must be a power of two for mask indexing, so round DOWN.
            // Warn loudly: a non-power-of-two Hash silently wastes up to ~50% otherwise.
            let rounded = (raw_buckets.next_power_of_two() >> 1).max(1);
            let used_mb = rounded * bucket_bytes / (1024 * 1024);
            eprintln!(
                "info string TT: Hash {} MB uses only {} MB ({} MB wasted) — set a power-of-two MB to fill it",
                mb, used_mb, mb.saturating_sub(used_mb)
            );
            rounded
        }.max(1);
        let mut storage = vec![TTBucket::empty(); num_buckets];
        let ptr = storage.as_mut_ptr();
        TTable {
            ptr,
            mask: num_buckets - 1,
            _storage: storage,
            gen: AtomicU8::new(0),
        }
    }

    pub fn new_shared(mb: usize) -> SharedTT {
        Arc::new(Self::new(mb))
    }

    /// Increment generation counter (call at start of each new search)
    pub fn new_generation(&self) {
        self.gen.fetch_add(1, AtomicOrd::Relaxed);
    }

    pub fn current_gen(&self) -> u8 {
        self.gen.load(AtomicOrd::Relaxed)
    }

    #[inline]
    fn bucket_index(&self, hash: u64) -> usize {
        (hash as usize) & self.mask
    }

    #[inline]
    fn key16(hash: u64) -> u16 {
        (hash >> 48) as u16
    }

    /// Probe the TT. Checks both entries in the bucket.
    /// Entries from older generations get a depth penalty to prevent stale
    /// shallow entries (cached from earlier ply analyses) from causing false
    /// cutoffs when they reappear as the root position in a new search.
    pub fn probe(&self, hash: u64) -> Option<TTEntry> {
        let idx = self.bucket_index(hash);
        let key16 = Self::key16(hash);
        let gen = self.current_gen();
        let bucket = unsafe { &*self.ptr.add(idx) };
        for i in 0..BUCKET_SIZE {
            let e = &bucket.entries[i];
            if e.key16 == key16 && (e.flag & FLAG_TYPE_MASK) != FLAG_NONE {
                let age = gen.wrapping_sub(e.gen) as i32;
                if age >= TT_AGING_THRESHOLD {
                    let mut aged = *e;
                    aged.depth = (aged.depth as i32 - TT_AGING_DEPTH_PENALTY * age).max(-1) as i8;
                    return Some(aged);
                }
                return Some(*e);
            }
        }
        None
    }

    /// Store an entry with age-aware replacement. `was_pv` marks the node as PV-path member.
    pub fn store(&self, hash: u64, depth: i8, flag_type: u8, was_pv: bool, score: i16, eval: i16, best_move: Move) {
        let flag = flag_type | if was_pv { FLAG_WAS_PV } else { 0 };
        let idx = self.bucket_index(hash);
        let key16 = Self::key16(hash);
        let gen = self.current_gen();
        let bucket = unsafe { &mut *self.ptr.add(idx) };

        // Find best replacement slot
        let mut replace_idx = 0;
        let mut worst_priority = i32::MAX;

        for i in 0..BUCKET_SIZE {
            let e = &bucket.entries[i];
            // Same position or empty: replace immediately
            if e.key16 == key16 || (e.flag & FLAG_TYPE_MASK) == FLAG_NONE {
                replace_idx = i;
                break;
            }
            // Priority: depth - 8 * age_difference
            let age = gen.wrapping_sub(e.gen) as i32;
            let priority = e.depth as i32 - 8 * age;
            if priority < worst_priority {
                worst_priority = priority;
                replace_idx = i;
            }
        }

        // Replace if: empty, same position, deeper/equal, or different generation
        let existing = &bucket.entries[replace_idx];
        if (existing.flag & FLAG_TYPE_MASK) == FLAG_NONE
            || existing.key16 == key16
            || depth >= existing.depth
            || gen != existing.gen
        {
            bucket.entries[replace_idx] = TTEntry {
                key16, depth, flag, score, eval, best_move, gen,
                _padding: [0; 5],
            };
        }
    }

    /// Total number of entries across all buckets
    pub fn entry_count(&self) -> usize {
        (self.mask + 1) * BUCKET_SIZE
    }

    /// Clear the entire TT
    pub fn clear(&self) {
        let len = self.mask + 1;
        for i in 0..len {
            unsafe { *self.ptr.add(i) = TTBucket::empty(); }
        }
        self.gen.store(0, AtomicOrd::Relaxed);
    }

    #[cfg(target_arch = "x86_64")]
    pub fn prefetch(&self, hash: u64) {
        let index = self.bucket_index(hash);
        unsafe {
            let ptr = self.ptr.add(index) as *const i8;
            core::arch::x86_64::_mm_prefetch(ptr, core::arch::x86_64::_MM_HINT_T0);
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch(&self, _hash: u64) {}

    /// Approximate fill rate (per mille) — only counts current generation
    pub fn hashfull(&self) -> u32 {
        let len = self.mask + 1;
        let sample = len.min(500);
        let gen = self.current_gen();
        let mut used = 0;
        for i in 0..sample {
            let bucket = unsafe { &*self.ptr.add(i) };
            for j in 0..BUCKET_SIZE {
                if bucket.entries[j].flag != FLAG_NONE && bucket.entries[j].gen == gen {
                    used += 1;
                }
            }
        }
        (used * 1000 / (sample.max(1) * BUCKET_SIZE)) as u32
    }
}
