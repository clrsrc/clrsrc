// ---- Zobrist hashing ----
// Deterministic PRNG generates keys at startup.

use crate::types::*;

pub static mut PIECE_KEYS: [[[u64; 64]; NUM_PIECE_TYPES]; NUM_COLORS] = [[[0; 64]; NUM_PIECE_TYPES]; NUM_COLORS];
pub static mut CASTLING_KEYS: [u64; 16] = [0; 16];
pub static mut EP_KEYS: [u64; 8] = [0; 8]; // per file
pub static mut SIDE_KEY: u64 = 0;

/// Simple xorshift64 PRNG (deterministic)
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// Initialize all Zobrist keys. Must be called once at startup.
pub fn init() {
    unsafe {
        let mut rng = Rng(0x2E8B57A1C3D4F6E9); // fixed seed

        for color in 0..NUM_COLORS {
            for pt in 0..NUM_PIECE_TYPES {
                for sq in 0..64 {
                    PIECE_KEYS[color][pt][sq] = rng.next();
                }
            }
        }

        for i in 0..16 {
            CASTLING_KEYS[i] = rng.next();
        }

        for i in 0..8 {
            EP_KEYS[i] = rng.next();
        }

        SIDE_KEY = rng.next();
    }
}

pub fn piece_key(color: Color, pt: PieceType, sq: Square) -> u64 {
    // Implizite Vorbedingung (Kani-Befund 11.06): PieceType::None hat index()==6, aber
    // PIECE_KEYS hat nur NUM_PIECE_TYPES==6 Einträge (0..5) → None wäre OOB. None wird nie
    // gehasht; der debug_assert macht die Annahme explizit (Release-frei, fängt's im Test).
    debug_assert!(pt.index() < NUM_PIECE_TYPES, "piece_key: kein Hash-Key für {:?} (index {} >= {})", pt, pt.index(), NUM_PIECE_TYPES);
    debug_assert!((sq as usize) < 64, "piece_key: Square {} out of range", sq);
    unsafe { PIECE_KEYS[color.index()][pt.index()][sq as usize] }
}

pub fn castling_key(rights: CastlingRights) -> u64 {
    unsafe { CASTLING_KEYS[rights.index()] }
}

pub fn ep_key(file: u8) -> u64 {
    debug_assert!((file as usize) < 8, "ep_key: file {} out of range (EP_KEYS hat 8 Einträge)", file);
    unsafe { EP_KEYS[file as usize] }
}

pub fn side_key() -> u64 {
    unsafe { SIDE_KEY }
}
