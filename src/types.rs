// ---- Fundamental types for clrsrc chess engine ----

// Square: 0=A1, 1=B1, ..., 7=H1, 8=A2, ..., 63=H8
pub type Square = u8;

pub const NO_SQ: Square = 64;

pub const fn sq(file: u8, rank: u8) -> Square {
    rank * 8 + file
}

pub const fn file_of(s: Square) -> u8 {
    s & 7
}

pub const fn rank_of(s: Square) -> u8 {
    s >> 3
}

pub const fn mirror_sq(s: Square) -> Square {
    s ^ 56
}

// Named squares for convenience
pub const A1: Square = 0;
pub const B1: Square = 1;
pub const C1: Square = 2;
pub const D1: Square = 3;
pub const E1: Square = 4;
pub const F1: Square = 5;
pub const G1: Square = 6;
pub const H1: Square = 7;
pub const A8: Square = 56;
pub const B8: Square = 57;
pub const C8: Square = 58;
pub const D8: Square = 59;
pub const E8: Square = 60;
pub const F8: Square = 61;
pub const G8: Square = 62;
pub const H8: Square = 63;

pub const SQ_NAMES: [&str; 64] = [
    "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
    "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
    "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
    "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
    "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
    "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
    "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
    "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
];

pub fn sq_from_str(s: &str) -> Option<Square> {
    let bytes = s.as_bytes();
    if bytes.len() < 2 { return None; }
    let file = bytes[0].wrapping_sub(b'a');
    let rank = bytes[1].wrapping_sub(b'1');
    if file < 8 && rank < 8 {
        Some(sq(file, rank))
    } else {
        None
    }
}

// ---- Color ----

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }

    pub const fn index(self) -> usize {
        self as usize
    }
}

pub const WHITE: Color = Color::White;
pub const BLACK: Color = Color::Black;
pub const NUM_COLORS: usize = 2;

// ---- PieceType ----

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
    None = 6,
}

pub const PAWN: PieceType = PieceType::Pawn;
pub const KNIGHT: PieceType = PieceType::Knight;
pub const BISHOP: PieceType = PieceType::Bishop;
pub const ROOK: PieceType = PieceType::Rook;
pub const QUEEN: PieceType = PieceType::Queen;
pub const KING: PieceType = PieceType::King;
pub const NUM_PIECE_TYPES: usize = 6;

impl PieceType {
    pub const fn index(self) -> usize {
        self as usize
    }
}

// ---- Mailbox piece encoding ----
// 0..5  = White Pawn..King
// 6..11 = Black Pawn..King
// 12    = Empty

pub const EMPTY: u8 = 12;

pub const fn make_piece(color: Color, pt: PieceType) -> u8 {
    (color as u8) * 6 + (pt as u8)
}

pub const fn piece_color(p: u8) -> Color {
    if p < 6 { Color::White } else { Color::Black }
}

pub const fn piece_type(p: u8) -> PieceType {
    match p % 6 {
        0 => PieceType::Pawn,
        1 => PieceType::Knight,
        2 => PieceType::Bishop,
        3 => PieceType::Rook,
        4 => PieceType::Queen,
        5 => PieceType::King,
        _ => PieceType::None,
    }
}

pub const PIECE_CHARS: [u8; 13] = *b"PNBRQKpnbrqk.";

pub fn piece_from_char(c: char) -> Option<u8> {
    match c {
        'P' => Some(0),
        'N' => Some(1),
        'B' => Some(2),
        'R' => Some(3),
        'Q' => Some(4),
        'K' => Some(5),
        'p' => Some(6),
        'n' => Some(7),
        'b' => Some(8),
        'r' => Some(9),
        'q' => Some(10),
        'k' => Some(11),
        _ => None,
    }
}

// ---- CastlingRights ----
// bit 0: White Kingside
// bit 1: White Queenside
// bit 2: Black Kingside
// bit 3: Black Queenside

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct CastlingRights(pub u8);

impl CastlingRights {
    pub const NONE: CastlingRights = CastlingRights(0);
    pub const WK: u8 = 1;
    pub const WQ: u8 = 2;
    pub const BK: u8 = 4;
    pub const BQ: u8 = 8;

    pub const fn has(self, flag: u8) -> bool {
        self.0 & flag != 0
    }

    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

// Lookup: given a square involved in a move, which castling rights to clear
// E.g., moving/capturing on A1 clears WQ, on H1 clears WK, etc.
pub static CASTLING_MASK: [u8; 64] = {
    let mut mask = [0xFFu8; 64];
    mask[A1 as usize] = 0xFF ^ CastlingRights::WQ;
    mask[E1 as usize] = 0xFF ^ (CastlingRights::WK | CastlingRights::WQ);
    mask[H1 as usize] = 0xFF ^ CastlingRights::WK;
    mask[A8 as usize] = 0xFF ^ CastlingRights::BQ;
    mask[E8 as usize] = 0xFF ^ (CastlingRights::BK | CastlingRights::BQ);
    mask[H8 as usize] = 0xFF ^ CastlingRights::BK;
    mask
};

// ---- Move ----
// 16-bit encoding:
// bits 0-5:   from square
// bits 6-11:  to square
// bits 12-15: flags
//   0  = quiet
//   1  = double pawn push
//   2  = king castle
//   3  = queen castle
//   4  = capture
//   5  = en passant capture
//   8  = knight promotion
//   9  = bishop promotion
//   10 = rook promotion
//   11 = queen promotion
//   12 = knight promo capture
//   13 = bishop promo capture
//   14 = rook promo capture
//   15 = queen promo capture

#[derive(Copy, Clone, PartialEq, Eq, Default)]
pub struct Move(pub u16);

impl Move {
    pub const NULL: Move = Move(0);

    pub const FLAG_QUIET: u16 = 0;
    pub const FLAG_DOUBLE_PAWN: u16 = 1;
    pub const FLAG_KING_CASTLE: u16 = 2;
    pub const FLAG_QUEEN_CASTLE: u16 = 3;
    pub const FLAG_CAPTURE: u16 = 4;
    pub const FLAG_EP: u16 = 5;
    pub const FLAG_KNIGHT_PROMO: u16 = 8;
    pub const FLAG_BISHOP_PROMO: u16 = 9;
    pub const FLAG_ROOK_PROMO: u16 = 10;
    pub const FLAG_QUEEN_PROMO: u16 = 11;
    pub const FLAG_KNIGHT_PROMO_CAP: u16 = 12;
    pub const FLAG_BISHOP_PROMO_CAP: u16 = 13;
    pub const FLAG_ROOK_PROMO_CAP: u16 = 14;
    pub const FLAG_QUEEN_PROMO_CAP: u16 = 15;

    pub const fn new(from: Square, to: Square, flags: u16) -> Move {
        Move((from as u16) | ((to as u16) << 6) | (flags << 12))
    }

    pub const fn from(self) -> Square {
        (self.0 & 0x3F) as Square
    }

    pub const fn to(self) -> Square {
        ((self.0 >> 6) & 0x3F) as Square
    }

    pub const fn flags(self) -> u16 {
        self.0 >> 12
    }

    pub const fn is_capture(self) -> bool {
        self.flags() & 4 != 0
    }

    pub const fn is_promotion(self) -> bool {
        self.flags() & 8 != 0
    }

    pub const fn is_castle(self) -> bool {
        self.flags() == Self::FLAG_KING_CASTLE || self.flags() == Self::FLAG_QUEEN_CASTLE
    }

    pub const fn is_ep(self) -> bool {
        self.flags() == Self::FLAG_EP
    }

    pub fn promo_piece_type(self) -> PieceType {
        match self.flags() & 3 {
            0 => PieceType::Knight,
            1 => PieceType::Bishop,
            2 => PieceType::Rook,
            3 => PieceType::Queen,
            _ => unreachable!(),
        }
    }

    pub fn to_uci(self) -> String {
        if self == Move::NULL {
            return "0000".to_string();
        }
        let mut s = String::with_capacity(5);
        s.push_str(SQ_NAMES[self.from() as usize]);
        s.push_str(SQ_NAMES[self.to() as usize]);
        if self.is_promotion() {
            s.push(match self.promo_piece_type() {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => '?',
            });
        }
        s
    }
}

impl std::fmt::Debug for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

impl std::fmt::Display for Move {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_uci())
    }
}

// ---- MoveList (stack-allocated) ----

pub const MAX_MOVES: usize = 256;

pub struct MoveList {
    pub moves: [Move; MAX_MOVES],
    pub len: usize,
}

impl MoveList {
    pub fn new() -> Self {
        MoveList {
            moves: [Move::NULL; MAX_MOVES],
            len: 0,
        }
    }

    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = mv;
        self.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &Move> {
        self.moves[..self.len].iter()
    }
}

// ---- UndoInfo ----

pub struct UndoInfo {
    pub captured: u8,           // mailbox piece value of captured piece (EMPTY if none)
    pub castling: CastlingRights,
    pub ep_square: Square,      // NO_SQ if none
    pub halfmove_clock: u16,
    pub hash: u64,
    pub pawn_hash: u64,
}
