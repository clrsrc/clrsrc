// ---- Position: board representation with make/unmake ----

use crate::types::*;
use crate::bitboard::*;
use crate::zobrist;
use crate::attacks;

pub const STARTPOS_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub struct Position {
    pub pieces: [u64; 6],       // bitboard per piece type
    pub colors: [u64; 2],       // bitboard per color
    pub mailbox: [u8; 64],      // combined piece (0-11) or EMPTY
    pub side: Color,
    pub castling: CastlingRights,
    pub ep_square: Square,      // NO_SQ if none
    pub halfmove_clock: u16,
    pub fullmove: u16,
    pub hash: u64,
    pub pawn_hash: u64,
}

impl Position {
    pub fn new() -> Self {
        Position {
            pieces: [0; 6],
            colors: [0; 2],
            mailbox: [EMPTY; 64],
            side: WHITE,
            castling: CastlingRights::NONE,
            ep_square: NO_SQ,
            halfmove_clock: 0,
            fullmove: 1,
            hash: 0,
            pawn_hash: 0,
        }
    }

    pub fn startpos() -> Self {
        Self::from_fen(STARTPOS_FEN).expect("valid startpos FEN")
    }

    // ---- Piece manipulation ----

    fn put_piece(&mut self, color: Color, pt: PieceType, sq: Square) {
        let bb = bit(sq);
        self.pieces[pt.index()] |= bb;
        self.colors[color.index()] |= bb;
        self.mailbox[sq as usize] = make_piece(color, pt);
        self.hash ^= zobrist::piece_key(color, pt, sq);
        if pt == PAWN {
            self.pawn_hash ^= zobrist::piece_key(color, pt, sq);
        }
    }

    fn remove_piece(&mut self, sq: Square) {
        let piece = self.mailbox[sq as usize];
        debug_assert!(piece != EMPTY);
        let color = piece_color(piece);
        let pt = piece_type(piece);
        let bb = bit(sq);
        self.pieces[pt.index()] ^= bb;
        self.colors[color.index()] ^= bb;
        self.mailbox[sq as usize] = EMPTY;
        self.hash ^= zobrist::piece_key(color, pt, sq);
        if pt == PAWN {
            self.pawn_hash ^= zobrist::piece_key(color, pt, sq);
        }
    }

    fn move_piece(&mut self, from: Square, to: Square) {
        let piece = self.mailbox[from as usize];
        debug_assert!(piece != EMPTY);
        let color = piece_color(piece);
        let pt = piece_type(piece);
        let from_to = bit(from) | bit(to);
        self.pieces[pt.index()] ^= from_to;
        self.colors[color.index()] ^= from_to;
        self.mailbox[to as usize] = piece;
        self.mailbox[from as usize] = EMPTY;
        self.hash ^= zobrist::piece_key(color, pt, from) ^ zobrist::piece_key(color, pt, to);
        if pt == PAWN {
            self.pawn_hash ^= zobrist::piece_key(color, pt, from) ^ zobrist::piece_key(color, pt, to);
        }
    }

    // ---- Accessors ----

    #[inline(always)]
    pub fn occupancy(&self) -> u64 {
        self.colors[0] | self.colors[1]
    }

    #[inline(always)]
    pub fn us(&self) -> u64 {
        self.colors[self.side.index()]
    }

    #[inline(always)]
    pub fn them(&self) -> u64 {
        self.colors[self.side.flip().index()]
    }

    pub fn king_sq(&self, color: Color) -> Square {
        lsb(self.pieces[KING.index()] & self.colors[color.index()])
    }

    pub fn is_in_check(&self) -> bool {
        let ksq = self.king_sq(self.side);
        if ksq >= 64 { return false; }
        attacks::is_attacked(&self.pieces, &self.colors, ksq, self.side.flip())
    }

    // ---- Make / Unmake ----

    pub fn make_move(&mut self, mv: Move) -> UndoInfo {
        let from = mv.from();
        let to = mv.to();
        let flags = mv.flags();
        let us = self.side;
        let them = us.flip();

        // Save undo info
        let undo = UndoInfo {
            captured: self.mailbox[to as usize],
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
        };

        // Remove old EP from hash (castling XOR deferred — most moves don't change rights)
        if self.ep_square != NO_SQ {
            self.hash ^= zobrist::ep_key(file_of(self.ep_square));
        }
        let old_castling = self.castling;

        // Reset EP
        self.ep_square = NO_SQ;
        self.halfmove_clock += 1;

        let moving_piece = self.mailbox[from as usize];
        let pt = piece_type(moving_piece);

        // Pawn move resets clock
        if pt == PAWN {
            self.halfmove_clock = 0;
        }

        match flags {
            Move::FLAG_QUIET => {
                self.move_piece(from, to);
            }
            Move::FLAG_DOUBLE_PAWN => {
                self.move_piece(from, to);
                // Set EP square (one rank behind the pawn)
                self.ep_square = if us == WHITE { to - 8 } else { to + 8 };
            }
            Move::FLAG_CAPTURE => {
                self.halfmove_clock = 0;
                self.remove_piece(to); // capture
                self.move_piece(from, to);
            }
            Move::FLAG_EP => {
                self.halfmove_clock = 0;
                // Captured pawn is on the same rank as `from`, same file as `to`
                let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
                self.remove_piece(cap_sq);
                self.move_piece(from, to);
            }
            Move::FLAG_KING_CASTLE => {
                // Move king
                self.move_piece(from, to);
                // Move rook
                if us == WHITE {
                    self.move_piece(H1, F1);
                } else {
                    self.move_piece(H8, F8);
                }
            }
            Move::FLAG_QUEEN_CASTLE => {
                self.move_piece(from, to);
                if us == WHITE {
                    self.move_piece(A1, D1);
                } else {
                    self.move_piece(A8, D8);
                }
            }
            _ if mv.is_promotion() => {
                // Promotion (with or without capture)
                if mv.is_capture() {
                    self.halfmove_clock = 0;
                    self.remove_piece(to);
                }
                // Remove pawn from `from`
                self.remove_piece(from);
                // Place promoted piece at `to`
                self.put_piece(us, mv.promo_piece_type(), to);
            }
            _ => {}
        }

        // Update castling rights
        self.castling.0 &= CASTLING_MASK[from as usize] & CASTLING_MASK[to as usize];

        // XOR castling key only when rights actually changed (~70% of moves don't)
        if self.castling != old_castling {
            self.hash ^= zobrist::castling_key(old_castling);
            self.hash ^= zobrist::castling_key(self.castling);
        }
        if self.ep_square != NO_SQ {
            self.hash ^= zobrist::ep_key(file_of(self.ep_square));
        }

        // Switch side
        self.side = them;
        self.hash ^= zobrist::side_key();

        if us == BLACK {
            self.fullmove += 1;
        }

        undo
    }

    pub fn unmake_move(&mut self, mv: Move, undo: UndoInfo) {
        let to = mv.to();
        let from = mv.from();
        let flags = mv.flags();

        // Switch side back
        self.side = self.side.flip();
        let us = self.side;

        if us == BLACK {
            self.fullmove -= 1;
        }

        match flags {
            Move::FLAG_QUIET | Move::FLAG_DOUBLE_PAWN => {
                self.move_piece(to, from);
            }
            Move::FLAG_CAPTURE => {
                self.move_piece(to, from);
                // Restore captured piece
                let cap_color = piece_color(undo.captured);
                let cap_pt = piece_type(undo.captured);
                // put_piece updates hash but we'll restore it, so use raw placement
                self.raw_put(cap_color, cap_pt, to, undo.captured);
            }
            Move::FLAG_EP => {
                self.move_piece(to, from);
                let cap_sq = if us == WHITE { to - 8 } else { to + 8 };
                let cap_piece = make_piece(us.flip(), PAWN);
                self.raw_put(us.flip(), PAWN, cap_sq, cap_piece);
            }
            Move::FLAG_KING_CASTLE => {
                self.move_piece(to, from);
                if us == WHITE {
                    self.move_piece(F1, H1);
                } else {
                    self.move_piece(F8, H8);
                }
            }
            Move::FLAG_QUEEN_CASTLE => {
                self.move_piece(to, from);
                if us == WHITE {
                    self.move_piece(D1, A1);
                } else {
                    self.move_piece(D8, A8);
                }
            }
            _ if mv.is_promotion() => {
                // Remove promoted piece
                self.remove_piece_raw(to);
                // Put pawn back at from
                self.raw_put(us, PAWN, from, make_piece(us, PAWN));
                // Restore capture if any
                if mv.is_capture() {
                    let cap_color = piece_color(undo.captured);
                    let cap_pt = piece_type(undo.captured);
                    self.raw_put(cap_color, cap_pt, to, undo.captured);
                }
            }
            _ => {}
        }

        // Restore state from undo
        self.castling = undo.castling;
        self.ep_square = undo.ep_square;
        self.halfmove_clock = undo.halfmove_clock;
        self.hash = undo.hash;
        self.pawn_hash = undo.pawn_hash;
    }

    /// Raw placement without hash update (used in unmake, hash is restored from undo)
    fn raw_put(&mut self, color: Color, pt: PieceType, sq: Square, piece: u8) {
        let bb = bit(sq);
        self.pieces[pt.index()] |= bb;
        self.colors[color.index()] |= bb;
        self.mailbox[sq as usize] = piece;
    }

    fn remove_piece_raw(&mut self, sq: Square) {
        let piece = self.mailbox[sq as usize];
        let color = piece_color(piece);
        let pt = piece_type(piece);
        let bb = bit(sq);
        self.pieces[pt.index()] ^= bb;
        self.colors[color.index()] ^= bb;
        self.mailbox[sq as usize] = EMPTY;
    }

    // ---- FEN parsing ----

    pub fn from_fen(fen: &str) -> Result<Self, &'static str> {
        let mut pos = Position::new();
        let parts: Vec<&str> = fen.split_whitespace().collect();
        if parts.len() < 4 {
            return Err("FEN needs at least 4 fields");
        }

        // 1. Piece placement
        let mut sq: i8 = 56; // Start at A8
        for ch in parts[0].chars() {
            match ch {
                '/' => {
                    sq -= 16; // Move down one rank
                }
                '1'..='8' => {
                    sq += (ch as i8) - ('0' as i8);
                }
                _ => {
                    if let Some(piece) = piece_from_char(ch) {
                        let color = piece_color(piece);
                        let pt = piece_type(piece);
                        pos.put_piece(color, pt, sq as u8);
                        sq += 1;
                    } else {
                        return Err("Invalid piece character");
                    }
                }
            }
        }

        // 2. Side to move
        pos.side = match parts[1] {
            "w" => WHITE,
            "b" => BLACK,
            _ => return Err("Invalid side to move"),
        };
        if pos.side == BLACK {
            pos.hash ^= zobrist::side_key();
        }

        // 3. Castling rights
        if parts[2] != "-" {
            for ch in parts[2].chars() {
                match ch {
                    'K' => pos.castling.0 |= CastlingRights::WK,
                    'Q' => pos.castling.0 |= CastlingRights::WQ,
                    'k' => pos.castling.0 |= CastlingRights::BK,
                    'q' => pos.castling.0 |= CastlingRights::BQ,
                    _ => {}
                }
            }
        }
        pos.hash ^= zobrist::castling_key(pos.castling);

        // 4. En passant
        if parts[3] != "-" {
            if let Some(ep) = sq_from_str(parts[3]) {
                pos.ep_square = ep;
                pos.hash ^= zobrist::ep_key(file_of(ep));
            }
        }

        // 5 & 6. Halfmove clock and fullmove number
        if parts.len() > 4 {
            pos.halfmove_clock = parts[4].parse().unwrap_or(0);
        }
        if parts.len() > 5 {
            pos.fullmove = parts[5].parse().unwrap_or(1);
        }

        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut fen = String::with_capacity(80);

        // Piece placement
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let piece = self.mailbox[sq(file, rank) as usize];
                if piece == EMPTY {
                    empty += 1;
                } else {
                    if empty > 0 {
                        fen.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    fen.push(PIECE_CHARS[piece as usize] as char);
                }
            }
            if empty > 0 {
                fen.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                fen.push('/');
            }
        }

        // Side to move
        fen.push(' ');
        fen.push(if self.side == WHITE { 'w' } else { 'b' });

        // Castling
        fen.push(' ');
        if self.castling.0 == 0 {
            fen.push('-');
        } else {
            if self.castling.has(CastlingRights::WK) { fen.push('K'); }
            if self.castling.has(CastlingRights::WQ) { fen.push('Q'); }
            if self.castling.has(CastlingRights::BK) { fen.push('k'); }
            if self.castling.has(CastlingRights::BQ) { fen.push('q'); }
        }

        // EP
        fen.push(' ');
        if self.ep_square != NO_SQ {
            fen.push_str(SQ_NAMES[self.ep_square as usize]);
        } else {
            fen.push('-');
        }

        // Halfmove clock + fullmove
        fen.push(' ');
        fen.push_str(&self.halfmove_clock.to_string());
        fen.push(' ');
        fen.push_str(&self.fullmove.to_string());

        fen
    }

    /// Parse a UCI move string (e.g. "e2e4", "a7a8q") and find matching legal move
    pub fn parse_uci_move(&self, s: &str) -> Option<Move> {
        let bytes = s.as_bytes();
        if bytes.len() < 4 { return None; }

        let from = sq_from_str(&s[0..2])?;
        let to = sq_from_str(&s[2..4])?;
        let promo = if bytes.len() > 4 {
            Some(bytes[4])
        } else {
            None
        };

        // Generate all legal moves and find the matching one.
        // Clone so we can call generate_legal (which needs &mut for make/unmake).
        let mut tmp = self.clone();
        let mut list = MoveList::new();
        crate::movegen::generate_legal(&mut tmp, &mut list);

        for i in 0..list.len {
            let mv = list.moves[i];
            if mv.from() == from && mv.to() == to {
                if let Some(p) = promo {
                    if mv.is_promotion() {
                        let expected = match p {
                            b'n' => PieceType::Knight,
                            b'b' => PieceType::Bishop,
                            b'r' => PieceType::Rook,
                            b'q' => PieceType::Queen,
                            _ => return None,
                        };
                        if mv.promo_piece_type() == expected {
                            return Some(mv);
                        }
                    }
                } else if !mv.is_promotion() {
                    return Some(mv);
                }
            }
        }
        None
    }

    #[allow(dead_code)]
    pub fn print(&self) {
        println!();
        for rank in (0..8).rev() {
            print!("  {} ", rank + 1);
            for file in 0..8 {
                let piece = self.mailbox[sq(file, rank) as usize];
                print!("{} ", PIECE_CHARS[piece as usize] as char);
            }
            println!();
        }
        println!("    a b c d e f g h");
        println!("  FEN: {}", self.to_fen());
        println!("  Hash: {:016x}", self.hash);
        println!();
    }

    /// Count of non-pawn material for a color
    pub fn non_pawn_material(&self, color: Color) -> u32 {
        let them = self.colors[color.index()];
        popcount(them & (self.pieces[KNIGHT.index()] | self.pieces[BISHOP.index()]
            | self.pieces[ROOK.index()] | self.pieces[QUEEN.index()]))
    }
}

impl Clone for Position {
    fn clone(&self) -> Self {
        Position {
            pieces: self.pieces,
            colors: self.colors,
            mailbox: self.mailbox,
            side: self.side,
            castling: self.castling,
            ep_square: self.ep_square,
            halfmove_clock: self.halfmove_clock,
            fullmove: self.fullmove,
            hash: self.hash,
            pawn_hash: self.pawn_hash,
        }
    }
}
