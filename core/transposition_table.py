import chess
from chess.polyglot import zobrist_hash, POLYGLOT_RANDOM_ARRAY
from collections import OrderedDict, namedtuple

# Klasa z inkrementalnym hashem
class ZobristBoard(chess.Board):
    ''' Szachownica z inkrementalnym hashowaniem Zobrista.'''
    def __init__(self, *args, **kwargs):
        ''' Inicjalizuje szachownicę i oblicza początkowy hash Zobrista.'''
        super().__init__(*args, **kwargs)
        self.zobrist_hash = zobrist_hash(self)
        self.hash_stack = [self.zobrist_hash]

    def _piece_hash(self, piece, square):
        ''' Zwraca wartość haszującą dla danej figury na danym polu.'''
        color_offset = 0 if piece.color == chess.WHITE else 6
        type_offset = piece.piece_type - 1
        return POLYGLOT_RANDOM_ARRAY[64 * (color_offset + type_offset) + square]

    def is_game_over(self):
        ''' Sprawdza, czy gra się zakończyła (mat, pat, niewystarczający materiał, powtórzenie trzykrotne lub reguła 50 posunięć).'''
        return (
            self.is_checkmate()
            or self.is_stalemate()
            or self.can_claim_fifty_moves()
            or self.is_insufficient_material()
            or self.can_claim_threefold_repetition()
            or self.is_fivefold_repetition()
        )
    
    def is_draw(self):
        ''' Sprawdza, czy pozycja jest remisem (pat, niewystarczający materiał, powtórzenie trzykrotne lub reguła 50 posunięć).'''
        return (
            self.is_stalemate()
            or self.can_claim_fifty_moves()
            or self.is_insufficient_material()
            or self.can_claim_threefold_repetition()
            or self.is_fivefold_repetition()
        )

    def push(self, move):
        ''' Wykonuje ruch i aktualizuje hash Zobrista.'''
        h = self.zobrist_hash

        from_square = move.from_square
        to_square = move.to_square
        moved_piece = self.piece_at(from_square)

        # Usuń przesuniętą figurę z pola
        h ^= self._piece_hash(moved_piece, from_square)

        # Zajmij się zdobytą figurą
        captured = self.piece_at(to_square)
        if captured:
            h ^= self._piece_hash(captured, to_square)

        # Dodaj przesunięty element do pola (lub awansuj)
        if move.promotion:
            promoted_piece = chess.Piece(move.promotion, moved_piece.color)
            h ^= self._piece_hash(promoted_piece, to_square)
        else:
            h ^= self._piece_hash(moved_piece, to_square)

        # En passant
        if self.is_en_passant(move):
            ep_square = to_square - 8 if self.turn == chess.WHITE else to_square + 8
            ep_pawn = chess.Piece(chess.PAWN, not self.turn)
            h ^= self._piece_hash(ep_pawn, ep_square)

        # Roszady
        if self.is_castling(move):
            if self.turn == chess.WHITE:
                if to_square == chess.G1:
                    rook_from, rook_to = chess.H1, chess.F1
                else:
                    rook_from, rook_to = chess.A1, chess.D1
            else:
                if to_square == chess.G8:
                    rook_from, rook_to = chess.H8, chess.F8
                else:
                    rook_from, rook_to = chess.A8, chess.D8
            rook = chess.Piece(chess.ROOK, self.turn)
            h ^= self._piece_hash(rook, rook_from)
            h ^= self._piece_hash(rook, rook_to)

        # Zmiana stron
        h ^= POLYGLOT_RANDOM_ARRAY[780]

        # Obsługa starych roszad i en passant
        old_castling_xors = 0
        if self.has_kingside_castling_rights(chess.WHITE):
            old_castling_xors ^= POLYGLOT_RANDOM_ARRAY[768]
        if self.has_queenside_castling_rights(chess.WHITE):
            old_castling_xors ^= POLYGLOT_RANDOM_ARRAY[769]
        if self.has_kingside_castling_rights(chess.BLACK):
            old_castling_xors ^= POLYGLOT_RANDOM_ARRAY[770]
        if self.has_queenside_castling_rights(chess.BLACK):
            old_castling_xors ^= POLYGLOT_RANDOM_ARRAY[771]

        old_ep_xor = 0
        if self.ep_square:
            old_ep_xor ^= POLYGLOT_RANDOM_ARRAY[772 + chess.square_file(self.ep_square)]

        # Zrób ruch
        super().push(move)

        # Obsługa nowych roszad i en passant
        new_castling_xors = 0
        if self.has_kingside_castling_rights(chess.WHITE):
            new_castling_xors ^= POLYGLOT_RANDOM_ARRAY[768]
        if self.has_queenside_castling_rights(chess.WHITE):
            new_castling_xors ^= POLYGLOT_RANDOM_ARRAY[769]
        if self.has_kingside_castling_rights(chess.BLACK):
            new_castling_xors ^= POLYGLOT_RANDOM_ARRAY[770]
        if self.has_queenside_castling_rights(chess.BLACK):
            new_castling_xors ^= POLYGLOT_RANDOM_ARRAY[771]

        new_ep_xor = 0
        if self.ep_square:
            new_ep_xor ^= POLYGLOT_RANDOM_ARRAY[772 + chess.square_file(self.ep_square)]

        h ^= old_castling_xors ^ new_castling_xors
        h ^= old_ep_xor ^ new_ep_xor

        self.zobrist_hash = h
        self.hash_stack.append(h)

    def pop(self):
        ''' Cofa ostatni ruch i aktualizuje hash Zobrista.'''
        move = super().pop()
        self.hash_stack.pop()
        self.zobrist_hash = self.hash_stack[-1]
        return move

# LRU Cache
class LRUCache(OrderedDict):
    ''' Implementacja tablicy transpozycji z usuwaniem najdłużej nieużywanych elementów.'''
    def __init__(self, maxsize):
        self.maxsize = maxsize
        super().__init__()

    def __getitem__(self, key):
        ''' Pobiera element i oznacza go jako najnowszy.'''
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        ''' Ustawia element i usuwa najstarszy, jeśli przekroczono rozmiar.'''
        super().__setitem__(key, value)
        self.move_to_end(key)
        if len(self) > self.maxsize:
            self.popitem(last=False)

Entry = namedtuple('Entry', ['value', 'depth', 'flag', 'best_move'])