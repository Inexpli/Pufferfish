import chess
from .heuristics import piece_position, piece_values_local

def evaluate_board(position):
    '''Ocena pozycji szachowej.'''
    if position.is_checkmate():
        return -100000 if position.turn == chess.WHITE else 10000
    if position.is_draw():
        return 0

    score = 0
    for square in chess.SQUARES:
        piece = position.piece_at(square)
        if piece:
            pt = piece.piece_type
            val = piece_values_local[pt]
            if piece.color == chess.WHITE:
                mirrored = chess.square_mirror(square)
                score += val + piece_position[pt][mirrored] + len(list(position.attacks(square))) * 10
            else:
                score -= val + piece_position[pt][square] + len(list(position.attacks(square))) * 10

    return score