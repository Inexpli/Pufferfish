import os
from chess import pgn
import numpy as np
import chess
from tqdm.notebook import tqdm
import torch

piece_to_idx = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_ndarray(board) -> np.ndarray:
    """ 
    Zamienia szachownicę na tensor 8x8x13
    Tensor ma 13 kanałów:
    - 6 kanałów dla białych figur (0-5)
    - 6 kanałów dla czarnych figur (6-11)
    - 1 kanał dla legalnych ruchów (12)
    Każdy kanał jest macierzą 8x8, gdzie 1 oznacza obecność figury lub ruchu, a 0 brak figury lub ruchu
    """
    tensor = np.zeros((8, 8, 13), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_idx[piece.piece_type] + (0 if piece.color else 6)
        tensor[row, col, idx] = 1.0

    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        tensor[7 - (from_sq // 8), from_sq % 8, 12] = 1.0
        tensor[7 - (to_sq // 8), to_sq % 8, 12] = 1.0
    return tensor

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Zamienia szachownicę na tensor [1, 13, 8, 8].
    """
    ndarray = board_to_ndarray(board)  # (8, 8, 13)
    tensor = torch.tensor(ndarray, dtype=torch.float32).permute(2, 0, 1)  # (13, 8, 8)
    return tensor.unsqueeze(0)  # (1, 13, 8, 8)

def move_to_index(move):
    """
    Zamienia ruch na indeks 0-4095
    """
    from_sq = move.from_square
    to_sq = move.to_square      
    index = from_sq * 64 + to_sq
    return index


def index_to_move(index):
    """
    Zamienia indeks 0-4095 z powrotem na ruch
    """
    from_sq = index // 64
    to_sq = index % 64
    return chess.Move(from_sq, to_sq)

def get_data_shapes(X,y):
    """
    Zwraca kształty danych X i y,
    przydaje się do sprawdzenia poprawności danych
    """
    return {
    "X_shape": np.array(X).shape,
    "y_shape": np.array(y).shape,
    "X_dtype": np.array(X).dtype,
    "y_dtype": np.array(y).dtype,
    "X_min": np.min(X),
    "X_max": np.max(X),
    "y_min": np.min(y),
    "y_max": np.max(y),
    "y_unique": len(np.unique(y)),
    }

def get_num_indexes(y):
    """
    Zwraca liczbę unikalnych indeksów ruchów w zbiorze danych
    """
    return len(np.unique(y))

def load_games(path="../data/"):
    files = [file for file in os.listdir(path) if file.endswith(".pgn")]
    games = []
    for file in tqdm(files, desc="Przetwarzanie plików PGN"):
        with open(f"{path}/{file}", 'r') as pgn_file:
            while True:
                game = pgn.read_game(pgn_file)
                if game is None:
                    break
                games.append(game)
    return games

def load_data(path="../data/"):
    """
    Wczytuje dane z listy gier i zwraca listę tensorów i etykiet
    """
    X = []  # Cechy
    y = []  # Etykiety
    games = load_games(path)
    
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            board_tensor = board_to_ndarray(board)
            X.append(board_tensor)
            y.append(move_to_index(move))
            board.push(move)
    
    return np.array(X), np.array(y), len(games)