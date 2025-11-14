import os
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import torch
import csv
import chess

import os

piece_to_idx = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Zamienia szachownicę na tensor [1, C, 8, 8].
    C = 18 (6 białe + 6 czarne + legal moves + turn + 4 castling)
    """
    ndarray = board_to_ndarray(board)  # (8, 8, C)
    tensor = torch.tensor(ndarray, dtype=torch.float32).permute(2, 0, 1)  # (C, 8, 8)
    return tensor.unsqueeze(0)  # (1, C, 8, 8)

def board_to_ndarray(board) -> np.ndarray:
    """ 
    Zamienia szachownicę na tensor 8x8x18
    Tensor ma 18 kanałów:
    - 6 kanałów dla białych figur (0-5)
    - 6 kanałów dla czarnych figur (6-11)
    - 1 kanał dla legalnych ruchów (12)
    - 1 kanał dla strony wokonującej ruch (13)
    - 4 kanały dla praw roszady (14-17)
    Każdy kanał jest macierzą 8x8, gdzie 1 oznacza obecność figury lub ruchu, a 0 brak figury lub ruchu
    """

    # 12 kanałów na figury
    tensor = np.zeros((8, 8, 18), dtype=np.float32)
    for square, piece in board.piece_map().items():
        row = 7 - (square // 8)
        col = square % 8
        idx = piece_to_idx[piece.piece_type] + (0 if piece.color else 6)
        tensor[row, col, idx] = 1.0

    # 1 kanał na legalne ruchy
    for move in board.legal_moves:
        from_sq = move.from_square
        to_sq = move.to_square
        tensor[7 - (from_sq // 8), from_sq % 8, 12] = 1.0
        tensor[7 - (to_sq // 8), to_sq % 8, 12] = 1.0

    # 1 kanał na stronę która wykonuje ruch
    turn = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
    tensor[:, :, 13] = turn

    # 4 kanały na prawa do roszady
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[:, :, 14] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[:, :, 15] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[:, :, 16] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[:, :, 17] = 1.0

    return tensor

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

def parse_evaluation(eval_str: str, clip_value: float = 1000.0) -> float | None:
    """
    Normalizuje ocenę pozycji z formatu silnika szachowego na wartość zmiennoprzecinkową w zakresie [-1.0, 1.0]
    """
    if eval_str is None:
        return None
    s = str(eval_str).strip()
    if s == '':
        return None

    if '#' in s:
        sign = -1.0 if ('-' in s and not '+' in s) else 1.0
        return sign * 1.0

    try:
        cp = float(s.replace('+', ''))
    except ValueError:
        return None

    cp = np.clip(cp, -clip_value, clip_value)
    return float(cp / clip_value)

def load_games(limit):
    """
    Wczytuje gry z pliku chessData.csv i zwraca listę obiektów gry
    """
    base_dir = os.path.dirname(__file__)
    path = os.path.join(base_dir, "../../data/chessData.csv")
    path = os.path.abspath(path)
    df = pd.read_csv(path, nrows=limit)
    games = []
    evaluations = []

    for index, row in df.iterrows():
        fen = row['FEN']
        evaluation = parse_evaluation(row['Evaluation'])

        game = chess.Board(fen)
        games.append(game)
        evaluations.append(evaluation)

    return games, evaluations

def load_data(path="../../data/", limit=100000):
    """
    Wczytuje dane z listy gier i zwraca listę tensorów i etykiet
    """
    X = []  # Cechy (ndarray per-position)
    y = []  # Etykiety (float)

    games, evaluations = load_games(limit)

    for game, eval_ in tqdm(zip(games, evaluations), total=len(games), desc="Przetwarzanie pozycji"):
        board_tensor = board_to_ndarray(game)
        X.append(board_tensor)
        y.append(eval_)

    Xnp = np.array(X, dtype=np.float32)
    Ynp = np.array(y, dtype=np.float32)

    return Xnp, Ynp, len(games)