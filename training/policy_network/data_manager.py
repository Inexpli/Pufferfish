import os
import numpy as np
import chess
import chess.pgn
import pickle
import lmdb

piece_to_idx = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
}

def board_to_ndarray_with_history(board_history: list[chess.Board], history_length: int = 4) -> np.ndarray:
    """
    Zamienia listę ostatnich boardów na tensor 8x8x(17*history_length).

    Na 17 kanałów składa się:
    - 12 kanałów na figury:
        - 6 białych figur
        - 6 czarnych figur
    - 1 kanał na strone wykonującą ruch
    - 4 kanały na prawa do roszady
    
    - 1 globalny kanał przeznaczony do regresji zaniku brzegów planszy
    To razem daje 4x17+1 = 69 kanałów -> (69x8x8)
    """
    if isinstance(board_history, chess.Board):
        board_history = [board_history]

    num_channels_per_board = 17
    tensor = np.zeros((8, 8, num_channels_per_board * history_length + 1), dtype=np.float32)

    # Uzupełnianie brakujących pozycji historią
    # Jeśli lista board_history jest krótsza niż history_length, dopasowujemy
    padded_history = [board_history[0]] * (history_length - len(board_history)) + board_history[-history_length:]

    for h_idx, board in enumerate(padded_history):
        
        # offset kanałów dla tej pozycji w historii
        channel_offset = h_idx * num_channels_per_board

        # 12 kanałów na figury
        for square, piece in board.piece_map().items():
            row = 7 - (square // 8)
            col = square % 8
            idx = piece_to_idx[piece.piece_type] + (0 if piece.color else 6)
            tensor[row, col, channel_offset + idx] = 1.0

        # 1 kanał na stronę wykonującą ruch
        turn = np.ones((8, 8), dtype=np.float32) if board.turn == chess.WHITE else np.zeros((8, 8), dtype=np.float32)
        tensor[:, :, channel_offset + 12] = turn

        # 4 kanały na prawa do roszady
        if board.has_kingside_castling_rights(chess.WHITE):
            tensor[:, :, channel_offset + 13] = 1.0
        if board.has_queenside_castling_rights(chess.WHITE):
            tensor[:, :, channel_offset + 14] = 1.0
        if board.has_kingside_castling_rights(chess.BLACK):
            tensor[:, :, channel_offset + 15] = 1.0
        if board.has_queenside_castling_rights(chess.BLACK):
            tensor[:, :, channel_offset + 16] = 1.0

    # 1 kanał na krańce szachownicy
    tensor[:, :, num_channels_per_board * history_length] = 1.0

    return tensor


# Definicje kierunków ruchów
# 56 typów ruchów: 8 kierunków queen moves (7 odległości każdy) + 8 kierunków knight moves
QUEEN_DIRECTIONS = [
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
]

KNIGHT_MOVES = [
    (-2, -1), (-2, 1),
    (-1, -2), (-1, 2),
    (1, -2), (1, 2),
    (2, -1), (2, 1),
]

# Typy promocji (bez queen, bo queen jest domyślny ruch)
UNDERPROMOTIONS = [
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
]


def fix_promotion(move: chess.Move, board: chess.Board) -> chess.Move:
    """
    Uzupełnia brakującą promocję jeśli pionek doszedł na ostatnią linię.
    Nie zmienia innych ruchów.
    """
    piece = board.piece_at(move.from_square)
    if piece is None:
        return move

    if piece.piece_type != chess.PAWN:
        return move

    to_rank = chess.square_rank(move.to_square)

    # Białe pionki
    if piece.color == chess.WHITE and to_rank == 7:
        if move.promotion is None:
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

    # Czarne pionki
    if piece.color == chess.BLACK and to_rank == 0:
        if move.promotion is None:
            return chess.Move(move.from_square, move.to_square, promotion=chess.QUEEN)

    return move

def move_to_index(move: chess.Move) -> int:
    """
    Konwertuje ruch na indeks 0-4671.
    
    Struktura indeksowania:
    - 64 pola startowe x 73 typy ruchów = 4672 możliwych ruchów
    - 73 typy ruchów: 56 queen-style moves + 8 knight moves + 9 underpromotions (3x3)
    """
    from_square = move.from_square
    to_square = move.to_square
    
    from_row = from_square // 8
    from_col = from_square % 8
    to_row = to_square // 8
    to_col = to_square % 8
    
    delta_row = to_row - from_row
    delta_col = to_col - from_col
    
    # Sprawdzanie promocji (underpomotion)
    if move.promotion is not None and move.promotion != chess.QUEEN:
        # Underpomotion
        # 3 typy promocji x 3 kierunki (forward, capture left, capture right)
        promo_piece_idx = UNDERPROMOTIONS.index(move.promotion)
        
        if delta_col == -1:  # capture left
            direction_idx = 0
        elif delta_col == 0:  # forward
            direction_idx = 1
        elif delta_col == 1:  # capture right
            direction_idx = 2
        else:
            raise ValueError(f"Nieprawidłowy ruch w ramach niedoawansowania: {move}")
        
        move_type = 56 + 8 + (promo_piece_idx * 3 + direction_idx)
    
    # Sprawdzanie ruchu skoczka
    elif (delta_row, delta_col) in KNIGHT_MOVES:
        knight_idx = KNIGHT_MOVES.index((delta_row, delta_col))
        move_type = 56 + knight_idx
    
    # Queen-style moves (wieża, goniec, hetman)
    else:
        # Normalizacja kierunku
        if delta_row != 0:
            dir_row = delta_row // abs(delta_row)
        else:
            dir_row = 0
            
        if delta_col != 0:
            dir_col = delta_col // abs(delta_col)
        else:
            dir_col = 0
        
        try:
            direction_idx = QUEEN_DIRECTIONS.index((dir_row, dir_col))
        except ValueError:
            raise ValueError(f"Nieprawidłowy kierunek ruchu: {move}")
        
        # Obliczanie dystansu (1-7)
        distance = max(abs(delta_row), abs(delta_col))
        
        if distance < 1 or distance > 7:
            raise ValueError(f"Nieprawidłowa odległość ruchu: {move}")
        
        move_type = direction_idx * 7 + (distance - 1)
    
    return from_square * 73 + move_type


def index_to_move(index: int, board: chess.Board = None) -> chess.Move:
    """
    Konwertuje indeks 0-4671 na ruch.
    """
    if index < 0 or index >= 4672:
        raise ValueError(f"Index out of range: {index}")
    
    from_square = index // 73
    move_type = index % 73
    
    from_row = from_square // 8
    from_col = from_square % 8
    
    # Underpromotions (indeksy 64-72)
    if move_type >= 64:
        promo_idx = move_type - 64
        promo_piece_idx = promo_idx // 3
        direction_idx = promo_idx % 3
        
        promotion_piece = UNDERPROMOTIONS[promo_piece_idx]
        
        # Określenie kierunku
        if direction_idx == 0:  # capture left
            delta_col = -1
        elif direction_idx == 1:  # forward
            delta_col = 0
        elif direction_idx == 2:  # capture right
            delta_col = 1
        
        # Promocja zawsze o 1 pole do przodu (lub w tył dla czarnych)
        # Zakładamy, że białe promują do góry (row+1), czarne w dół (row-1)
        if board and not board.turn:  # Czarne
            to_row = from_row - 1
        else:  # Białe lub brak boardu
            to_row = from_row + 1
        
        to_col = from_col + delta_col
        to_square = to_row * 8 + to_col
        
        return fix_promotion(chess.Move(from_square, to_square, promotion=promotion_piece), board)
    
    # Knight moves (indeksy 56-63)
    elif move_type >= 56:
        knight_idx = move_type - 56
        delta_row, delta_col = KNIGHT_MOVES[knight_idx]
        
        to_row = from_row + delta_row
        to_col = from_col + delta_col
        
        if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
            raise ValueError(f"Skoczek poza krańcami: index={index}")
        
        to_square = to_row * 8 + to_col
        return fix_promotion(chess.Move(from_square, to_square), board)
    
    # Queen-style moves (indeksy 0-55)
    else:
        direction_idx = move_type // 7
        distance = (move_type % 7) + 1
        
        dir_row, dir_col = QUEEN_DIRECTIONS[direction_idx]
        
        to_row = from_row + dir_row * distance
        to_col = from_col + dir_col * distance
        
        if to_row < 0 or to_row > 7 or to_col < 0 or to_col > 7:
            raise ValueError(f"Wyjść poza krańce: index={index}")
        
        to_square = to_row * 8 + to_col
        
        # Sprawdzanie promocji do hetmana
        promotion = None
        if board:
            piece = board.piece_at(from_square)
            if piece and piece.piece_type == chess.PAWN:
                if (piece.color == chess.WHITE and to_row == 7) or \
                   (piece.color == chess.BLACK and to_row == 0):
                    promotion = chess.QUEEN
        
        return fix_promotion(chess.Move(from_square, to_square, promotion=promotion), board)

def write_lmdb(output_path="../../data/lmdb/", pgn_path="../../data/"):
    """
    Zapisuje cechy oraz etykiety z partii PGN do LMDB
    """

    os.makedirs(output_path, exist_ok=True)

    env = lmdb.open(
        output_path,
        map_size=1024 * 1024 * 1024 * 15,  # 15 GB
        subdir=True,
        lock=True,
        readonly=False,
        meminit=False,
        map_async=True
    )

    txn = env.begin(write=True)
    idx = 0
    commit_interval = 10000

    files = [f for f in os.listdir(pgn_path) if f.endswith(".pgn")]
    print("Znaleziono plików PGN:", len(files))

    try:
        for file in files:
            with open(os.path.join(pgn_path, file)) as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    history = []
                    board = game.board()
                    history.append(board.copy())

                    for move in game.mainline_moves():
                        x = board_to_ndarray_with_history(history)
                        y = move_to_index(move)

                        board.push(move)
                        history.append(board.copy())
                        
                        sample = {"x": x.astype(np.float32), "y": y}
                        key = f"sample_{idx}".encode()

                        try:
                            txn.put(key, pickle.dumps(sample))
                        except lmdb.MapFullError:
                            print("Baza danych osiągnęła limit! Kończę zapis.")
                            txn.abort()
                            env.sync()
                            env.close()
                            print("Zapisano rekordów:", idx)
                            return

                        idx += 1

                        if idx % commit_interval == 0:
                            txn.commit()
                            txn = env.begin(write=True)
                            print("Zapisano:", idx)

        txn.commit()
        env.sync()
        env.close()
        print("Zapisano rekordów:", idx)

    except Exception as e:
        try:
            txn.abort()
        except:
            pass
        env.close()
        raise e
