import json
import numpy as np
import chess
import chess.gaviota
import onnxruntime as ort
from core.utils import resource_path
from core.minimax import minimax
from core.transposition_table import LRUCache, ZobristBoard
from core.gaviota import get_move_from_table
from training.policy_network.data_manager import board_to_tensor, index_to_move

CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = resource_path("models/policy_network/CN2_BN2_RLROP.onnx")
TB_DIR = resource_path("tablebases/gaviota")
MOVE_MAPPING = resource_path("models/policy_network/move_mapping.json")

transposition_table = LRUCache(maxsize=100000)

with open(MOVE_MAPPING, "r") as f:
    int_to_move = json.load(f)

try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    print(f"Błąd ładowania modelu ONNX: {e}")
    ort_session = None

def softmax(x):
    """
    Implementacja softmax na bibliotece NumPy zamiast torch.softmax.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_move_with_confidence(board: chess.Board):
    """
    Zwraca przewidziany ruch i confidence używając ONNX Runtime.
    """
    if ort_session is None:
        return None, None

    X_input = board_to_tensor(board)

    if hasattr(X_input, 'cpu'):
        X_input = X_input.cpu().detach().numpy()
    elif not isinstance(X_input, np.ndarray):
        X_input = np.array(X_input)

    X_input = X_input.astype(np.float32)

    try:
        logits = ort_session.run(None, {input_name: X_input})[0]
        
        probabilities = softmax(logits.squeeze(0))
        
        sorted_indices = np.argsort(probabilities)[::-1]
        legal_moves = list(board.legal_moves)

        for idx in sorted_indices:
            move_idx_str = str(idx)
            if move_idx_str in int_to_move:
                predicted_uci = index_to_move(int(int_to_move[move_idx_str])).uci()
                move = chess.Move.from_uci(predicted_uci)
                
                if move in legal_moves:
                    return move, probabilities[idx]

    except Exception as e:
        print(f"Błąd inferencji: {e}") 
        pass

    return None, None

def engine_select(board_obj, white_to_move, depth, start_time=None, time_limit=None):
    """
    Wybiera ruch na podstawie tabeli końcówek Gaviota, modelu ONNX i algorytmu minimax.
    """
    try:
        with chess.gaviota.open_tablebase(TB_DIR) as tb:
            score, nodes, best_move = get_move_from_table(board_obj, tb)
            if best_move is not None:
                return score, nodes, best_move
    except Exception:
        pass

    model_move, confidence = predict_move_with_confidence(board_obj)
    
    if model_move is not None and confidence is not None and confidence >= CONFIDENCE_THRESHOLD:
        reduced_depth = max(1, depth // 2)
        score, nodes, best_move = minimax(board_obj, reduced_depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
        
        if best_move == model_move:
            return score, nodes, best_move
    
    return minimax(board_obj, depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)