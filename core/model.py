import numpy as np
import chess
import chess.gaviota
import onnxruntime as ort

from core.utils import resource_path
from core.minimax import minimax
from core.transposition_table import LRUCache, ZobristBoard
from core.gaviota import get_move_from_table
from training.policy_network.data_manager import board_to_ndarray_with_history, move_to_index, index_to_move

CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = resource_path("models/policy_network/BetaChess.onnx")
TB_DIR = resource_path("tablebases/gaviota")

transposition_table = LRUCache(maxsize=100000)

try:
    ort_session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = ort_session.get_inputs()[0].name
except Exception as e:
    print(f"Błąd krytyczny: Nie można załadować modelu ONNX z {MODEL_PATH}")
    print(f"Szczegóły: {e}")
    ort_session = None

def softmax(x):
    """
    Stabilna numerycznie funkcja softmax zaimplementowana w NumPy.
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def predict_move_with_confidence(board: chess.Board):
    """
    Wykonuje predykcję używając ONNX Runtime i NumPy.
    """
    if ort_session is None:
        return None, 0.0

    try:
        raw_ndarray = board_to_ndarray_with_history(board)
        transposed = np.transpose(raw_ndarray, (2, 0, 1))
        input_tensor = np.expand_dims(transposed, axis=0).astype(np.float32)
        logits = ort_session.run(None, {input_name: input_tensor})[0]
        logits = np.squeeze(logits)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return None, 0.0

        legal_indices = [move_to_index(mv) for mv in legal_moves]
        legal_logits_values = logits[legal_indices]
        probs = softmax(legal_logits_values)
        best_local_idx = np.argmax(probs)
        best_global_idx = legal_indices[best_local_idx]
        confidence = probs[best_local_idx]
        
        best_move = index_to_move(best_global_idx, board)

        return best_move, confidence

    except Exception as e:
        print(f"Błąd predykcji ONNX: {e}") 
        return None, 0.0

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
    
    if model_move is not None and confidence >= CONFIDENCE_THRESHOLD:
        reduced_depth = max(1, depth // 2)
        score, nodes, best_move = minimax(board_obj, reduced_depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
        
        if best_move == model_move:
            return score, nodes, best_move
    
    return minimax(board_obj, depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)