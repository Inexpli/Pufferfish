import json
import os
import sys
import numpy as np
import chess
import torch
from core.minimax import minimax
from core.transposition_table import LRUCache, ZobristBoard
from core.gaviota import get_move_from_table
from training.policy_network.data_manager import board_to_tensor, index_to_move
from training.policy_network.model import Model

def resource_path(rel_path: str) -> str:
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS 
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(base_path, rel_path)

CONFIDENCE_THRESHOLD = 0.7
MODEL_PATH = resource_path("models/policy_network/CN2_BN2_RLROP.pth")
TB_DIR = resource_path("tablebases/gaviota")
MOVE_MAPPING = resource_path("models/policy_network/move_mapping.json")
USE_CUDA = False  #torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")
transposition_table = LRUCache(maxsize=100000)

with open(MOVE_MAPPING, "r") as f:
    int_to_move = json.load(f)

model = Model(len(int_to_move))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
torch.set_grad_enabled(False)

def predict_move_with_confidence(board: chess.Board):
    """
    Zwraca przewidziany ruch i confidence
    """
    X_tensor = board_to_tensor(board).to(device)
    with torch.no_grad():
        logits = model(X_tensor)
    
    probabilities = torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
    sorted_indices = np.argsort(probabilities)[::-1]
    legal_moves = list(board.legal_moves)

    for idx in sorted_indices:
        move_index = int(int_to_move[str(idx)])
        move = chess.Move.from_uci(index_to_move(move_index).uci())
        if move in legal_moves:
            return move, probabilities[idx]

    return None, None

def engine_select(board_obj, white_to_move, depth, start_time=None, time_limit=None):
    """
    Wybiera ruch na podstawie modelu i minimax
    """
    try:
        with chess.gaviota.open_tablebase(TB_DIR) as tb:
            score, nodes, best_move = get_move_from_table(board_obj, tb)
            return score, nodes, best_move
    except Exception:
        pass  # brak tablebase - lecimy dalej

    model_move, confidence = predict_move_with_confidence(board_obj)
    if model_move is not None and confidence >= CONFIDENCE_THRESHOLD:
        score, nodes, best_move = minimax(board_obj, depth // 2, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
        if best_move == model_move:
            return score, nodes, best_move
    
    return minimax(board_obj, depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
