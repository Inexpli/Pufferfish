import chess
import torch
from core.utils import resource_path
from core.minimax import minimax
from core.transposition_table import LRUCache, ZobristBoard
from core.gaviota import get_move_from_table
from training.policy_network.data_manager import  board_to_ndarray_with_history, move_to_index, index_to_move
from training.policy_network.model import ChessPolicyNet as Model

CONFIDENCE_THRESHOLD = 0.30
MODEL_PATH = resource_path("models/policy_network/BetaChess.pt")
TB_DIR = resource_path("tablebases/gaviota")
USE_CUDA = False  #torch.cuda.is_available()

device = torch.device("cuda" if USE_CUDA else "cpu")
transposition_table = LRUCache(maxsize=100000)

model = Model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
torch.set_grad_enabled(False)

def predict_move_with_confidence(board: chess.Board, model):
	"""
	Na podstawie szachownicy robi predykcję najlepszego ruchu.
	"""
	tensor = board_to_ndarray_with_history(board)
	tensor = torch.tensor(tensor, dtype=torch.float32)
	tensor = tensor.permute(2, 0, 1)
	tensor = tensor.unsqueeze(0).to(device)

	with torch.no_grad():
		logits = model(tensor)[0]

		# Legalne ruchy
		legal_moves = list(board.legal_moves)
		legal_indices = [move_to_index(mv) for mv in legal_moves]

		# Maskowanie nielegalnych ruchów
		mask = torch.full_like(logits, float('-inf'))
		mask[legal_indices] = 0
		masked_logits = logits + mask

		# Wyznaczenie prawdopodobieństw tylko legalnych ruchów
		legal_logits = masked_logits[legal_indices]
		legal_probs = torch.softmax(legal_logits, dim=0)

		# Najlepszy ruch
		best_idx_within_legal = torch.argmax(legal_probs).item()
		best_move_index = legal_indices[best_idx_within_legal]
		best_move = index_to_move(best_move_index, board)

		# Prawdopodobieństwo najlepszego ruchu
		confidence = legal_probs[best_idx_within_legal].item()

		return best_move, confidence

def engine_select(board_obj, white_to_move, depth, start_time=None, time_limit=None):
    """
    Wybiera ruch na podstawie tabeli końcówek Gaviota, wyuczonego modelu i algorytmu minimax.
    """
    try:
        with chess.gaviota.open_tablebase(TB_DIR) as tb:
            score, nodes, best_move = get_move_from_table(board_obj, tb)
            return score, nodes, best_move
    except Exception:
        pass

    model_move, confidence = predict_move_with_confidence(board_obj, model)
    if model_move is not None and confidence >= CONFIDENCE_THRESHOLD:
        score, nodes, best_move = minimax(board_obj, depth // 2, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
        if best_move == model_move:
            return score, nodes, best_move
    
    return minimax(board_obj, depth, float('-inf'), float('inf'), white_to_move, start_time, time_limit)
