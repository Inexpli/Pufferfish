import time
import chess
from .evaluation import evaluate_board
from .transposition_table import LRUCache, Entry
from .utils import get_stop_flag

# Inicjalizacja tablicy transpozycji
transposition_table = LRUCache(maxsize=100000)

def get_victim_type(position, move):
    '''Zwraca wartość figury zbitej przez dany ruch.'''
    if position.is_en_passant(move):
        return chess.PAWN
    piece = position.piece_at(move.to_square)
    return piece.piece_type if piece else None

def order_moves(position, moves, tt_best_move=None):
    '''Sortuje ruchy według wartości, aby przyspieszyć alfa-beta.'''
    def move_score(move):
        score = 0
        if tt_best_move and move == tt_best_move:
            score += 10000
        victim_type = get_victim_type(position, move)
        if victim_type:
            attacker_type = position.piece_at(move.from_square).piece_type
            score += 10 * victim_type - attacker_type
        if move.promotion:
            score += 5 * move.promotion
        return score
    return sorted(moves, key=move_score, reverse=True)

def minimax(position, depth, alpha, beta, maximizingPlayer, start_time=None, time_limit=None):
    '''Algorytm minimax z przycinaniem alfa-beta i tablicą transpozycji.'''
    nodes = 1
    state_key = position.zobrist_hash
    
    if start_time is None:
        start_time = time.time()

    if get_stop_flag() or (time_limit and (time.time() - start_time) * 1000 >= time_limit):
        return evaluate_board(position), nodes, None

    tt_best_move = None
    if state_key in transposition_table:
        entry = transposition_table[state_key]
        if entry.depth >= depth:
            if entry.flag == 'exact':
                return entry.value, nodes, entry.best_move
            elif entry.flag == 'lowerbound' and entry.value >= beta:
                return entry.value, nodes, entry.best_move
            elif entry.flag == 'upperbound' and entry.value <= alpha:
                return entry.value, nodes, entry.best_move
        tt_best_move = entry.best_move
    
    if depth == 0:
        value = quiescence(position, alpha, beta, maximizingPlayer, start_time=start_time, time_limit=time_limit)
        return value, nodes, None
    
    if position.is_game_over():
        value = evaluate_board(position)
        transposition_table[state_key] = Entry(value, depth, 'exact', None)
        return value, nodes, None
    
    alpha_orig = alpha
    beta_orig = beta
    best_move = None
    
    moves = order_moves(position, list(position.legal_moves), tt_best_move)
    
    if maximizingPlayer:
        maxEval = float('-inf')
        for move in moves:
            position.push(move)
            evaluation, child_nodes, _ = minimax(position, depth-1, alpha, beta, False, start_time, time_limit)
            nodes += child_nodes
            position.pop()
            if evaluation > maxEval:
                maxEval = evaluation
                best_move = move
            alpha = max(alpha, evaluation)
            if beta <= alpha:
                break
        value = maxEval
    else:
        minEval = float('inf')
        for move in moves:
            position.push(move)
            evaluation, child_nodes, _ = minimax(position, depth-1, alpha, beta, True, start_time, time_limit)
            nodes += child_nodes
            position.pop()
            if evaluation < minEval:
                minEval = evaluation
                best_move = move
            beta = min(beta, evaluation)
            if beta <= alpha:
                break
        value = minEval
    
    flag = 'exact'
    if value <= alpha_orig:
        flag = 'upperbound'
    elif value >= beta_orig:
        flag = 'lowerbound'
    
    transposition_table[state_key] = Entry(value, depth, flag, best_move)

    return value, nodes, best_move

def quiescence(position, alpha, beta, maximizingPlayer, qs_depth=0, max_qs_depth=4, start_time=None, time_limit=None):
    '''Kontynuuje wyszukiwanie tylko dla captures, aż do cichej pozycji.'''
    if start_time is None:
        start_time = time.time()

    if get_stop_flag() or (time_limit and (time.time() - start_time) * 1000 >= time_limit):
        return evaluate_board(position)

    if qs_depth > max_qs_depth or position.is_game_over():
        return evaluate_board(position)
    
    stand_pat = evaluate_board(position)
    
    if maximizingPlayer:
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
    else:
        if stand_pat <= alpha:
            return alpha
        beta = min(beta, stand_pat)
    
    captures = [move for move in position.legal_moves if position.piece_at(move.to_square) is not None]
    if not captures:
        return stand_pat
    
    captures = order_moves(position, captures)
    
    if maximizingPlayer:
        for move in captures:
            position.push(move)
            score = quiescence(position, alpha, beta, False, qs_depth + 1, max_qs_depth, start_time, time_limit)
            position.pop()
            if score >= beta:
                return beta
            alpha = max(alpha, score)
    else:
        for move in captures:
            position.push(move)
            score = quiescence(position, alpha, beta, True, qs_depth + 1, max_qs_depth, start_time, time_limit)
            position.pop()
            if score <= alpha:
                return alpha
            beta = min(beta, score)
    
    return alpha if maximizingPlayer else beta