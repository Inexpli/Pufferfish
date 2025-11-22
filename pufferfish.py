import sys
import shlex
import time
from typing import Optional
from core.utils import set_stop_flag, get_stop_flag
from core.polyglot import get_opening_book_move
from core.model import engine_select, ZobristBoard, TB_DIR

import chess
import chess.gaviota

uci_running = True

DEFAULT_MAX_DEPTH = 5
DEFAULT_DEPTH = 4

def iterative_deepening(
        board: ZobristBoard,
        white_to_move: bool,
        time_limit_ms: Optional[int] = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        preferred_depth: Optional[int] = None
    ):
    '''
    Wykonuje wyszukiwanie ruchu z iteracyjnym pogłębianiem

    Szuka najlepszego ruchu aż do osiągnięcia limitu czasu lub maksymalnej głębokości
    '''

    best_move = None
    best_val = None
    total_nodes = 0
    start = time.time()
    elapsed_ms = 0
    depth_start = 1

    if preferred_depth is not None and time_limit_ms is None:
        try:
            val, nodes, mv = engine_select(board, white_to_move, preferred_depth)
            total_nodes += nodes
            print(f"info depth {preferred_depth} score cp {int(val * 100)} pv {mv.uci() if mv else '(none)'}")
            return val, total_nodes, mv
        except Exception:
            return None, 0, None

    for depth in range(depth_start, max_depth + 1):
        if get_stop_flag():
            break

        if time_limit_ms is not None:
            elapsed_ms = int((time.time() - start) * 1000)
            if elapsed_ms >= time_limit_ms:
                break

        try:
            t0 = time.time()
            val, nodes, mv = engine_select(board, white_to_move, depth)
            t1 = time.time()
            elapsed_this_depth = int((t1 - t0) * 1000)
            total_nodes += nodes
        except Exception as e:
            print(f"info string minimax exception at depth {depth}: {e}")
            break

        if mv is not None:
            best_move = mv
            best_val = val

        if best_move is not None:
            print(f"info depth {depth} time {elapsed_this_depth} nodes {nodes} score cp {int(best_val * 100)} pv {best_move.uci()}")

        if time_limit_ms is not None:
            elapsed_ms = int((time.time() - start) * 1000)
            if elapsed_ms >= time_limit_ms * 0.9:
                break

        if get_stop_flag():
            break

        if preferred_depth is not None and depth >= preferred_depth:
            break

    return best_val, total_nodes, best_move

def compute_movetime_from_clock(
        board: ZobristBoard, 
        wtime: Optional[int], 
        btime: Optional[int], 
        winc: Optional[int], 
        binc: Optional[int], 
        moves_to_go: Optional[int] = None
    ):
    '''
    Oblicza czas na ruch na podstawie aktualnego stanu zegara
    '''

    if board.turn == chess.WHITE:
        time_left = wtime if wtime is not None else None
        inc = winc
    else:
        time_left = btime if btime is not None else None
        inc = binc

    if time_left is None:
        return None

    movetime = int(max(50, min(time_left / 40.0, 8000)))
    if inc:
        movetime += int(inc * 0.7)

    if movetime < 50:
        movetime = 50
    if movetime > time_left:
        movetime = int(time_left * 0.8)

    return movetime

def uci_loop():
    '''
    Główna pętla komunikacji w protokole UCI (Universal Chess Interface)

    Obsługuje komendy wysyłane przez GUI, w naszym przypadku dla CuteChess
    '''
    global uci_running
    cur_board = ZobristBoard()
    last_best_move = None

    while uci_running:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "":
            continue

        parts = shlex.split(line)
        cmd = parts[0]

        if cmd == "uci":
            print("id name Pufferfish")
            print("id author Jakub Jeż")
            print("uciok")
            sys.stdout.flush()

        elif cmd == "isready":
            print("readyok")
            sys.stdout.flush()

        elif cmd == "ucinewgame":
            cur_board = ZobristBoard()
            set_stop_flag(False)
            sys.stdout.flush()

        elif cmd == "position":
            if len(parts) >= 2 and parts[1] == "startpos":
                cur_board = ZobristBoard()
                if "moves" in parts:
                    m_idx = parts.index("moves")
                    moves_parts = parts[m_idx+1:]
                    for m in moves_parts:
                        try:
                            mv = chess.Move.from_uci(m)
                            if mv in cur_board.legal_moves:
                                cur_board.push(mv)
                        except Exception:
                            pass
            elif len(parts) >= 2 and parts[1] == "fen":
                try:
                    if "moves" in parts:
                        m_idx = parts.index("moves")
                        fen_parts = parts[2:m_idx]
                        moves_parts = parts[m_idx+1:]
                    else:
                        fen_parts = parts[2:]
                        moves_parts = []
                    fen_str = " ".join(fen_parts)
                    cur_board = ZobristBoard(fen_str)
                    for m in moves_parts:
                        try:
                            mv = chess.Move.from_uci(m)
                            if mv in cur_board.legal_moves:
                                cur_board.push(mv)
                        except Exception:
                            pass
                except Exception:
                    cur_board = ZobristBoard()
            else:
                pass

        elif cmd == "go":
            depth = None
            movetime = None
            wtime = None
            btime = None
            winc = None
            binc = None
            movestogo = None
            i = 1
            while i < len(parts):
                tok = parts[i]
                if tok == "depth" and i+1 < len(parts):
                    try:
                        depth = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "movetime" and i+1 < len(parts):
                    try:
                        movetime = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "wtime" and i+1 < len(parts):
                    try:
                        wtime = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "btime" and i+1 < len(parts):
                    try:
                        btime = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "winc" and i+1 < len(parts):
                    try:
                        winc = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "binc" and i+1 < len(parts):
                    try:
                        binc = int(parts[i+1])
                    except:
                        pass
                    i += 2
                elif tok == "movestogo" and i+1 < len(parts):
                    try:
                        movestogo = int(parts[i+1])
                    except:
                        pass
                    i += 2
                else:
                    i += 1

            set_stop_flag(False)

            try:
                book_move = get_opening_book_move(cur_board)
                if book_move is not None:
                    uci_move = book_move.uci()
                    print(f"info string book move {uci_move}")
                    print(f"bestmove {uci_move}")
                    sys.stdout.flush()
                    continue
            except Exception as e:
                print(f"info string opening book error: {e}")

            chosen_move = None
            if depth is not None:
                try:
                    val, nodes, mv = engine_select(cur_board, cur_board.turn == chess.WHITE, depth)
                    chosen_move = mv
                    print(f"info depth {depth} score cp {int(val * 100)} pv {mv.uci() if mv else '(none)'}")
                except Exception as e:
                    print(f"info string minimax exception single depth: {e}")
                    chosen_move = None
            else:
                if movetime is None and (wtime is not None or btime is not None):
                    movetime = compute_movetime_from_clock(cur_board, wtime, btime, winc, binc, movestogo)
                if movetime is None:
                    try:
                        val, nodes, mv = engine_select(cur_board, cur_board.turn == chess.WHITE, DEFAULT_DEPTH)
                        chosen_move = mv
                        print(f"info depth {DEFAULT_DEPTH} score cp {int(val * 100)} pv {mv.uci() if mv else '(none)'}")
                    except Exception as e:
                        print(f"info string minimax exception default depth: {e}")
                        chosen_move = None
                else:
                    safety_ms = 50
                    tl_ms = max(10, movetime - safety_ms)
                    max_depth = DEFAULT_MAX_DEPTH
                    val, nodes, mv = iterative_deepening(cur_board, cur_board.turn == chess.WHITE, time_limit_ms=tl_ms, max_depth=max_depth)
                    chosen_move = mv

            if chosen_move is None:
                legal_moves = list(cur_board.legal_moves)
                if legal_moves:
                    chosen_move = legal_moves[0]
                    print(f"info string Fallback to first legal move: {chosen_move.uci()}")

            print(f"bestmove {chosen_move.uci()}")
            sys.stdout.flush()
            set_stop_flag(False)

        elif cmd == "stop":
            set_stop_flag(True)
            sys.stdout.flush()

        elif cmd == "quit":
            uci_running = False
            break

        elif cmd == "setoption":
            try:
                if "name" in parts and "value" in parts:
                    ni = parts.index("name")
                    vi = parts.index("value")
                    name = " ".join(parts[ni+1:vi])
                    value = " ".join(parts[vi+1:])
            except Exception:
                pass
            sys.stdout.flush()

        else:
            sys.stdout.flush()

if __name__ == "__main__":
    try:
        uci_loop()
    except KeyboardInterrupt:
        pass