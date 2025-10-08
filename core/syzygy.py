import chess
import chess.syzygy

def get_move_from_table(board, tb):
    ''' Zwraca najlepszy ruch z tabel Syzygy.'''
    best_move = None

    for mv in board.legal_moves:
        board.push(mv)

        try:
            wdl = tb.probe_wdl(board) 
        except chess.syzygy.MissingTableError:
            board.pop()
            continue
        except Exception:
            board.pop()
            continue

        try:
            dtz = tb.probe_dtz(board)
        except chess.syzygy.MissingTableError:
            dtz = None
        except Exception:
            dtz = None

        if best_move is None and wdl != 0:
            best_move = (mv, wdl, dtz)
        else:
            # Wygrywający
            if wdl < 0:                    
                if dtz is not None and dtz > best_move[2]:
                    best_move = (mv, wdl, dtz)
            # Przegrywający
            elif wdl > 0:
                if dtz is not None and dtz > best_move[2]:
                    best_move = (mv, wdl, dtz)

        board.pop()

    return chess.Move.from_uci(best_move[0].uci())

