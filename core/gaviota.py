import chess
import chess.gaviota

def get_move_from_table(board, tb):
    best_move = None

    for mv in board.legal_moves:
        board.push(mv)

        try:
            wdl = tb.probe_wdl(board) 
        except chess.gaviota.MissingTableError:
            board.pop()
            continue
        except Exception:
            board.pop()
            continue

        try:
            dtm = tb.probe_dtm(board)
        except chess.gaviota.MissingTableError:
            dtm = None
        except Exception:
            dtm = None

        if best_move is None and wdl != 0:
            best_move = (mv, wdl, dtm)
        else:
            # Wygrywający
            if wdl < 0:                    
                if dtm is not None and dtm > best_move[2]:
                    best_move = (mv, wdl, dtm)
            # Przegrywający
            elif wdl > 0:
                if dtm is not None and dtm > best_move[2]:
                    best_move = (mv, wdl, dtm)

        board.pop()

    return chess.Move.from_uci(best_move[0].uci())

