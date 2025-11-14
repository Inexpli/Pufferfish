import chess
import chess.gaviota

def get_move_from_table(board, tb):
    """Zwraca najlepszy ruch z tabeli końcówek Gaviota dla danej pozycji."""
    best_move = None
    best_wdl = None  
    best_dtm = None

    for mv in board.legal_moves:
        board.push(mv)
        try:
            wdl_post = tb.probe_wdl(board)
        except chess.gaviota.MissingTableError:
            board.pop()
            continue
        except Exception:
            board.pop()
            continue

        try:
            dtm_post = tb.probe_dtm(board)
        except chess.gaviota.MissingTableError:
            dtm_post = None
        except Exception:
            dtm_post = None

        wdl = -wdl_post
        dtm = -dtm_post if dtm_post is not None else None

        if best_move is None:
            best_move = mv
            best_wdl = wdl
            best_dtm = dtm
            board.pop()
            continue

        if wdl == 1:
            if best_wdl != 1:
                best_move, best_wdl, best_dtm = mv, wdl, dtm
            else:
                if dtm is not None and best_dtm is not None:
                    if abs(dtm) < abs(best_dtm):
                        best_move, best_wdl, best_dtm = mv, wdl, dtm

        elif wdl == 0:
            if best_wdl == -1:
                best_move, best_wdl, best_dtm = mv, wdl, dtm

        elif wdl == -1:
            if best_wdl == -1:
                if dtm is not None and best_dtm is not None:
                    if abs(dtm) > abs(best_dtm):
                        best_move, best_wdl, best_dtm = mv, wdl, dtm

        board.pop()

    if best_move is None:
        raise Exception("Brak ruchów w tabeli końcówek Gaviota dla tej pozycji.")

    return best_wdl*1000, best_dtm, best_move

