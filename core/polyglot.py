import chess
import chess.polyglot
from core.utils import resource_path

OPENING_BOOK_PATH = resource_path("tablebases/polyglot/Cerebellum3Merge.bin")

def get_opening_book_move(board, book_path=OPENING_BOOK_PATH):
    """Zwraca ruch z książki debiutów Polyglot dla danej pozycji."""
    try:
        with chess.polyglot.open_reader(book_path) as reader:
            entry = reader.find(board)
            if entry is not None:
                return entry.move
            else:
                print("Nie ma ruchu w książce debiutów dla tej pozycji.")
    except FileNotFoundError:
        print(f"Nie znaleziono książki debiutów pod adresem {book_path}")
    except IndexError:
        print("Nie znaleziono prawidłowego ruchu dla tej pozycji na planszy w książce debiutów.")
    return None