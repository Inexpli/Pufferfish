import threading

stop_search = False
stop_lock = threading.Lock()

def set_stop_flag(val: bool):
    '''
    Ustawia flagę zatrzymania wyszukiwania
    '''
    global stop_search
    with stop_lock:
        stop_search = val

def get_stop_flag() -> bool:
    '''
    Zwraca stan flagi zatrzymania wyszukiwania
    '''
    with stop_lock:
        return stop_search