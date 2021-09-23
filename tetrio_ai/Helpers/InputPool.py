import multiprocessing as mp
import queue

from Types.Singleton import Singleton

class InputPool(metaclass=Singleton):
    def __init__(self):
        self.input_queue = mp.Queue()

    def add_key(self, key):
        self.input_queue.put(key)

    def get_key(self):
        while not self.input_queue.empty():
            try:
                key = self.input_queue.get(block=False)
                yield key
            except queue.Empty:
                pass
