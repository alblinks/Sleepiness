from itertools import cycle
from threading import Thread
import time

class Loader:
    def __init__(
        self,
        desc="Buffering",
        timeout=0.1,
        ):
        """
        Args:
            desc (str, optional): The loader's description. Defaults to "Buffering...".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["🌑","🌒","🌓","🌔","🌕","🌖","🌗","🌘"]
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps[::-1]):
            if self.done:
                break
            print(f"{self.desc}... {c}", flush=True, end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()
        return self

    def stop(self):
        self.done = True
        self.t_end = time.perf_counter()
        print(f"{self.desc} completed in [{(self.t_end-self.t_start):.1f} s]")

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()
        
