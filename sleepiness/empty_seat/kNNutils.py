from threading import Thread
import time
from PIL import Image
import numpy as np
import psutil

# Image downscaling factor
SCALE = 0.2

def _to_grey(img: Image) -> np.ndarray:
    """Convert an image to greyscale"""
    return np.array(img.convert("L"))

def _scale_img(img: np.array, scale: float) -> np.ndarray:
    """
    Scale an image by a factor
    """
    return np.array(
        Image.fromarray(img).resize(
            (int(img.shape[1]*scale), int(img.shape[0]*scale))
        )
    )

#Context manager that continuously shows memory usage
# while running the code inside the context.
class MemoryLoader:
    def __init__(self):
        self.timeout = 0.2

        self._thread = Thread(target=self._show_memory_usage, daemon=True)
        self.done = False

    def start(self):
        self.t_start = time.perf_counter()
        self._thread.start()
        return self

    def _show_memory_usage(self):
        """
        Prints memory usage every `timeout` seconds
        """
        while True:
            if self.done:
                break
            print(
                f"Memory usage: {psutil.virtual_memory().percent}% "
                f"[{psutil.virtual_memory().used/1e9:.2f} GB]", 
                end="\r")
            time.sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        print(" "*100, end = "\r")
        self.t_end = time.perf_counter()
        print(
            f"Loading took {self.t_end - self.t_start:.2f} seconds \n"
            f"Memory usage: {psutil.virtual_memory().percent}% "
            f"[{psutil.virtual_memory().used/1e9:.2f} GB]"
            )

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()