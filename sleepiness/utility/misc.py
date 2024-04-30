from itertools import cycle
from threading import Thread
import time
import requests
from tqdm import tqdm

def download_file_with_progress(url: str, 
                                destination: str,
                                descr: str) -> None:
    """
    Downloads a file from a specified URL with a progress bar and saves it to a destination file.

    Parameters:
        url (str): The URL of the file to download.
        destination (str): The local path to save the downloaded file.
        descr (str): The description of the download process.
    """
    # Send a HTTP request to the server.
    response = requests.get(url, stream=True)

    # Check if the request was successful.
    if response.status_code == 200:
        # Get the total file size via the header (if available).
        total_size = int(response.headers.get('content-length', 0))

        # Open a local file for writing in binary mode.
        with open(destination, 'wb') as file, tqdm(
                desc=descr,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):  # Read the data in chunks
                size = file.write(data)  # Write data chunk to the file
                bar.update(size)  # Update the progress bar
    else:
        print("Failed to download the file: HTTP ", response.status_code)


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
        
