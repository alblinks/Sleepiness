
"""
This empty_seat detector first calculates
the average pixel values of the rescaled and 
grey-scaled images in the training set. Then,
it calculates the difference between the
average pixel values of the training set and
the average pixel values of the test set. If
the difference is greater than a certain
threshold, the seat is considered empty.
"""
from pathlib import Path
from typing import Generator
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import pairwise

from sleepiness.utility.misc import Loader

IMAGE_WIDTH = 100 // 2
IMAGE_HEIGHT = 116 // 2

AVGMAP = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH))

def crop_bottom(image: Image.Image, percent: float) -> Image.Image:
    """
    Crop the bottom of an image.
    """
    width, height = image.size
    return image.crop((0, 0, width, height - int(height * percent)))

def crop_sides(image: Image.Image, percent: float) -> Image.Image:
    """
    Crop the sides of an image.
    """
    width, height = image.size
    return image.crop((int(width * percent), 0, width - int(width * percent), height))

def rescale_greyscale_normalize(image: Image.Image) -> np.ndarray:
    """
    Rescale and greyscale an image.
    """
    # Rescale the image to 100x116
    image = image.resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    # Convert to greyscale
    image = image.convert('L')
    # Max-min normalization
    image = (np.array(image) - np.array(image).min()) / np.array(image).ptp()
    return image


def dataloader(path:Path) -> Generator[np.ndarray, None, None]:
    """
    Load images from a directory.
    """
    for image_path in path.glob('*.jpg'):
        image = Image.open(image_path)
        yield preprocess(image)
        
def preprocess(image: Image.Image) -> np.ndarray:
    """
    Preprocess an image.
    """
    image = crop_bottom(image, 0.3)
    image = crop_sides(image, 0.2)
    return rescale_greyscale_normalize(image)
        
def running_average(image: np.ndarray, n: int) -> np.ndarray:
    """
    Calculate the running average of an image.
    """
    global AVGMAP
    if n == 0:
        AVGMAP = image
        return AVGMAP
    AVGMAP = (AVGMAP * n + image) / (n + 1)
    return AVGMAP

def pixdiff(image: np.ndarray, map: np.ndarray | None = None) -> float:
    """
    Calculate the pixel difference between an image
    and the AVGMAP.
    """
    if map is not None:
        return np.abs(image - map).mean()
    return np.abs(image - AVGMAP).mean()

def pixdiff_distribution(path: Path) -> np.ndarray:
    """
    Calculate the pixel difference distribution
    between the training set and the test set.
    """
    diff_distribution = []
    with Loader("Calculating pixel difference distribution") as loader:
        for n, image in enumerate(dataloader(path)):
            loader.desc = f"Calculating pixel difference distribution {n}"
            diff_distribution.append(pixdiff(image))
    return np.array(diff_distribution)

def paiwise_pixdiff_distribution(path: Path) -> np.ndarray:
    """
    Calculate the pixel difference distribution
    between the training set and the test set.
    """
    diff_distribution = []
    with Loader("Calculating pixel difference distribution") as loader:
        for n, (image1, image2) in enumerate(pairwise(dataloader(path))):
            loader.desc = f"Calculating pixel difference distribution {n}"
            diff_distribution.append(np.abs(image1 - image2).mean())
    return np.array(diff_distribution)

def populate_avg_map(train_path: Path) -> bool:
    """
    Detect if a seat is empty.
    """
    # Calculate the running average of the training set
    with Loader("Calculating running average") as loader:
        for n, image in enumerate(dataloader(train_path)):
            loader.desc = f"Calculating running average {n}"
            running_average(image, n)
    
def is_empty(image: np.ndarray,
             threshold: float,
             map: np.ndarray) -> bool:
    """
    Detect if a seat is empty.
    """
    # Calculate the pixel difference between 
    # the test set and the running average
    if pixdiff(image,map) > threshold:
        return False
    return True

def save_avg_map() -> None:
    """
    Save the average map.
    """
    import pickle
    with open('sleepiness/empty_seat/pixdiff/avgmap.pkl', 'wb') as f:
        pickle.dump(AVGMAP, f)

def plot_distribution(distr: np.ndarray, 
                      title: str) -> None:
    """
    Plot the pixel difference distribution.
    """
    plt.hist(distr, bins=100)
    plt.title(f"Pixel difference distribution {title}")
    plt.xlabel("Pixel difference")
    plt.ylabel("Frequency")
    plt.savefig(f"pixdiff_distribution_{title}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    # Define the paths
    train_path = Path("pictures/empty_seat_dataset/train/not there")
    #test_path = Path("pictures/empty_seat_dataset/test/awake")
    
    populate_avg_map(train_path)
    save_avg_map()
    plt.imshow(AVGMAP, cmap='gray')
    plt.savefig("average_empty.png", dpi=300)
    plt.close()
    
    distr_awake = pixdiff_distribution(Path("pictures/empty_seat_dataset/train/awake"))
    plot_distribution(distr_awake, "awake")
    
    distr_nt = pixdiff_distribution(Path("pictures/empty_seat_dataset/train/not there"))
    plot_distribution(distr_nt, "not there")
    
    # Plot the pairwise pixel difference distribution inside both training sets
    pw_dist = paiwise_pixdiff_distribution(Path("pictures/empty_seat_dataset/train/awake"))
    plot_distribution(pw_dist, "awake_pairwise")
    
    pw_dist = paiwise_pixdiff_distribution(Path("pictures/empty_seat_dataset/train/not there"))
    plot_distribution(pw_dist, "not there_pairwise")