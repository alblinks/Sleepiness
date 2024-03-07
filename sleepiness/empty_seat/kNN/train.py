"""
Train a kNN model to classify whether an
airplane seat is empty or not
"""
import os
import dotenv
import pickle
import numpy as np

from pathlib import Path 
from PIL import Image
from sklearn.neighbors import KNeighborsClassifier as KNN
from .utils import MemoryLoader, _to_grey, _scale_img, SCALE

# Load the .env file
dotenv.load_dotenv()

EMPTYPATHTRAIN = Path(os.environ.get("EMPTYPATHTRAIN")).resolve()
NONEMPTYPATHTRAIN = Path(os.environ.get("NONEMPTYPATHTRAIN")).resolve()
EMPTYPATHTEST = Path(os.environ.get("EMPTYPATHTEST")).resolve()
NONEMPTYPATHTEST = Path(os.environ.get("NONEMPTYPATHTEST")).resolve()


def _train_knn():
    """
    Train a kNN model to classify whether an airplane seat is empty or not
    """
    # Get the images
    empty_imgs = [f for f in EMPTYPATHTRAIN.iterdir() if f.is_file()]
    nonempty_imgs = [f for f in NONEMPTYPATHTRAIN.iterdir() if f.is_file()]
    # Create the labels
    empty_labels = np.zeros(len(empty_imgs))
    nonempty_labels = np.ones(len(nonempty_imgs))
    # Create the data
    data = []
    labels = []
    for img in empty_imgs:
        data.append(_scale_img(_to_grey(Image.open(img)),scale=SCALE).flatten())
    for img in nonempty_imgs:
        data.append(_scale_img(_to_grey(Image.open(img)),scale=SCALE).flatten())
    labels = np.concatenate([empty_labels, nonempty_labels])
    # Train the model
    model = KNN(n_neighbors=3)
    model.fit(data, labels)
    # Save the model
    with open("empty_seat_classifier.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model trained and saved successfully")

def _test_kNN():
    """
    Test the kNN model
    """
    # Load the model
    with open("empty_seat_classifier.pkl", "rb") as f:
        model = pickle.load(f)
    # Get the images
    empty_imgs = [f for f in EMPTYPATHTEST.iterdir() if f.is_file()]
    nonempty_imgs = [f for f in NONEMPTYPATHTEST.iterdir() if f.is_file()]
    # Create the labels
    empty_labels = np.zeros(len(empty_imgs))
    nonempty_labels = np.ones(len(nonempty_imgs))
    # Create the data
    data = []
    labels = []
    for img in empty_imgs:
        data.append(_scale_img(_to_grey(Image.open(img)),scale=SCALE).flatten())
    for img in nonempty_imgs:
        data.append(_scale_img(_to_grey(Image.open(img)),scale=SCALE).flatten())
    labels = np.concatenate([empty_labels, nonempty_labels])
    # Test the model
    accuracy = model.score(data, labels)
    print(f"Accuracy: {accuracy}")
    
if __name__ == "__main__":
    with MemoryLoader():
        _train_knn()
    _test_kNN()
    print("kNN model trained and tested successfully")