"""
Use a pre-trained kNN model to classify whether an
airplane seat is empty or not
"""
import os
import pickle

from PIL import Image
import requests
from sklearn.neighbors import KNeighborsClassifier as KNN
from .kNNutils import _to_grey, _scale_img, SCALE

cls_url = "https://www.dropbox.com/scl/fi/12nbm9uof9iiktbhsszs3/kNN-empty_seat_classifier.pkl?rlkey=tdrk3oj3xghamdjoakiwkfu2s&dl=1"

# Download the pre-trained model if it doesn't exist
if not os.path.exists("empty_seat/kNN-empty_seat_classifier.pkl"):
    print("Downloading pre-trained model")
    res = requests.get(cls_url)
    if res.status_code != 200:
        raise ValueError("Failed to download the pre-trained model:", res.status_code)
    with open("empty_seat/kNN-empty_seat_classifier.pkl", "wb") as f:
        f.write(res.content)

with open("empty_seat/kNN-empty_seat_classifier.pkl", "rb") as f:
    model: KNN = pickle.load(f)
    
def is_empty(img: Image) -> bool:
    """
    Predict whether an airplane seat is empty or not
    """
    img = _scale_img(_to_grey(img), scale=SCALE).flatten()
    return model.predict([img])[0] == 0
