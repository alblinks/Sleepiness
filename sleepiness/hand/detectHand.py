from pathlib import Path
from sleepiness.hand.handYolo import HandYOLO
from sleepiness import __path__ as p

def load_hand_model() -> HandYOLO:
    """Loads and returns the hand model."""

    try:
        cfg_path     = Path(p[0]) / "hand" / "cross-hands.cfg"
        weights_path = Path(p[0]) / "hand" / "cross-hands.weights"
        hand_model = HandYOLO(cfg_path, weights_path, ["hand"])
    except:
        raise FileNotFoundError("Error: Could not load the hand model. Check the paths.")
    
    hand_model.size = 416
    hand_model.confidence = 0.2

    print("Hand model loaded.")
    return hand_model
