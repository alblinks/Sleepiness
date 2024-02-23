from pathlib import Path
from sleepiness.hand.handYolo import HandYOLO

def load_hand_model() -> HandYOLO:
    """Loads and returns the hand model."""

    try:
        cfg_path     = Path("sleepiness") / "hand" / "cross-hands.cfg"
        weights_path = Path("sleepiness") / "hand" / "cross-hands.weights"
        hand_model = HandYOLO(cfg_path, weights_path, ["hand"])
    except:
        raise FileNotFoundError("Error: Could not load the hand model. Check the paths.")
    
    hand_model.size = 416
    hand_model.confidence = 0.2

    print("Hand model loaded.")
    return hand_model
