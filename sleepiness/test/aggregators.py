from typing import Callable
from collections import deque
from abc import ABC, abstractmethod
from sleepiness import PassengerState, ReducedPassengerState

class LabelAggregator(ABC):
    """
    Abstract class for the aggegation of labels.
    Subclasses should implement the `aggregate` method,
    which takes a PassengerState and adds it    
    """
    SPAN = 10 # Number of labels to aggregate
    
    def __init__(self):
        self.labels = deque(maxlen=self.SPAN)
        self._all_labels: list[PassengerState] | list[ReducedPassengerState] = []
        
    def add(self, label: PassengerState) -> None:
        """
        Add a label to the aggregator.
        """
        assert isinstance(label, PassengerState), "Invalid label"
        self.labels.append(label)
        self._all_labels.append(label)
    
    @property
    @abstractmethod
    def state(self) -> PassengerState:
        """Returns the aggregated state."""
        pass

class MajorityVoting(LabelAggregator):
    """
    Majority voting label aggregator.
    """
    @property
    def state(self) -> PassengerState:
        """
        Returns the most common label.
        """
        return max(set(self.labels), key=self.labels.count)
    
# Small test
if __name__ == "__main__":
    # Create a majority voting aggregator
    aggregator = MajorityVoting()
    # Add some labels
    aggregator.add(PassengerState.AWAKE)
    aggregator.add(PassengerState.AWAKE)
    aggregator.add(PassengerState.SLEEPING)
    aggregator.add(PassengerState.AWAKE)
    aggregator.add(PassengerState.SLEEPING)
    aggregator.add(PassengerState.SLEEPING)
    aggregator.add(PassengerState.AWAKE)
    
    # Get the aggregated state
    print(aggregator.state()) # Expected output: AWAKE