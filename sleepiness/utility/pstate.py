from __future__ import annotations
from enum import Enum

class PassengerState(Enum):
    """
    Enum class for the state of the passenger
    and the corresponding labels.
    
    We define the following states in alphabetical order:
    """
    AWAKE = 0
    NOTTHERE = 1
    SLEEPING = 2
    
    def __str__(self):
        return self.name

    def from_int(i: int) -> PassengerState:
        """
        Convert an integer to the corresponding
        PassengerState.
        """
        if i == 0:
            return PassengerState.AWAKE
        elif i == 1:
            return PassengerState.NOTTHERE
        elif i == 2:
            return PassengerState.SLEEPING
        else:
            raise ValueError(f"Invalid integer value {i} for PassengerState.")

    def from_str(s: str) -> PassengerState:
        """
        Convert a string to the corresponding
        PassengerState.
        """
        s = s.upper()
        if s == "AWAKE":
            return PassengerState.AWAKE
        elif s == "NOTTHERE":
            return PassengerState.NOTTHERE
        elif s == "SLEEPING":
            return PassengerState.SLEEPING
        else:
            raise ValueError(f"Invalid string value {s} for PassengerState.")

class ReducedPassengerState(Enum):
    """
    Reduced Enum class for the state 
    of the passenger to only test 
    for sleeping or awake.
    
    This is used as the classification
    for empty or occupied seats is relatively
    trivial, which is why we can reduce the
    number of states to two.
    
    We define the following states in alphabetical order:
    """
    AWAKE = 0
    SLEEPING = 1
    
    def __str__(self):
        return self.name
    
    def from_int(i: int) -> ReducedPassengerState:
        """
        Convert an integer to the corresponding
        ReducedPassengerState.
        """
        if i == 0:
            return ReducedPassengerState.AWAKE
        elif i == 1:
            return ReducedPassengerState.SLEEPING
        else:
            raise ValueError(f"Invalid integer value {i} for ReducedPassengerState.")
        
    def from_str(s: str) -> ReducedPassengerState:
        """
        Convert a string to the corresponding
        ReducedPassengerState.
        """
        s = s.upper()
        if "AWAKE" in s:
            return ReducedPassengerState.AWAKE
        elif "SLEEPING" in s:
            return ReducedPassengerState.SLEEPING
        else:
            raise ValueError(f"Invalid string value {s} for ReducedPassengerState.")

# Transform the PassengerState to a ReducedPassengerState
def reduce_state(state: PassengerState) -> ReducedPassengerState:
    """
    Reduce the state of the passenger to 
    either sleeping or awake.
    """
    if state == PassengerState.AWAKE:
        return ReducedPassengerState.AWAKE
    elif state == PassengerState.SLEEPING:
        return ReducedPassengerState.SLEEPING
    else: return None