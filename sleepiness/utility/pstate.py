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
    NOTAVAILABLE = -1
    
    def __str__(self):
        return self.name

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
    else: return ReducedPassengerState.NOTAVAILABLE