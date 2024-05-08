from sleepiness.utility.logger import logger
from sleepiness.utility.pstate import PassengerState, ReducedPassengerState

__version__ = "1.0.2"
__greeting__ = f"""
---------------Welcome to-------------------
 __ _                 _                     
/ _\ | ___  ___ _ __ (_)_ __   ___  ___ ___ 
\ \| |/ _ \/ _ \ '_ \| | '_ \ / _ \/ __/ __|
_\ \ |  __/  __/ |_) | | | | |  __/\__ \__ \\
\__/_|\___|\___| .__/|_|_| |_|\___||___/___/
               |_|                          
------------Version: {__version__}-----------------
"""
print(__greeting__)