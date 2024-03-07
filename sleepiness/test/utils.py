import torch
from warnings import warn
from sleepiness.utility.misc import Loader
from numpy import ndarray


_Array = ndarray[float] | list | tuple | torch.Tensor

def with_loader(func):
    """
    Decorator to run a function with an 
    animated loader.
    """
    def wrapper(*args, **kwargs):
        if "n_samples" in kwargs:
            n_samples = kwargs["n_samples"]
        with Loader(desc=f"Evaluating model on {n_samples} samples"):
            return func(*args, **kwargs)
    return wrapper


class ClassifierMetrics:
    """
    Context manager to calculate and store recall, 
    precision, and F1 score for a given class prediction.
    
    Input arguments:
    - outputs: Predicted class labels
    - labels: True class labels
    - class_index: Index of the class the metrics are calculated for
    
    Input arrays must be one dimensional and coerable to torch.Tensor.
    """
    def __init__(self, outputs: _Array, labels: _Array, class_index: int):
        self.outputs = outputs
        self.labels = labels
        self.class_index = class_index
        
        # Intermediary variables
        self._precision = None
        self._recall = None
        
        # Check input arrays
        if not (isinstance(outputs,list) and isinstance(labels,list))\
            and (outputs.ndim != 1 or labels.ndim != 1):
            raise ValueError("Input arrays must be one dimensional.")
        
        # Coerce inputs to torch.Tensor
        if not isinstance(outputs, torch.Tensor):
            self.outputs = torch.tensor(outputs)
        if not isinstance(labels, torch.Tensor):
            self.labels = torch.tensor(labels)
            
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        pass
        
    def recall(self):
        """
        Calculate the recall for a given class.
        """
        true_positives = (self.outputs == self.class_index) & (self.labels == self.class_index)
        actual_positives = (self.labels == self.class_index)
        if actual_positives.sum() == 0:
            print(f"No actual positives for class {self.class_index}. Returning 0.0.")
            return 0
        recall = true_positives.sum() / actual_positives.sum()
        self._recall = recall
        return recall
    
    def precision(self):
        """
        Calculate the precision for a given class.
        """
        true_positives = (self.outputs == self.class_index) & (self.labels == self.class_index)
        predicted_positives = (self.outputs == self.class_index)
        if predicted_positives.sum() == 0:
            print(f"No predicted positives for class {self.class_index}."
                  " Returning 0.0.")
            return 0
        precision = true_positives.sum() / predicted_positives.sum()
        self._precision = precision
        return precision
    
    def f1_score(self):
        """
        Calculate the F1 score for a given class.
        """
        if self._precision is None:
            self.precision()
        if self._recall is None:
            self.recall()
            
        if self._precision == 0 and self._recall == 0:
            print(f"No actual or predicted positives for class {self.class_index}. F1 Score is 0.0.")
            return 0
        
        prec = self._precision
        rec = self._recall
        f1 = 2 * (prec * rec) / (prec + rec)
        return f1

class ClassifierMetricsPrinter:
    """
    Context manager to print a colorful and 
    formatted table of Accuracy, Precision, Recall, 
    and F1 Score for each class.
    
    Keeps track of its header automatically
    
    Prints the header and the lines from the internal buffer
    when the context manager is exited.
    
    """
    # ANSI escape codes for colors and styles
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    COLWIDTH = 10

    def __init__(self):
        
            self.lines = []
            
            self.header = (
                f"{self.BOLD}{self.UNDERLINE}{'Class'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Accuracy'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Precision'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Recall'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'F1 Score'.center(self.COLWIDTH)}{self.ENDC}"
            )
            
    def __enter__(self):
        return self
        
    def log_metics(self,
                      class_name: str,accuracy: float, 
                      precision: float, recall: float, f1: float): 
        """
        Adds a colored and formatted table of Accuracy, 
        Precision, Recall, and F1 Score for each class
        to the internal line buffer.
        """
        # Print metrics for each class
        col_width = self.COLWIDTH
        self.lines.append(
            f"{self.BOLD}{class_name.ljust(col_width)}{self.ENDC}" +
            f"| {self.FAIL}{accuracy:^{col_width-1}.2f}{self.ENDC} " +
            f"| {self.GREEN}{precision:^{col_width-1}.2f}{self.ENDC} " +
            f"| {self.BLUE}{recall:^{col_width-1}.2f}{self.ENDC} " +
            f"| {self.WARNING}{f1:^{col_width-1}.2f}{self.ENDC}"
        )
        
    def __exit__(self, exc_type, exc_value, traceback):
        print(self.header)
        for line in self.lines:
            print(line)