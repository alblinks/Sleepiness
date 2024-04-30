import numpy as np
import torch
import matplotlib.pyplot as plt
from sleepiness.utility.misc import Loader
from sleepiness.evaluation.aggregators import LabelAggregator
from sleepiness import logger, PassengerState, ReducedPassengerState
from pathlib import Path
from sklearn.metrics import (
    confusion_matrix as _confusion_matrix, ConfusionMatrixDisplay
)

class TimeSeriesDataLoader:
    """
    A minimal Time Series DataLoader 
    class to iterate over image 
    files in a given folder.
    
    The images in the folder are assumed
    to appear in a chronological order.
    No sorting or shuffling is done.
    
    This class offers minimal functionality
    and is intended to be used as a fallback
    for the torchvision DataLoader, if time
    series data is to be loaded.
    """
    image_formats = [".jpg", ".jpeg", ".png"]
    class _MiniDataset:
        def __init__(self, folder: str):
            # Get all image files in the folder
            self.pictures = list(Path(folder).iterdir())
            for pic in self.pictures:
                if pic.suffix not in TimeSeriesDataLoader.image_formats:
                    self.pictures.remove(pic)

            class_label = ReducedPassengerState.from_str(folder.split("/")[-1])
            self.imgs = [(str(img), class_label.value) for img in self.pictures]
            self.classes = [class_label.name]
    
    def __init__(self, folder: str):
        self.dataset = self._MiniDataset(folder)

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

class MetricsPrinter:
    """
    Print colorful and formatted table of 
    Accuracy, Precision, Recall, 
    and F1 Score for each class.
    
    Header is added automatically.
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
            
            self.lines.append(
                f"{self.BOLD}{self.UNDERLINE}{'Class'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Accuracy'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Precision'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'Recall'.center(self.COLWIDTH)}{self.ENDC}" +
                f"| {self.BOLD}{self.UNDERLINE}{'F1 Score'.center(self.COLWIDTH)}{self.ENDC}"
            )

            
    def log(self, class_name: str,accuracy: float, 
                  precision: float, recall: float, f1: float) -> None: 
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
        return
        
class MetricsCollector:
    """
    Context manager to calculate and store recall, 
    precision, and F1 score for a given class prediction.
    
    Input arguments:
    - all_outputs: Predicted labels for all samples
    - all_labels: True labels for all samples
    - class_index: Index of the class the metrics are calculated for
    - class_names: List of class names
    
    Input arrays must be one dimensional and coerable to torch.Tensor.
    
    On instantiation, the confusion matrix is calculated and saved to
    the "confmat" folder. The folder is created in the current working
    directory if it doesn't exist.
    """
    def __init__(self, 
                 model_class: object,
                 all_outputs: torch.Tensor, 
                 all_labels: torch.Tensor,
                 class_names: list[str],
                 aggregator: LabelAggregator = None,
                 reduced: bool = False):
        self.model_class = model_class
        self.outputs = all_outputs
        self.labels = all_labels
        self.class_names = class_names
        self.aggregator = aggregator
        self.reduced = reduced
        
        # Calculate accuracy
        self._accuracy = (self.outputs == self.labels).sum() / len(self.outputs)
        
        # Label map for class names
        ps = PassengerState if not reduced else ReducedPassengerState
        self.label_map = {p.name: p.value for p in ps}
        
        # Printer for pretty metrics
        self.printer = MetricsPrinter()
        
        # Intermediary variables
        self._precision = None
        self._recall = None

        # Generate "plots" folder if it doesn't exist
        self._path = Path("plots")
        if not self._path.exists():
            self._path.mkdir()
        
        # Check input arrays
        if not (isinstance(all_outputs,list) and isinstance(all_labels,list))\
            and (all_outputs.ndim != 1 or all_labels.ndim != 1):
            raise ValueError("Input arrays must be one dimensional.")
        
        # Coerce inputs to torch.Tensor
        if not isinstance(all_outputs, torch.Tensor):
            self.outputs = torch.tensor(all_outputs)
        if not isinstance(all_labels, torch.Tensor):
            self.labels = torch.tensor(all_labels)
            
    
    def summary(self):
        """
        Calculates a confusion matrix, and pretty-prints
        recall, precision, and F1 score for each class.
        """
        self.printable_summary()
        if self.aggregator is None:
            self.confusion_matrix()
        if self.aggregator is not None:
            self.timeline()
        pass

    def printable_summary(self):
        """
        Calculates and pretty-prints recall, precision, and F1 score
        for each class.
        """
        for class_name, i in self.label_map.items():
            self.class_index = i
            self.recall()
            self.precision()
            f1 = self.f1_score()
            self.printer.log(
                class_name, self._accuracy, self._precision, self._recall, f1
            )
        for line in self.printer.lines:
            print(line)
            
    def timeline(self):
        """
        Plots a timeline of the aggregated labels 
        and the raw labels over time.
        """
        ltf = [l.value for l in self.aggregator._all_labels]
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax: plt.Axes
        ax.scatter(
            np.arange(len(ltf)),
            ltf, 
            label="Raw labels", 
            marker="x",
            s=10,
            color="#c1121f"
        )
        
        # The aggregated labels are the normal output, 
        # because if the test run was done using the
        # aggregator, the labels are already aggregated.
        ax.plot(self.outputs, label="Aggregated labels", color="#003049")
        
        # Plot the true labels
        ax.plot(self.labels, label="True labels", color="#ffb703")
        
        # Set the y axis to the PassengerState enum values
        if not self.reduced:
            ax.set_yticks([0, 1, 2])
            ax.set_yticklabels([l.name for l in PassengerState])
        else:
            _r = ReducedPassengerState
            ax.set_yticks([0, 1])
            ax.set_yticklabels([l.name for l in [_r.AWAKE, _r.SLEEPING]])
        
        ax.set_xlabel("Frame / Time")
        ax.set_ylabel("Label")
        
        plt.tight_layout()
        plt.legend()
        plt.savefig(self._path / f"timeline_{type(self.model_class).__name__}.png", dpi=300)
        
        
    def confusion_matrix(self):
        """
        Calculate the confusion matrix for a given class.
        """
        fix, ax = plt.subplots()
        cf = _confusion_matrix(
            self.labels, self.outputs, 
            labels=list(self.label_map.values()), normalize="true"
        )
        disp = ConfusionMatrixDisplay(
            cf, display_labels=list(self.label_map.items())
        )
        disp.plot(ax=ax, xticks_rotation="vertical")
        # Rotate xticks by 45 degrees
        xticks = ax.get_xticklabels()
        for label in xticks:
            label.set_rotation(45)
        
        plt.tight_layout()
        plt.savefig(self._path / f"confmat_{type(self.model_class).__name__}.png")
        logger.info(f"Confusion matrix saved to {self._path.absolute()}.")
        plt.close()
        
    def recall(self):
        """
        Calculate the recall for a given class.
        """
        true_positives = (self.outputs == self.class_index) & (self.labels == self.class_index)
        actual_positives = (self.labels == self.class_index)
        if actual_positives.sum() == 0:
            logger.warning(f"No actual positives for class {self.class_index}. Returning 0.0.")
            self._recall = 0
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
            logger.waring(f"No predicted positives for class {self.class_index}."
                  " Returning 0.0.")
            self._recall = 0
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
            logger.warning(
                f"No actual or predicted positives for "
                f"class {self.class_index}. F1 Score is 0.0."
            )
            return 0
        
        prec = self._precision
        rec = self._recall
        f1 = 2 * (prec * rec) / (prec + rec)
        return f1
