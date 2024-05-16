"""
This module provides the command-line interface (CLI) for the `sleepiness` package,
which is designed to detect the state sleepiness from images or real-time video feeds.
The CLI allows users to interact with the package's functionality from the command line,
such as real-time detection, single image detection, and continuous image detection
from a directory of image files.

The CLI supports several command options to specify the mode of operation:
- Display the current version of the package.
- Perform sleepiness detection in real-time using a webcam.
- Detect sleepiness from a single image.
- Perform continuous detection from a sequence of images, 
    with the user prompted for new image paths after each detection.

Usage:
    sleepiness [options] <command> [<args>...]

Options:
    -h, --help            Show this screen and exit.
    --version             Show version and exit.
    --calibrate           Calibrate the empty seat detection model.
    -r                    Activate real-time sleepiness detection using webcam input.
    -k <frames>           Specify the number of frames used for aggregation in real-time detection.
    -p, --path <path>     Path to a single image for sleepiness detection. The process terminates after detection.
    -c, --cpath <path>    Continuous, open-loop detection using image paths. 
                          Keeps all models loaded and prompts for a new image path after each detection.
    --hbbox <values>      Specify a bounding box for the hand detection
                          model as xmin xmax ymin ymax. Values must be within [0, 1] 
                          and correspond to the percentage of the image.

Example:
    sleepiness --version                  # Display the version of the package
    sleepiness -r -k 10                   # Start real-time detection, aggregating over 10 frames
    sleepiness -p "/path/to/image.jpg"    # Detect sleepiness in a specified single image
    sleepiness -c "/path/to/images"       # Continuously detect sleepiness from images in a directory, prompting for new paths after each detection

This module also defines the `SleepinessCLI` class which encapsulates the CLI functionality,
handling command parsing, and execution routing based on the user inputs.
"""

import argparse
import sys

from pathlib import Path

from sleepiness import __version__, logger
from sleepiness.pipelines import FullPipeline
from sleepiness.evaluation.aggregators import MajorityVoting
from sleepiness.utility.realTimeDetection import aggregated_real_time_detection

# Custom type conversion function to parse space-separated floats
def parse_bbox(value):
    try:
        # Split the input string and convert each part to float
        return [float(x) for x in value.split()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid bounding box format: '{value}'")

class SleepinessCLI:
    HAND_MODEL_CONFIDENCE = 0.15
    EYE_MODEL_CONFIDENCE = 0.2
    
    def __init__(self, parser: argparse.ArgumentParser):
        self.parser = parser
        self.args = parser.parse_args()
        
        if hasattr(self.args, "calibrate"):
            if self.args.calibrate:
                from sleepiness.empty_seat.pixdiff._pixdiff import calibrate_from_device
                logger.info("Calibrating empty seat detection model...")
                calibrate_from_device()
                logger.info("Calibration complete. Exiting...")
                sys.exit()
        
        if hasattr(self.args, "path"):
            if self.args.path and self.args.cpath:
                sys.exit(
                    "Options -p and -c are mutually exclusive. "
                    "Single image detection cannot be combined with continuous detection."
                )
            
            if self.args.r and (self.args.path or self.args.cpath):
                sys.exit(
                    "Options -r and -p or -pc are mutually exclusive. "
                    "Real time detection cannot be combined with single image detection."
                )
            
        # Parse the hand model crop factors
        hbbox = self.args.hbbox
        if not all([0 <= f <= 1 for f in hbbox]):
            raise ValueError("Crop factors must be in the range [0, 1]")
        if not (hbbox[0] < hbbox[1] and hbbox[2] < hbbox[3]):
            raise ValueError(
                "xmin must be less than xmax and ymin must be less than ymax. "
                f"Got: {hbbox}"
            )
        
        self._load_model(
            hand_model_confidence=self.HAND_MODEL_CONFIDENCE,
            eye_model_confidence=self.EYE_MODEL_CONFIDENCE,
            hand_model_crop=hbbox
        )
        
        self.argmap = {
            "version": self._get_version,
            "r": self._real_time_detection,
            "p": self._single_image_detection,
            "c": self._continuous_detection
        }
        
    def run(self) -> None:
        """
        Runs the command line interface based on the parsed arguments.
        """
        if self.args.version:
            self.argmap["version"]()
        elif self.args.r:
            self.argmap["r"]()
        elif self.args.path:
            self.argmap["p"]()
        elif self.args.cpath:
            self.argmap["c"]()
        else:
            self.parser.print_help()
            
    def _load_model(
        self,*,
        hand_model_confidence: float,
        eye_model_confidence: float,
        hand_model_crop: tuple[float, float, float, float]) -> None:
        """
        Load the full pipeline model with the specified confidence and crop factors.
        """
        assert all([0 <= f <= 1 for f in hand_model_crop]), "Crop factors must be in the range [0, 1]"
        assert 0 <= hand_model_confidence <= 1, "Hand model confidence must be in the range [0, 1]"
        assert 0 <= eye_model_confidence <= 1, "Eye model confidence must be in the range [0, 1]"
        self.model = FullPipeline(
            hand_model_confidence=hand_model_confidence,
            eye_model_confidence=eye_model_confidence,
            hand_model_crop=hand_model_crop
        )
            
    def _get_version(self) -> None:
        print(f"Sleepiness version: {__version__}")
        
    
    def _real_time_detection(self) -> None:
        """
        start real-time sleepiness detection using webcam input.
        """
        nagg_frames = self.args.k
        assert nagg_frames > 0, "Number of frames for aggregation must be greater than 0."
        return aggregated_real_time_detection(
            self.model, MajorityVoting(horizon=nagg_frames), True
        )
        
    def _single_image_detection(self) -> None:
        """
        Perform sleepiness detection on a single image.
        """
        assert Path(self.args.path).exists(), "Path to image does not exist."
        if Path(self.args.path).is_dir():
            raise ValueError(
                "Path is a directory. Please use the -c option for continuous detection.")
        return logger.inference(self.model.classify(self.args.path).name)
    
    def _continuous_detection(self) -> None:
        """
        Perform sleepiness detection on a single image and prompt for a new image path.
        """
        while True:
            try:
                assert Path(self.args.cpath).exists(), "Path to image does not exist."
                if Path(self.args.cpath).is_dir():
                    for file in Path(self.args.cpath).iterdir():
                        if file.suffix in [".jpg", ".png"]:
                            logger.inference(self.model.classify(str(file)).name)
                        else: 
                            logger.warning(f"Skipping file {file.name} as it is not a valid image file.")
                            continue
                else:
                    logger.inference(self.model.classify(self.args.cpath))
                    self.args.cpath = input("Enter a new image path: ")
            except KeyboardInterrupt:
                logger.info("Exiting...")
                break
        return None

def main() -> None:
    parser = argparse.ArgumentParser(description="Command line interface for the sleepiness package.")
    parser.add_argument("--version", action="store_true", help="Show version and exit.")
    parser.add_argument("-r", help="Real-time sleepiness detection using webcam input.", action="store_true")
    parser.add_argument("-k", help="Number of frames used for aggregation.", type=int, default=30)
    parser.add_argument(
        "-p", "--path", 
        help="Path to a single image for sleepiness detection. Terminates after detection.")
    parser.add_argument(
        "-c", "--cpath", 
        help=(
            "Continuous, open loop detection using a single image paths. Contrary to -p, "
            "this command keeps all models loaded and prompts for a new image path "
            "after each detection."
        )
    )
    parser.add_argument(
        "--hbbox",
        help=(
            "Bounding box for the hand detection in the format: xmin,xmax,ymin,ymax. "
            "Values must be in the range [0, 1] and correspond to the percentage of the image. "
            "y=0 is the top of the image and x=0 is the left of the image. "
            "Default: 0.25 0.75 0.0 0.8"
        ),
        nargs=4,
        type=parse_bbox,
        default="0.25 0.75 0.0 0.8"
    )

    parser.add_argument(
        "--calibrate", action="store_true", 
        help=(
            "Calibrate the empty seat detection model. Make sure that your camera is directed "
            "towards the empty seat and that the seat is not obstructed. "
            "The calibration process will take 200 images of the empty seat and calculate the "
            "running average of the pixel difference. This average will be saved to a file "
            "and can be used for the empty seat detection."
        )
    )

    cli = SleepinessCLI(parser)
    cli.run()

if __name__ == "__main__":
    main()