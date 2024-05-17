# Sleepiness classification
This repository provides functionality for detecting whether a person is awake or asleep.

The `sleepiness` package offers a robust command-line interface (CLI) for detecting signs of sleepiness from static images and real-time video feeds. The package can handle various operational modes, including real-time detection, single-image analysis, and continuous detection from a sequence of images or a folder with images. The CLI also supports custom bounding boxes for the hand detection model.

## Features

- **Real-Time Detection**: Utilize your webcam to detect sleepiness in real time.
- **Single Image Detection**: Analyze a single image for signs of sleepiness and terminate.
- **Continuous Detection**: Continuously analyze a series of images from a specified directory, with prompts for new paths after each detection. Image folders are also supported.
- **Customizable Model Parameters**: Set custom bounding boxes for detection models.

## Installation

To get started with `Sleepiness`, install the repository using the following command:

```bash
pip install -e git+https://github.com/MarWaltz/Sleepiness.git
```

or download the latest release from the [releases page](https://github.com/MarWaltz/Sleepiness/releases) and install it using the following command:

```bash
pip install path/to/sleepiness-<version-tag>.tar.gz
```

## Usage

The CLI supports several options to specify the mode of operation:

Usage:
    `sleepiness [options] <command>`

Options:

    -h, --help            Show this screen and exit.
    --version             Show version and exit.
    --calibrate           Calibrate the empty seat detection model.
    -r                    Activate real-time sleepiness detection using webcam input.
    -k <frames>           Specify the number of frames used for aggregation in real-time detection.
    -p, --path <path>     Path to a single image for sleepiness detection. The process terminates after detection.
    -c, --cpath <path>    Continuous, open-loop detection using image paths. Keeps all models 
                          loaded and prompts for a new image path after each detection.
    --hbbox <values>      Specify a bounding box for the hand detection model as xmin xmax ymin ymax. 
                          Values must be within [0, 1] and correspond to the percentage of the image.

## Examples

1. Real-time detection using the webcam:
```bash
sleepiness -r
```

2. Single-image detection:
```bash
sleepiness -p path/to/image.jpg
```

3. Continuous detection using a sequence of images or a folder with images:
```bash
# Folder with images
sleepiness -c [--cpath] path/to/folder/with/images

# Single image (prompts for new path after detection)
sleepiness -c [--cpath] path/to/image.jpg
```

4. Custom detection area for hand detection model:
```bash
sleepiness -r --hbbox 0.1 0.9 0.1 0.9
```