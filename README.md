# Sleepiness classification
This repository provides functionality for detecting whether a person is awake or asleep. The core of the approach is the [yolov5](https://github.com/ultralytics/yolov5) network. To reproduce the results, consider the following steps.

## 1. Generate training data
Use the file ```generateTrainingData.py``` to take pictures of yourself (or possible others) using your local webcam. Make sure to specify the correct storage path; they should be saved in ```data/images```.

After you created a sufficient amount of data, or imported them from some other source, it's time to label them. For this, you can use [labelImg](https://github.com/HumanSignal/labelImg) as follows:
1. Create the folder ```data/labels``` if not already existent.
2. Install [conda](https://docs.anaconda.com/) and create a virtual environment as follows. Make sure to use Python version 3.9; otherwise [labelImg](https://github.com/HumanSignal/labelImg) procudes issues.
```bash
$ conda create --name envp39 python=3.9
$ conda activate envp39
$ pip install labelImg
$ labelImg
```
3. Label each of the images in ```data/images``` manually and make sure to store them in the ```data/labels``` file.

## 2. Fine-tune the pre-trained model
Now we are fine-tuning the parameters of the model using the labeled data from the first step as follows:
1. Clone the ```yolov5``` repo into this repo:
```bash
$ git clone https://github.com/ultralytics/yolov5
```
2. Modify the ```dataset.yml``` in the ```yolov5``` folder to link correctly to your labels and images. Moreover, specify the desired classes correctly. For the binary case, we have either ```awake``` or ```sleepy```. 
2. Run the following command from inside the ```yolov5``` directory, where you can adjust the parameters as desired:
```bash
$ python train.py --img 320 --batch 16 --epochs 300 --data dataset.yml --weights yolov5s.pt
```
3. You can find the results in the folder ```yolov5/runs```.

## 3. Real-time detection
Run ```realTimeDetection.py``` to see how it works! Make sure to change the path to the trained weights from the second step accordingly. The output should look like this:

![Example Image](https://github.com/MarWaltz/Sleepiness/blob/main/initial_tests/exampleOutput.jpg)





