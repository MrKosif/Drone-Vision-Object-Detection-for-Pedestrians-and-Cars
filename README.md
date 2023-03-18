# Drone Vision: Object Detection for Pedestrians and Cars

<img src="/readme/drone.jpeg" align="left" width="220px" height="192px"/>
<img align="left" width="0" height="192px" hspace="10"/>

### Description
> This project features a deep learning model for detecting pedestrians and cars in drone footage. It aims to provide an effective solution for low altitude vehicles that require proper object detection to navigate safely.
<br>
<br>

## About Project
The purpose of this project was to develop an object detection model capable of detecting pedestrians and cars from drone footages. With the increasing popularity of low altitude vehicles, it is becoming more crucial to have a reliable object detection system that can detect potential obstacles under the vehicle. This project aimed to address this issue by developing an accurate and efficient model that can detect these common objects. By doing so, the model can help improve safety in various low altitude vehicle applications, such as package delivery, aerial photography, and inspection of infrastructure. Ultimately, the goal of this project is to contribute to the development of safe and efficient low altitude vehicle technology, while ensuring the safety of pedestrians and drivers on the ground.


## Installing the Required Libraries

To use this model, you need to first install Detectron2. You can install it with the following command:
```sh
$ pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

To install other required libraries:
```sh
$ pip3 install requirements.txt
```

## Dataset
The dataset used to train and validate the object detection model consists of 1.5k images in the training set and 163 images in the validation set. The dataset contains 9327 instances of cars and 3586 instances of pedestrians. The images were augmented with various transformations such as grayscale, hue, saturation, brightness, exposure, blur, and noise to improve the model's performance on different lighting conditions and backgrounds. Specifically, the augmentations applied to the images include grayscale (applied to 25% of images), hue (between -25° and +25°), saturation (between -25% and +25%), brightness (between -25% and +25%), exposure (between -25% and +25%), blur (up to 10px), and noise (up to 5% of pixels). These augmentations help the model to generalize better to different scenarios and increase its accuracy.

You can download the dataset by clicking [here](https://universe.roboflow.com/kerem-kosif/drone-vision-project)

## Project Files
This repository contains the following files:

1. prepare_data.py: This file is used to prepare the data for the model. It reads the annotations file, registers the dataset and metadata. You can modify this file to fit your own dataset.

2. configure_model.py: This file sets the configuration for the model, creates the trainer and predictor. You can modify this file to set your own configurations for the model.

3. predictions.py: This file performs prediction on the given image and returns the output. It also visualizes the prediction. You can modify this file to visualize the predictions in your own way.

4. evaluate.py: This file evaluates the model on the test dataset and returns the results. You can modify this file to fit your own evaluation criteria.

## Usage
To use this object detection model, you can run the example code provided in the example.ipynb notebook. The notebook uses the provided data in the data folder. You can modify the code to fit your own use case.

## Sample Predictions
![im1](/readme/image_1.png)
<hr>

![im2](/readme/image_2.png)
<hr>

![im3](/readme/image_3.png)
<hr>

![im4](/readme/image_4.png)
<hr>


## Credits
This repository was inspired by the Detectron2 Object Detection on Custom Dataset Tutorial by Facebook Research.

## License
This project is licensed under the [Apache-2.0 License](/LICENSE).
