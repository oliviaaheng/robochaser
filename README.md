# RoboChaser

The code for this project is separated into 2 locations and 4 parts total.

In this GitHub repository.. there is YoloV3 Mimic and YoloV8 Custom, and in google 
collaboratory, there is YoloV8 Custom Dataset.ipynb, and Robochaser.ipynb. The links 
to the notebooks are...

YoloV8 Custom Dataset.ipynb:

https://colab.research.google.com/drive/1BWBPR-3JByzCpfTfLLK-QWccXgno4xnJ?usp=sharing 

Robochaser.ipynb:

https://colab.research.google.com/drive/1kXrwOf2AtftOlT595oGaUWyG50dLOMRg?usp=sharing

### Brief Overview of Parts

#### YoloV8 Custom Dataset
This google collaboratory notebook was used to upload our dataset from Roboflow, 
and then train the state of the art YoloV8 model to detect our toycar, and give 
accurate bounding box predictions. More details can be found in the notebook.

#### Robochaser
This notebook defines a custom model architechture that uses resnet50 as a backbone 
for feature extraction, followed by dense layers. After realizing our data was not 
sufficient to train a Yolo-like model from scratch, we realized we could acheive 
our desired results more explicitly by segmenting the image and finding the image 
segmentation with the highest probability of being a car. More details can be 
found in the notebook.

#### YoloV3 Mimic
Our attempt to recreate the YoloV3 architecture in python using TensorFlow and 
train it on our custom model. It contains two versions, of which the finder tries to 
accurately model YoloV3's structure of a feature extraction backbone connected to 
3 different sizes of image detection in a feature pyramid network. 

#### YoloV8 Custom
This package downloads the YoloV8 weights that we trained in YoloV8 Custom 
Dataset.ipynb, and then connects to the iRobot and the computers webcam to dynamically 
take picture and send commands to the robot based on where the toycar is in the x 
dimension of the image.

## YoloV3 Mimic
### Dataset
This part of the project uses our bounding box dataset of 1900 images of the car, 
and 1300 images with no car. The data was uploaded into CVAT.ai, but transferred to 
Roboflow, where we added augmentations to create a set of 5000 total images.

### Custom Metrics
In order to judge the accuracy of both a car with a bounding box, and a 'none' image 
with no bounding box, a custom loss and accuracy function was designed to handle both 
cases,while maintaining differentiability.

If the label of the image was a car, then the loss was described by the bounding box loss, 
and the classification or confidence loss; however, it the label was no car, the the loss 
was only described by the classification or confidence loss. The accuracy was implemented 
in a similar fashion. 

After several failed attempts at training the full bounding box model, those functions 
were modified to focus only on classification to see if the model could at least capture 
that information.

### Finder Architecture
Finder (finder.py) attempts to mimic the architecture of YoloV3.

1. Capture a residual of the image

2. Feature extraction backbone with resnet

3. Add back residual and capture another residual

FEATURE PYRAMID NETWORK

4. Convolution to detect small instances

5. Upsample to medium size

6. Add back residual

7. Convolution to detect medium instances

8. Upsample to large size

9. Add back residual

10. Convolution to detect large instances

PREDICTION

11. Pass all sizes through dense layers

12. Sum predictions and sigmoid activation

##### Results
Initial attempts to train this model led to mode collapse to 1. (the output no matter 
the image, was always one). We believed that a big cause for this was that we realized
our dataset at this point consisted of roughly 95% images of the car, and so by guessing
a car every time, the model could achieve high accuracy. This, however, was not fixed by
the addition of more blank images. We decided to simplify the architecture and see if we
first simply classify the images.

### Simple Architecture
Simple (simple.py) attempts to form a more basic version of Yolo for classification.

1. Capture a residual of the image

2. Feature extraction backbone with resnet

3. Add back residual

4. Convolutional detection layer

5. Fully connected linear network

6. Sigmoid activation

##### Results
Similar to with Finder, this architecture was unable to capture the toy car, and so 
we decided to try to train our model on YoloV8, to see if the dataset was the issue,
and not the model itself. In actuality, although YoloV8 was able to train on our data,
we think the data we had was still far from sufficient to train a Yolo-like model
from scratch.

## YoloV8 Custom
This package loads the YoloV8 model we trained and the calls on the robot to move
according to the x value of the bounding box.

#### Model
The model.py file loads our weights from 'best.pt', which is the file for the weights 
for YoloV8, which achieved the highest single epoch accuracy during training in the ipynb.

#### Create Movement
In create_movement.py, we connect to the iRobot, capture images from the webcam, and,
in a while loop, generate predictions on those images through our model and convert
that prediction into an angle for our robot to turn. 
