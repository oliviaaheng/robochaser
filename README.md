# robochaser

The code for this project is separated into 2 locations and 4 parts total.

In this github repository.. there is YoloV3 Mimic and YoloV8 Custom, and in google collaboratory, there is YoloV8 Custom Dataset.ipynb, and Classifier.ipynb. The links to the notebooks are...

YoloV8 Custom Dataset.ipynb: 
https://colab.research.google.com/drive/1BWBPR-3JByzCpfTfLLK-QWccXgno4xnJ?usp=sharing 

Classifier.ipynb:
https://colab.research.google.com/drive/1kXrwOf2AtftOlT595oGaUWyG50dLOMRg?usp=sharing

### Brief Overview of Parts

#### YoloV8 Custom Dataset
This google collaboratory notebook was used to upload our dataset from Roboflow, and then train the state of the art YoloV8 model to detect our toycar, and give accurate bounding box predictions.

#### Classifier
This notebook defines a custom model architechture that uses resnet50 as a backbone for feature extraction, followed by dense layers. After realizing our data was not sufficient to train a Yolo-like model from scratch, we realized we could acheive our desired results more explicitly by segmenting the image and finding the image segmentation with the highest probability of being a car.

#### YoloV3 Mimic
Our attempt to recreate the YoloV3 architechture in python using tensorflow and train it on our custom model. It contains two versions, of which finder tries to accurately model YoloV3's structure of a feature extraction backbone connected to 3 different sizes of image detection in a feature pyramid network. 

#### YoloV8 Custom

### YoloV3 Mimic

### YoloV8 Custom
