# Emerging Technologies
This repository supplements the Emerging Technologies course. By the end of the course, the learner is expected to demonstrate Artificial Intelligence-driven solutions to solve engineering problems.


## Course Outline

 | Activities | Goals | Resources |
| -------- | -------- | -------- |
| Python Basics and Exercises | To enhance practical skills in basic Python syntax |  [activities/Lastname_Python_Basics_and_Exercises.ipynb](activities/Lastname_Python_Basics_and_Exercises.ipynb) |
| Advanced Python and Exercises | To learn NumPy and Pandas libraries | [activities/Lastname_Advanced_Python_and_Exercises.ipynb](activities/Lastname_Advanced_Python_and_Exercises.ipynb) |
| Coin Counting | To master OpenCV's image processing available APIs. Read the [coin_counting/README.md](activities/opencv_samples/coin_counting/README.md) for the details on the sample images. | [activities/opencv_samples/coin_counting/coin_counting.py](activities/opencv_samples/coin_counting/coin_counting.py) |
| Color Tracking | To master OpenCV's color tracking capability. Read the [color_tracking/README.md](activities/opencv_samples/color_tracking/README.md). | [activities/opencv_samples/color_tracking/color_tracking.py](activities/opencv_samples/color_tracking/color_tracking.py) |
| Face Tracking | To implement face detection and tracking using Haar cascades. Read the [face_tracking/README.md](activities/opencv_samples/face_tracking/README.md) | [activities/opencv_samples/face_tracking/facetracking.py](activities/opencv_samples/face_tracking/facetracking.py) |
| Eye Tracking | To implement eye detection and tracking using Haar cascades. Read the [eye_tracking/README.md](activities/opencv_samples/eye_tracking/README.md) | [activities/opencv_samples/eye_tracking/eyetracking.py](activities/opencv_samples/eye_tracking/eyetracking.py) | 
| Hand Detection | To demonstrate Mediapipe's detection model to determine the left and right hands, with their other useful information. | [activities/opencv_samples/hand_detection/hand.py](activities/opencv_samples/hand_detection/hand.py) |
| Hand Gesture Detection | To demonstrate  Mediapipe's detection model to detect and classify specific hand gestures.  | [activities/opencv_samples/hand_gesture_detection/gesture.py](activities/opencv_samples/hand_gesture_detection/gesture.py) |
| Facial Emotion Recognition | To demonstrate Mediapipe's detection model to identify facial emotions. Read the [facial_emotion_recognition/README.md](activities/opencv_samples/facial_emotion_recognition/README.md)|[activities/opencv_samples/facial_emotion_recognition/facial_emotion.py](activities/opencv_samples/facial_emotion_recognition/facial_emotion.py)|
| Gender and Age Recognition | To demonstrate deep learning-based models to identify gender and age. Read the [gender_and_age_detection/README.md](activities/opencv_samples/gender_and_age_detection/README.md)| [activities/opencv_samples/gender_and_age_detection/gender_age.py](activities/opencv_samples/gender_and_age_detection/gender_age.py) |
| Pose Estimation | To demonstrate Mediapipe's pose detection model to estimate the joints of the body's pose. Read the [pose_estimation/README.md](activities/opencv_samples/pose_estimation/README.md) | [activities/opencv_samples/pose_estimation/pose.py](activities/opencv_samples/pose_estimation/pose.py)|
| Classification on MNIST Digits | To prepare, train, test, and deploy a Pytorch pretrained machine learning model for classifying MNIST Digits. Use the three notebooks. | [activities/machine_learning_samples/Lastname_ML_Notebook_1.ipynb](activities/machine_learning_samples/Lastname_ML_Notebook_1.ipynb)  [activities/machine_learning_samples/Lastname_ML_Notebook_2.ipynb](activities/machine_learning_samples/Lastname_ML_Notebook_2.ipynb)  [activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb](activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb) |
| Classification on Labeled Faces in the Wild | To prepare, train, test, and deploy a Pytorch pretrained machine learning model for classifying faces. Revise the notebook 3 to import LFW dataset. | [activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb](activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb) |
| Regression on Boston Housing Prices | To prepare, train, test, and deploy a Pytorch pretrained machine learning model for regressing Boston housing prices. Revise the notebook 3 to import Boston dataset.  | [activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb](activities/machine_learning_samples/Lastname_ML_Notebook_3.ipynb) |
| Classification on CIFAR-10 | To prepare, train, test, and deploy a Pytorch pretrained deep learning model for classifying CIFAR-10 images. | [activities/deep_learning_samples/classification_cifar10/](activities/deep_learning_samples/classification_cifar10/) |
| Classification on a custom dataset | To prepare, train, test, and deploy a Pytorch pretrained deep learning model for classifying images from a custom dataset of your own choice | [activities/deep_learning_samples/classification_custom/](activities/deep_learning_samples/classification_custom/) |
| Classification on a custom dataset using YOLO | To prepare, train, test, and deploy a YOLO model for classifying images from a custom dataset of your own choice | [activities/deep_learning_samples/classification_custom_yolo/](activities/deep_learning_samples/classification_custom_yolo/) |
| Segmentation on a custom dataset using YOLO | To prepare, train, test, and deploy a YOLO model for performing image segmentation from a subset of pavement panel dataset. Read the [segmentation_yolo11/README.md](activities/deep_learning_samples/segmentation_yolo11/README.md) on how to prepare the dataset. However, if you want to access, use, and cite our complete image datasets, kindly check out our [NoMUR repository](https://github.com/earlaleluya/NoMUR). | [activities/deep_learning_samples/segmentation_yolo11/](activities/deep_learning_samples/segmentation_yolo11/) |




##Create Python Virtual Environment
# Create a virtual environment
python3 -m venv myenv

# Activate the virtual environment
source myenv/bin/activate

# Now install numpy
pip install numpy
