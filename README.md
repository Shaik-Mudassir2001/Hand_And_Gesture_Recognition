Hand Gesture Recognition with Machine Learning

This project implements a hand gesture recognition system using deep learning and computer vision techniques. The system detects hand gestures from real-time webcam input and performs predefined actions based on the recognized gestures.


Table of Contents

Overview
Project Structure
Dataset
Installation
Usage
Training the Model
Real-Time Gesture Recognition
Model
Technologies Used
Contributing
License


Overview
This project is designed to recognize hand gestures and use them to interact with the computer in real time. The gesture recognition is powered by a Convolutional Neural Network (CNN) that is trained on a dataset of hand gesture images. The project utilizes OpenCV for video capture and pre-processing, TensorFlow for model training and inference, and PyAutoGUI for performing system actions based on recognized gestures.


Dataset
The dataset used for training, testing, and validation consists of gesture images and their corresponding labels. The dataset should be organized as follows:

	train/: Contains training images (JPG) and train_labels.csv.
	test/: Contains testing images (JPG) and test_labels.csv.
	val/: Contains validation images (JPG) and val_labels.csv.
Each CSV file contains two columns: image_filename and label, where image_filename refers to the name of the image and label refers to the gesture class.


Installation
	1. Clone the repository:
		
		git clone https://github.com/your-username/hand_gesture_recognition.git
		cd hand_gesture_recognition
		
	2. Install the required dependencies:
		
		pip install -r requirements.txt
		
	3. Prepare the dataset: Ensure that the dataset is in the correct directory structure under the data/ folder.
	

Usage

Training the Model

	To train the model with your dataset:
		python train_model.py
	
	This will load the training and validation data, train the model, and save the trained model weights to a file (gesture_recognition_weights.h5).
	
Real-Time Gesture Recognition

	Once the model is trained, you can run the real-time gesture recognition system:
		python main.py

	This will start the webcam, detect hand gestures in real-time, and perform actions based on the recognized gestures.


Model
The CNN model is built using TensorFlow and Keras. It consists of several convolutional layers followed by pooling layers, a flatten layer, and fully connected layers. The final output layer has a softmax activation, used to classify hand gestures into one of the predefined classes.

The model is designed to work with input images of size 64x64 with three color channels (RGB). You can adjust the model architecture in gesture_model.py to experiment with different configurations.


Technologies Used

	-> Python: The main programming language used.
	-> TensorFlow/Keras: For building and training the deep learning model.
	-> OpenCV: For image capture and hand detection.
	-> PyAutoGUI: For simulating system actions based on recognized gestures.
	-> NumPy: For data manipulation and processing.
	-> Pandas: For handling CSV files and labels.


Contributing

	Contributions are welcome! If you would like to contribute to this project, please follow these steps:

		Fork the repository.
		Create a new branch (git checkout -b feature-branch).
		Make your changes and commit them (git commit -m 'Add some feature').
		Push to the branch (git push origin feature-branch).
		Create a pull request.
		

License
	
	This project is licensed under the MIT License - see the LICENSE file for details.
