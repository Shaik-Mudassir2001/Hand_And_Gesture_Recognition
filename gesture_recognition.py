import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from gesture_model import create_model

# Load pre-trained model
input_shape = (64, 64, 3)  # Adjust according to your input shape
num_classes = 5  # Adjust according to your number of classes
model = create_model(input_shape, num_classes)
model.load_weights('gesture_recognition_weights.h5')

def preprocess_frame_for_model(frame):
    frame_resized = cv2.resize(frame, (64, 64))
    frame_array = img_to_array(frame_resized) / 255.0  # Normalize
    return np.expand_dims(frame_array, axis=0)

def recognize_gesture(frame):
    processed_frame = preprocess_frame_for_model(frame)
    prediction = model.predict(processed_frame)
    gesture_class = np.argmax(prediction)
    return gesture_class
