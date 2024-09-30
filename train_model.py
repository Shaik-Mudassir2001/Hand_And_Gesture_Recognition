from gesture_model import create_model
from utils import load_dataset
from sklearn.model_selection import train_test_split

# Load training and validation data
X_train, y_train = load_dataset('data/train', 'data/train_labels.csv')
X_val, y_val = load_dataset('data/val', 'data/val_labels.csv')

# Define input shape and number of classes
input_shape = X_train.shape[1:]  # e.g., (64, 64, 3)
num_classes = y_train.shape[1]  # Number of gesture classes

# Create the model
model = create_model(input_shape, num_classes)

# Train the model
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Save the trained model weights
model.save_weights('gesture_recognition_weights.h5')
