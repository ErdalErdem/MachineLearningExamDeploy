import sys
from PIL import Image
import numpy as np
import tensorflow as tf
import warnings

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*NotOpenSSLWarning.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*Compiled the loaded model.*")

def load_model():
    # Load the pre-trained model
    model = tf.keras.models.load_model('fruit_model.h5')
    return model

def preprocess_image(image_path):
    # Load the image
    image = Image.open(image_path)
    # Resize the image to the input size expected by the model (assuming 224x224 here)
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the image array
    image_array = image_array / 255.0
    # Add a batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image_path):
    # Load the model
    model = load_model()
    # Preprocess the image
    image_array = preprocess_image(image_path)
    # Make predictions
    predictions = model.predict(image_array)
    print(f"Predictions: {predictions}")  # Debug: print the predictions
    # Get the index of the class with the highest probability
    predicted_class_index = np.argmax(predictions)
    print(f"Predicted class index: {predicted_class_index}")  # Debug: print the predicted class index
    # Map the class index to the class name (complete list of class names)
    class_names = [
        'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
        'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
        'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
        'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
        'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon'
    ]
    # Check if predicted_class_index is within the range of class_names list
    if predicted_class_index < len(class_names):
        predicted_class = class_names[predicted_class_index]
    else:
        predicted_class = "Unknown"
    return predicted_class

if __name__ == "__main__":
    image_path = sys.argv[1]
    result = predict(image_path)
    print(result)
