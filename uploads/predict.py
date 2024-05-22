import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import os

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

def load_model():
    model = tf.keras.models.load_model('fruit_model.h5')  # Make sure the path is correct
    return model

def preprocess_image(frame):
    image = Image.fromarray(frame)
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(frame, model):
    image_array = preprocess_image(frame)
    predictions = model.predict(image_array)
    predicted_class_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100  # Convert probability to percentage
    class_names = ['apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot',
                   'cauliflower', 'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger',
                   'grapes', 'jalepeno', 'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange',
                   'paprika', 'pear', 'peas', 'pineapple', 'pomegranate', 'potato', 'raddish',
                   'soy beans', 'spinach', 'sweetcorn', 'sweetpotato', 'tomato', 'turnip', 'watermelon']
    return class_names[predicted_class_index], confidence

def capture_and_predict():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    
    model = load_model()  # Load the model once, outside the loop

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        
        prediction, confidence = predict(frame, model)
        cv2.putText(frame, f"{prediction}: {confidence:.2f}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # Draw a circle in the center of the frame
        height, width, _ = frame.shape
        center_coordinates = (width // 2, height // 2)
        radius = 100  # Adjust radius as needed
        color = (0, 255, 0)  # Green color
        thickness = 2  # Thickness of the circle
        cv2.circle(frame, center_coordinates, radius, color, thickness)

        cv2.imshow('Camera Feed', frame)

        if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit the loop
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_and_predict()
