import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (ensure the path is correct)
model = tf.keras.models.load_model('C:/Users/aravi/OneDrive/Desktop/codeClause/project 4/crop_disease_model.h5')

# Define allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the image before feeding it to the model
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize image to model's expected size
    img_array = image.img_to_array(img)  # Convert image to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)
        
        # Predict the class using the model
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions)
        
        class_names = ['Pepperbell Healthy', 'Pepperbell Infected', 'Potato Healthy', 'Potato Infected','Tomato Healthy', 'Tomato Infected']
        
        predicted_class = class_names[class_idx]

        return render_template('result.html', predicted_class=predicted_class, image_path=file_path)

    return 'Invalid file format. Please upload a valid image.'

if __name__ == '__main__':
    app.run(debug=True)

