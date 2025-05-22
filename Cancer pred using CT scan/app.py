#app.py
import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import UnidentifiedImageError

app = Flask(__name__)

# Define class labels
class_labels = ['Normal', 'Adenocarcinoma', 'Large Cell Carcinoma', 'Squamous Cell Carcinoma']

# Load the pre-trained model
model_path = os.path.join('/Users/ruthvekkannan/Desktop/python/CT scan/lung_cancer_model.h5')
try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading the model: {e}")

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if request.method == 'POST':
            # Get the file from the request
            file = request.files['file']

            # Save the file to a temporary location
            file_path = os.path.join('static', 'uploads', 'temp_image.jpg')
            file.save(file_path)

            # Preprocess the image for the model
            try:
                img = image.load_img(file_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = img_array / 255.0

                # Make predictions using the trained model
                prediction = model.predict(np.expand_dims(img_array, axis=0))

                # Get the predicted class label
                predicted_class_index = np.argmax(prediction)
                predicted_class = class_labels[predicted_class_index]

                # Get predicted probabilities for each class
                predicted_probabilities = {class_labels[i]: prediction[0][i] for i in range(len(class_labels))}

                return render_template('result.html', predicted_class=predicted_class, predicted_probabilities=predicted_probabilities)

            except UnidentifiedImageError as e:
                print(f"UnidentifiedImageError: {e}")
                error_message = "Error processing the image. The file may not be a valid image file."
                return render_template('error.html', error_message=error_message)

            except Exception as e:
                print(f"An error occurred: {e}")
                error_message = f"An error occurred: {e}"
                return render_template('error.html', error_message=error_message)

    except Exception as e:
        print(f"An error occurred: {e}")
        error_message = f"An error occurred: {e}"
        return render_template('error.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)