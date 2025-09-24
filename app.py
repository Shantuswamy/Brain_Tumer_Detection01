import os
from flask import Flask, request, render_template, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# ------------------ Configuration ------------------
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASS_NAMES = ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary']

SOLUTIONS = {
    'No Tumor': "No tumor detected. Maintain a healthy lifestyle and regular checkups.",
    'Glioma': "Consult a neurologist for further imaging and treatment options. Early intervention is important.",
    'Meningioma': "Consult a specialist for assessment. Surgery or monitoring may be recommended.",
    'Pituitary': "Consult an endocrinologist and neurologist. Hormone tests and imaging may be needed."
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ------------------ Load Model ------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "brain_tumor_resnet50.keras")
model = tf.keras.models.load_model(MODEL_PATH)
model.predict(np.zeros((1, 128, 128, 3)))  # Build the model

# ------------------ Helpers ------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------ Routes ------------------
@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/')
def main():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """ Single image prediction """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image(file_path)
        preds = model.predict(img_array)[0]
        pred_idx = np.argmax(preds)
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(preds[pred_idx])
        class_confidences = dict(zip(CLASS_NAMES, map(float, preds)))
        solution = SOLUTIONS.get(pred_class, "Consult a medical professional for advice.")

        return jsonify({
            'prediction': pred_class,
            'confidence': confidence,
            'class_confidences': class_confidences,
            'solution': solution
        })
    else:
        return jsonify({'error': 'Invalid file type'}), 400


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    true_labels = request.form.getlist('true_labels[]')  # true labels from JS
    results = []

    for idx, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])
            solution = SOLUTIONS.get(pred_class, "Consult a medical professional.")

            results.append({
                'filename': filename,
                'true_label': true_labels[idx],
                'prediction': pred_class,
                'confidence': confidence,
                'solution': solution
            })

    return jsonify({'results': results})




@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/predict_batch_with_labels', methods=['POST'])
def predict_batch_with_labels():
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400

    files = request.files.getlist('files')
    true_labels = request.form.getlist('true_labels[]')  # send as array from JS
    results = []

    for idx, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # preprocess + predict
            img_array = preprocess_image(filepath)
            preds = model.predict(img_array)[0]
            pred_idx = np.argmax(preds)
            pred_class = CLASS_NAMES[pred_idx]
            confidence = float(preds[pred_idx])
            solution = SOLUTIONS.get(pred_class, "Consult a medical professional.")

            results.append({
                'filename': filename,
                'true_label': true_labels[idx],
                'prediction': pred_class,
                'confidence': confidence,
                'solution': solution
            })

    return jsonify({'results': results})



# ------------------ Run App ------------------
if __name__ == "__main__":
    app.run(debug=True)
