from flask import Flask, render_template, request
import numpy as np
import cv2
from sklearn.cluster import KMeans
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_colors(image_path, num_colors=10):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))  # Flatten the image to (pixels, 3)

    kmeans = KMeans(n_clusters=num_colors)
    labels = kmeans.fit_predict(img)
    counts = np.bincount(labels)

    # Sort colors by frequency
    sorted_idx = np.argsort(counts)[::-1]
    colors = kmeans.cluster_centers_.astype(int)[sorted_idx]
    percentages = counts[sorted_idx] / counts.sum() * 100

    hex_colors = ['#%02x%02x%02x' % tuple(c) for c in colors]
    percents_rounded = [round(p, 2) for p in percentages]

    return list(zip(hex_colors, percents_rounded))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        img_file = request.files['image']
        filename = secure_filename(img_file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img_file.save(path)

        color_data = get_colors(path)
        
        return render_template('result.html', color_data=color_data, image=path)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
