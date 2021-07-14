import cv2
from flask import Flask, request, redirect, render_template, flash
from google.cloud import vision
import io
import numpy as np
import os
import re
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from receipt_functions import allowed_file, draw_boxes, get_sorted_lines, get_document_bounds, read_costco

UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
client = vision.ImageAnnotatorClient()

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('There is no such file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No file')
            return redirect(request.url)
        if file and allowed_file(file.filename, ALLOWED_EXTENSIONS):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            pred_store = 'failed'
            price = 'failed'
            date_dt = 'failed'

            # #predict the store from the receipt image
            # img = image.load_img(filepath, grayscale=True)
            # img = image.img_to_array(img)
            # data = np.array([img])
            # result = model.predict(data)[0]
            # predicted = result.argmax()
            # pred_store = classes[predicted]
            pred_store = 'costco'

            #get result from google vision API
            with io.open(filepath, 'rb') as image_file:
                content = image_file.read()
            image_2 = vision.Image(content=content)
            response = client.document_text_detection(image=image_2)

            if pred_store == 'costco':
                price, date_dt = read_costco(response)
            else:
                pass

            return render_template("index.html", store=pred_store, date=date_dt, price=price, uploaded_image=filepath)

    return render_template("index.html", store="xx", date="input date", price="price")

if __name__ == '__main__':
    app.run()
