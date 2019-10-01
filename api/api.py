import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from predict.run_prediction import Predict

UPLOAD_PATH = 'uploads'
ERROR = 'error'

app = Flask(__name__)
app.config['UPLOAD_PATH'] = UPLOAD_PATH


@app.route('/api/predict/', methods=['GET', 'POST'])
def call_predict():
    if request.method == 'POST':
        try:
            if not os.path.exists(UPLOAD_PATH):
                os.makedirs(UPLOAD_PATH)
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_PATH'], filename))
            image_path = os.path.join(UPLOAD_PATH, filename)
            predict = Predict()
            model = predict.load_model()
            response = predict.predict(model, image_path=image_path)
            os.remove(image_path)
            return jsonify(response)
        except Exception as e:
            return jsonify({ERROR: e})

    elif request.method == 'GET':
        return '''
                    <!doctype html>
                    <title>Upload Image File For Prediction</title>
                    <h1>Upload Image File For Prediction</h1>
                    <form method=post enctype=multipart/form-data>
                      <input type=file name=file>
                      <input type=submit value=Upload>
                    </form>
               '''


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
