from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods =['POST',  'GET'])
def predict_datapoint():
    if request.method == 'POST':
        data = CustomData(
            gender = request.form.get('gender'),
            race_ethnicity = request.form.get('race_ethnicity'),
            parental_level_of_education = request.form.get('parental_level_of_education'),
            lunch = request.form.get('lunch'),
            test_preparation_course = request.form.get('test_preparation_course'),
            reading_score = int(request.form.get('reading_score')),
            writing_score = int(request.form.get('writing_score'))
        )
        
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")
        
        predict_pipeline = PredictPipeline()
        print("mid prediction")
        results = predict_pipeline.predict(pred_df)
        print("After Prediction")
        return render_template('home.html', results=results)
    else:
        return render_template('home.html')


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)