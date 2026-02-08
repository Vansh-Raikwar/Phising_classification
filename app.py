from flask import Flask, render_template, jsonify, request, send_file
from src.exception import CustomException
from src.logger import logging as lg
import os,sys

from src.pipeline.train_pipeline import TrainingPipeline
from src.pipeline.predict_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html')


@app.route("/train")
def train_route():
    try:

        
        train_pipeline = TrainingPipeline()
        model_path = train_pipeline.run_pipeline()

        lg.info(f"Training completed. Model saved at {model_path}")
        return jsonify({
            "status": "success",
            "message": "Model trained successfully and saved on server.",
            "model_path": model_path
        })

    except Exception as e:
        error = CustomException(e, sys)
        lg.error(error.error_message)
        return jsonify({"error": str(e), "message": error.error_message}), 500

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    
    try:
        if request.method == 'POST':
            prediction_pipeline = PredictionPipeline(request)
            prediction_file_detail = prediction_pipeline.run_pipeline()

            lg.info("prediction completed. Downloading prediction file.")
            return send_file(prediction_file_detail.prediction_file_path,
                            download_name= prediction_file_detail.prediction_file_name,
                            as_attachment= True)
        
        else:
            return render_template('prediction.html')

    except Exception as e:
        error = CustomException(e, sys)
        lg.error(error.error_message)
        return jsonify({"error": str(e), "message": error.error_message}), 500
    


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5000
    
    print(f"App is running on : http://{host}:{port}")
    app.run(host=host, port=port, debug= True)