import shutil
import os,sys
import pandas as pd
from src.logger import logging

from src.exception import CustomException
import sys
from flask import request
from src.constant import *
from src.utils.main_utils import MainUtils

from dataclasses import dataclass
        
        
@dataclass
class PredictionFileDetail:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    prediction_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)



class PredictionPipeline:
    def __init__(self, request: request):

        self.request = request
        self.utils = MainUtils()
        self.prediction_file_detail = PredictionFileDetail()



    def save_input_files(self)-> str:

        """
            Method Name :   save_input_files
            Description :   This method saves the input file to the prediction artifacts directory. 
            
            Output      :   input dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """

        try:
            pred_file_input_dir = os.path.join("prediction_artifacts")
            os.makedirs(pred_file_input_dir, exist_ok=True)

            input_csv_file = self.request.files['file']
            filename = input_csv_file.filename
            if filename == '':
                filename = "input_file.csv"
            
            pred_file_path = os.path.join(pred_file_input_dir, filename)
            
            logging.info(f"Saving input file to: {pred_file_path}")
            input_csv_file.save(pred_file_path)
            
            file_size = os.path.getsize(pred_file_path)
            logging.info(f"Saved file size: {file_size} bytes")
            
            if file_size == 0:
                raise Exception(f"Uploaded file {filename} is empty.")

            return pred_file_path
        except Exception as e:
            raise CustomException(e,sys)

    def predict(self, features):
        try:
            model_path = "model.pkl"
            try:
                logging.info("Attempting to download model from S3...")
                model_path = self.utils.download_model(
                    bucket_name=AWS_S3_BUCKET_NAME,
                    bucket_file_name="model.pkl",
                    dest_file_name="model.pkl",
                )
                logging.info("Model downloaded successfully from S3.")
            except Exception as s3_error:
                logging.warning(f"S3 Download failed: {str(s3_error)}. Checking for local model fallback.")
                if os.path.exists("model.pkl"):
                    logging.info("Found local model.pkl. Proceeding with local model.")
                    model_path = "model.pkl"
                else:
                    logging.error("Neither S3 model nor local model.pkl found.")
                    raise Exception("Model not found in S3 and no local model.pkl available.") from s3_error

            logging.info(f"Loading model from: {model_path}")
            model = self.utils.load_object(file_path=model_path)
            logging.info(f"Model loaded successfully. Model type: {type(model)}")

            logging.info(f"Input features shape: {features.shape}")
            preds = model.predict(features)
            logging.info(f"Prediction successful. Preds count: {len(preds)}")

            return preds

        except Exception as e:
            logging.error(f"Error during prediction: {str(e)}")
            raise CustomException(e, sys)
        
    def get_predicted_dataframe(self, input_dataframe_path:str):

        """
            Method Name :   get_predicted_dataframe
            Description :   this method returns the dataframw with a new column containing predictions

            
            Output      :   predicted dataframe
            On Failure  :   Write an exception log and then raise an exception
            
            Version     :   1.2
            Revisions   :   moved setup to cloud
        """
   
        try:

            prediction_column_name : str = TARGET_COLUMN
            
            logging.info(f"Reading CSV from: {input_dataframe_path}")
            if not os.path.exists(input_dataframe_path):
                raise Exception(f"File not found: {input_dataframe_path}")
                
            file_size = os.path.getsize(input_dataframe_path)
            logging.info(f"File size before reading: {file_size} bytes")
            
            input_dataframe: pd.DataFrame = pd.read_csv(input_dataframe_path)
            
            if input_dataframe.empty:
                 raise Exception(f"Dataframe is empty after reading {input_dataframe_path}")
            
            logging.info(f"DataFrame columns: {input_dataframe.columns.tolist()}")
            predictions = self.predict(input_dataframe)
            input_dataframe[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {0:'phising', 1:'safe'}

            input_dataframe[prediction_column_name] = input_dataframe[prediction_column_name].map(target_column_mapping)
            
            os.makedirs( self.prediction_file_detail.prediction_output_dirname, exist_ok= True)
            input_dataframe.to_csv(self.prediction_file_detail.prediction_file_path, index= False)
            logging.info(f"Predictions saved to: {self.prediction_file_detail.prediction_file_path}")

        except Exception as e:
            logging.error(f"Error in get_predicted_dataframe: {str(e)}")
            raise CustomException(e, sys) from e
        

        
    def run_pipeline(self):
        try:
            input_csv_path = self.save_input_files()
            self.get_predicted_dataframe(input_csv_path)

            return self.prediction_file_detail


        except Exception as e:
            raise CustomException(e,sys)
            
        

 
        

        