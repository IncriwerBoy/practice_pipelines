import os
import sys
from src.logger import logging
from src.exception import CustomException
from sklearn.model_selection import train_test_split
import pandas as pd
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import Modeltrainer
@dataclass
class DataIngestionConfig:
    train_data_path = os.path.join('artifacts', 'train.csv')
    test_data_path = os.path.join('artifacts', 'test.csv')
    raw_data_path = os.path.join('artifacts', 'data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            df = pd.read_csv('stud.csv')
            logging.info("Reading data as Dataset")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info("Train Test Split Initiated")
            
            train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
            
            train_df.to_csv(self.ingestion_config.train_data_path, index = False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__=='__main__':
    obj = DataIngestion()
    train_arr, test_arr = obj.initiate_data_ingestion()
    
    preprocess_obj = DataTransformation()
    train_arr, test_arr, _ = preprocess_obj.initiate_data_transformation(train_arr, test_arr)
    
    model = Modeltrainer()
    r2_sq = model.initiate_model_train(train_arr, test_arr)
    print(f"R-squared value: {r2_sq}")