import os
import sys
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformation_obj(self):
        try:
            num_columns = ['writing_score', 'reading_score']
            cat_columns =  [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            
            num_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='median')),
                    ("scaler", StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy='constant', fill_value='missing')),
                    ("encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Categorical columns: {cat_columns}")
            logging.info(f"Numerical columns: {num_columns}")
            
            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,num_columns),
                ("cat_pipelines",cat_pipeline,cat_columns)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Data loaded successfully.")
            
            preprocessing_obj = self.get_data_transformation_obj()
            target_columns = 'math_score'
            
            input_features_train_df = train_df.drop(columns=[target_columns], axis=1)
            target_features_train_df = train_df[target_columns]
            input_features_test_df = test_df.drop(columns=[target_columns], axis = 1)
            target_features_test_df = test_df[target_columns]
            
            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe.")
            
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            train_arr = np.c_[(input_features_train_arr, np.array(target_features_train_df))]
            test_arr = np.c_[(input_features_test_arr, np.array(target_features_test_df))]
            
            logging.info("Saved preprocessing object")
            
            save_object(
                file_path = self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocess_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)