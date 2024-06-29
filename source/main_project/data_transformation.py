import os
import sys
import numpy as np
import pandas as pd
from source.exception import UserException
from source.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from source.commons import save_object
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_path:str = os.path.join('elements','preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_trans_config = DataTransformationConfig()
        
    def get_preprocessor_obj(self):
        try:
            ord_cols = ['work-life balance','job satisfaction','performance rating','education level','job level','company size','company reputation','employee recognition']
            scale_cols = ['gender','years at company','number of promotions','overtime','distance from home','marital status','number of dependents','remote work']

            ordinal = OrdinalEncoder()
            scaler = StandardScaler(with_mean=False)
            
            ord_pipe = Pipeline(steps=[
                ('ordinal',OrdinalEncoder()),
                ('scaler',StandardScaler(with_mean=False))
            ])
            
            scale_pipe = Pipeline(steps=[
                ('scaler',StandardScaler(with_mean=False))
            ])
            logging.info('Building the preprocessor object')

            preprocessor = ColumnTransformer([
                ('ord_pipe',ord_pipe,ord_cols),
                ('scale_pipe',scale_pipe,scale_cols)
            ])
            return preprocessor
        except Exception as e:
            raise UserException(e,sys)
        
    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logging.info('Successfully read trin and test set')
            logging.info('Obtaining preprocessor object')
            
            preprocessor_obj = self.get_preprocessor_obj()
            target_column = 'attrition'
            
            logging.info('Separating into independent and dependent features for both sets')
            input_feature_train_df = train_df.drop(target_column,axis=1)
            dependent_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(target_column,axis=1)
            dependent_feature_test_df = test_df[target_column]
            
            logging.info('Fitting preprocessor object on the data')
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            logging.info('Concatenating splits into 2D arrays')
            
            train_arr = np.c_[input_feature_train_arr,np.array(dependent_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(dependent_feature_test_df)]
            
            save_object(
                file_path = self.data_trans_config.preprocessor_obj_path,
                obj = preprocessor_obj)
            
            return(
                train_arr,
                test_arr,
            )
            
        except Exception as e:
            raise UserException(e,sys)
            
            