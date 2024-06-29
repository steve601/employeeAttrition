import os
import sys
from source.exception import UserException
from source.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataCollectionConfig:
    # where the outputs will be saved
    raw_data_path:str = os.path.join('elements','raw.csv')
    train_data_path:str = os.path.join('elements','train.csv')
    test_data_path:str = os.path.join('elements','test.csv')
    
class DataCollection:
    def __init__(self):
        self.collection_config = DataCollectionConfig()
    def start_collection(self):
        logging.info('Start of data ingestion method')
        try:
            data = pd.read_csv('notebook\data\processed_data.csv')
            logging.info('Data collected/read successfully')
            
            # now let's create folders for all these data
            os.makedirs(os.path.dirname(self.collection_config.raw_data_path),exist_ok=True)
            #saving the raw data
            data.to_csv(self.collection_config.raw_data_path,index=False,header=True)
            logging.info('Train Test Split starting...')
            train_data,test_data = train_test_split(data,test_size=0.25,random_state=42)
            # savng the train and test data
            train_data.to_csv(self.collection_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.collection_config.test_data_path,index=False,header=True)
            
            logging.info('Data colection and saving complete!')
            
            return(
                self.collection_config.train_data_path,
                self.collection_config.test_data_path
            )
        except Exception as e:
            raise UserException(e,sys)
        
if __name__ == '__main__':
    obj = DataCollection()
    obj.start_collection()