import numpy as np
import pandas as pd 
import os
from sklearn.model_selection import train_test_split 
import logging
import yaml

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


#logging configuration
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_hendler=logging.StreamHandler()
console_hendler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter=logging.Formatter('%(asctime)s- %(name)s - %(levelname)s -%(message)s')
console_hendler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_hendler)
logger.addHandler(file_handler)

def load_params(params_path: str) ->dict:
    """load parameters from YAML file"""
    
    try:
        with open(params_path,'r') as file:
            params=yaml.safe_load(file)
        logger.debug("Parameters restrived from %s",params_path)
        return params
    except FileNotFoundError:
        logger.error('File Not found %s', params_path)
        raise
    
    except yaml.YAMLError as e:
        logger.error('YAML error: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error :%s',e)
        raise
    
def load_data(data_url: str)->pd.DataFrame:
    """load the data from csv file"""
    
    try:
        df=pd.read_csv("C:/Users/HP/Downloads/calories_exercise_combined.csv")
        logger.debug('Data loaded from %s',"C:/Users/HP/Downloads/calories_exercise_combined.csv")
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file :%s',e)
        raise 
    except Exception as e:
        logger.error('Unexpected error occured while loading the data %s',e)
        raise
    
def preprocess_data(df: pd.DataFrame)-> pd.DataFrame:
    """prepress the data"""
    
    try:
        df.drop('User_ID',inplace=True,axis=1)
        logger.debug('User id is been droped')
        return df
    except KeyError as e:
        logger.error('Missing the column in the dataframe %s',e)
    except Exception as e:
        logger.error('Unexpected error %s',e)
        raise
    
def save_data(train_data:pd.DataFrame, test_data:pd.DataFrame, data_path:str)->None:
    """saving the train and test datasets"""
    
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        
        
        logger.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logger.error('Unexpected error occured while saving the data %s',e)
        raise
    
def main():
    try:
        params=load_params(params_path='params.yaml')
        test_size=params['data_ingestion']['test_size']
        data_path="C:/Users/HP/Downloads/calories_exercise_combined.csv"
        df=load_data(data_path)
        fina_df=preprocess_data(df)
        train_df,test_df=train_test_split(fina_df,test_size=test_size,random_state=1)
        save_data(train_df,test_df,data_path='./data')
    except Exception as e:
        logger.error('Failed to compute the data ingestion process: %s',e)
        print(e)
        
    
if __name__=='__main__':
    main()
    
        