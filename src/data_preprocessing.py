import os
import logging
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder 

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)

logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_preprocessing.log')

file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -%(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def preprocess_df(df,colum_name):
    """ preprocessing the data"""
    try:
       logger.debug('Starting preprocessing for dataframe')
       encoder=LabelEncoder()
       df[colum_name]=encoder.fit_transform(df[colum_name])
       logger.debug("Transforming the string colum into numbers")
        
       df=df.drop_duplicates(keep='first')
       logger.debug('remove duplicates ')
       return df
    except KeyError as e:
        logger.error('Column not found :%s',e)
        raise
    except Exception as e:
        logger.error('Error during the preprocessing %s',e)
        raise
def main(colum_name='Gender'):
    """Main function to load raw data and preprocess it"""
    try:
        train_data=pd.read_csv('./data/raw/train.csv')
        test_data=pd.read_csv('./data/raw/test.csv')
        logger.debug('Data loaded preperly')
    
        #transform the data
        train_processed_data=preprocess_df(train_data,colum_name)
        test_processed_data=preprocess_df(test_data,colum_name)
    
        #store the data inside the data/preprocessed

        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)
    
        train_processed_data.to_csv(os.path.join(data_path,'trained_processed.csv'),index=False)
        test_processed_data.to_csv(os.path.join(data_path,'test_processed.csv'),index=False)
    
        logger.debug('Processed data saved to %s',data_path)
        
    except FileNotFoundError as e:
        logger.error('File not found :%s',e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data %s',e)
    except Exception as e:
        logger.error('Faied to complter the data transformation process %s',e)
        print(e)
        
if __name__=="__main__":
    main()
    
    