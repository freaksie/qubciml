import os
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np


class DataIngestion():
    '''
    ## This class is responsible for data loading and validation.
    ### Arguments:
        - qid - string - ID of current qubit.
        - data - numpy array - Integarted Complex ACC data per shots of current qubit.
        - target - numpy array - State per shots of current qubit.
    ### Description:
        1. Checks if all the element numpy array is complex.
        2. Checks if first dimension of data and target are equal.
        3. Changes format of data from complex to two dimension.
        4. Checks total number of unique classes in target.
        5. Saves data.

    '''
    def __init__(self, qid:str, data:np.ndarray, target:np.ndarray):
        self.qid = qid
        self.data = data
        self.target = target
        self.data_new = None
        self.check_format()
        self.check_shape()
        self.change_format()
        self.check_class()
        self.save_data()
        
    def check_format(self):
        # check if all the element numpy array is complex
        if not np.all(np.iscomplex(self.data)):
            logging.info('{} data is not in complex format'.format(self.qid))
            raise CustomException("{} data is not in complex format".format(self.qid),sys)
    
    # check if first dimension of data and target are equal
    def check_shape(self):
        if self.data.shape[0] != self.target.shape[0]:
            logging.info('Input data and target are not of equal dimension for {}'.format(self.qid))
            raise CustomException("Data and target are not of equal dimension for {}".format(self.qid), sys)
        
    def change_format(self):
        try:
            self.data_new = np.stack((self.data.real,self.data.imag),axis=1)
            logging.info('{} data changed to two dimension'.format(self.qid))
        except Exception as e:
            logging.info('Error in changing data format for {}' .format(self.qid))
            raise CustomException(e, sys)
    
    #check total number of unique classes in target
    def check_class(self):
        CLASSES = len(np.unique(self.target))
        if CLASSES < 2 or CLASSES > 3:
            logging.info('Number of classes is not 2 or 3 in {}'.format(self.qid))
            raise CustomException("Number of classes is not 2 or 3 in {}".format(self.qid), sys)
    
    #Save data
    def save_data(self):
        try:
            os.makedirs(f'./data/{self.qid}', exist_ok=True)
            np.save(f'./data/{self.qid}/X.npy', self.data_new)
            np.save(f'./data/{self.qid}/Y.npy', self.target)
            logging.info('{} data saved'.format(self.qid))
        except Exception as e:
            logging.info('Error in saving data for {}'.format(self.qid))
            raise CustomException(e, sys)
    




