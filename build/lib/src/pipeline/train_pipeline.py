import numpy as np
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransform
from src.components.model_trainer import ModelTrainer
from src.components.model_converter import ModelConverter


class Trainer():
    '''
    ## This class is used to train the model
    ### Arguments:
       - qid - string - ID of current qubit.
       - data - numpy array - Integarted Complex ACC data per shots of current qubit.
       - target - numpy array - State per shots of current qubit.

    ### Functions:
        - start() - Returns list of model parameters compatible for FPGA.
    '''
    def __init__(self, qid:str, data:np.ndarray, target:np.ndarray):
        self.qid = qid
        self.data = data
        self.target = target
        self.parameters=None
    
    def start(self) -> list:
        '''
        ## This function starts the training for given Trainer instance.
        ### Arguments:
            - None
        ### Returns:
            - list: list of model parameters compatible for FPGA.
        '''
        logging.info('********Starting the training for {}*************'.format(self.qid))
        logging.info('Data Ingestion started for {}'.format(self.qid))
        DataIngestion(self.qid, self.data, self.target)
        logging.info('Data Ingestion completed for {}'.format(self.qid))

        #Call data scaler
        logging.info('Data scaling started for {}' .format(self.qid))
        transformer= DataTransform(self.qid)
        logging.info('Data scaling completed for {}' .format(self.qid))
        self.nI, self.nQ, self.mean_I, self.mean_Q = transformer.getScalingParameter()
        logging.info('Fetched Scaling parameter for {}'.format(self.qid))


        #Call model trainer
        logging.info('Model training started for {}'.format(self.qid))
        ModelTrainer(self.qid)
        
        #Call model converter
        logging.info('Model conversion started for {}'.format(self.qid))
        self.parameters= ModelConverter(self.qid).getParams()
        self.parameters.extend([self.nI, self.nQ, self.mean_I, self.mean_Q])
        logging.info('Parameter loaded for {}'.format(self.qid))
        
        #Return list of parameter
        return np.array(self.parameters)
    
