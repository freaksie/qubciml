import numpy as np
from tensorflow.keras.models import load_model
from src.logger import logging
from src.exception import CustomException
from src.components.model_trainer import WeightConstraint
from src.utils import convert_18bit,convert_27bit
import os
import sys

class ModelConverter():
    def __init__(self, qid:str) -> None:
        self.qid = qid
        self.DATA_PATH = './data'
        self.MODEL_PATH=os.path.join(self.DATA_PATH, self.qid, 'model.keras')
        self.loadModel()
    
    def loadModel(self) -> None:
        try:
            self.model = load_model(self.MODEL_PATH, custom_objects={'WeightConstraint': WeightConstraint}, compile=False)
            logging.info(f"Model loaded from {self.MODEL_PATH}")
        except Exception as e:
            logging.error('No model found for {}'.format(self.qid))
            raise CustomException(e, sys)

    def getParams(self) -> None:
        try:
            all_weights = self.model.get_weights()
            weight,bias=[],[]
            for k,i in enumerate(all_weights):
                if k%2==0:
                    for j in i:
                        for l in j:
                            weight.append(l)
                else:
                    for j in i:
                        bias.append(j)
            
            self.parameters=[]
            for i in weight:
                self.parameters.append(convert_18bit(i))
            for i in bias:
                self.parameters.append(convert_27bit(i))
            logging.info('Model converted for {}'.format(self.qid))
            return self.parameters

        except Exception as e:
            logging.error('Unable to convert parameter for {}'.format(self.qid))
            raise CustomException(e, sys)