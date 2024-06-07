import numpy as np
from tensorflow.keras.models import Sequential,load_model # type: ignore
from tensorflow.keras import Input # type: ignore
from tensorflow.keras.layers import Dense # type: ignore
from tensorflow.keras.callbacks import ModelCheckpoint # type: ignore
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.constraints import Constraint # type: ignore
from sklearn.metrics import confusion_matrix
from src.logger import logging
from src.exception import CustomException
import os
import sys



class WeightConstraint(Constraint):
    def __init__(self, min_value=-2, max_value=2, precision=6):
        self.min_value = min_value
        self.max_value = max_value
        self.precision = precision

    def __call__(self, w):
        clipped = K.clip(w, self.min_value, self.max_value)
        return K.round(clipped * 10**self.precision) / 10**self.precision

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value, 'precision': self.precision}
    
class ModelTrainer:
    def __init__(self, qid: str):
        self.qid = qid
        self.DATA_PATH = './data'
        self.load_data()
        self.getModel()
        self.train()
        self.evaluate()

    def load_data(self):
        try:
            self.X_train = np.load(os.path.join(self.DATA_PATH, self.qid, 'X_train.npy'))
            self.Y_train = np.load(os.path.join(self.DATA_PATH, self.qid, 'Y_train.npy'))
            self.X_test = np.load(os.path.join(self.DATA_PATH, self.qid, 'X_test.npy'))
            self.Y_test = np.load(os.path.join(self.DATA_PATH, self.qid, 'Y_test.npy'))
        except Exception as e:
            logging.error('Data loading failed for {}'.format(self.qid))
            raise CustomException(e, sys)
    
    def getModel(self):
        try:
            self.model=Sequential()
            self.model.add(Input(shape=(2,)))
            self.model.add(Dense(8,activation='relu', name='HiddenLayer1',kernel_constraint=WeightConstraint()))
            self.model.add(Dense(3,activation='softmax', name='OuputLayer',kernel_constraint=WeightConstraint()))
        except Exception as e:
            logging.error('Model creation failed for {}'.format(self.qid))
            raise CustomException(e, sys)
    
    def train(self):
        try:
            logging.info('Training in progress......')
            # Configuration
            self.model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
            self.model_path=os.path.join(self.DATA_PATH, self.qid, 'model.keras')
            checkpoint = ModelCheckpoint(self.model_path, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max')
            callbacks_list = [checkpoint]

            # Training
            self.model.fit(self.X_train,self.Y_train,
            validation_data=(self.X_test,self.Y_test),
            epochs=100,batch_size=512,
            callbacks=callbacks_list,
            verbose=0)
            logging.info('Training completed for {}.'.format(self.qid))
            logging.info('Model saved at {}'.format(self.model_path))

        except Exception as e:
            logging.error('Training failed for {}'.format(self.qid))
            raise CustomException(e, sys)
    
    def evaluate(self):
        try:
            logging.info('Evaluating model...')
            self.model=load_model(self.model_path, custom_objects={'WeightConstraint': WeightConstraint}, compile=False)
            Y_pred = self.model.predict(self.X_test)
            Y_pred_classes = np.argmax(Y_pred, axis=1)
            Y_true = np.argmax(self.Y_test, axis=1)

            # Compute confusion matrix
            confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
            total_samples_per_class = np.sum(confusion_mtx, axis=1)

            # Calculate diagonal values (true positives) per class
            diagonal_values_per_class = np.diag(confusion_mtx)

            # Calculate accuracy for each class
            accuracies_per_class = diagonal_values_per_class / total_samples_per_class
            print('Classification accuracy for {} \n {}' .format(self.qid, accuracies_per_class))
            logging.info('Confusion Matrix:\n%s', confusion_mtx)
            logging.info('Model evaluation completed for {}.'.format(self.qid))
        
        except Exception as e:
            logging.error('Model evaluation failed for {}'.format(self.qid))
            raise CustomException(e, sys)
        