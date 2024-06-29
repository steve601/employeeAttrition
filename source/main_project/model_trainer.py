import os
import sys
from source.logger import logging
from source.exception import UserException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from source.commons import evaluate_model,save_object
from sklearn.neighbors import KNeighborsClassifier

@dataclass # allows us to only use variable within a class
class ModelTrainingConfig:
    trained_model_file_path:str = os.path.join('elements','model.pkl') #creating a path where we will save our model
    
class ModelTrainer:
    def __init__(self):
        self.modeltrainerconfig = ModelTrainingConfig() #creating an instance of the class above
        
    def start_of_model_training(self,train_arr,test_arr):
        try:
            X_train,y_train,X_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            models = {
                'Logistic Reg.':LogisticRegression(),
                'SVC':SVC(),
                'KNeughbors':KNeighborsClassifier(),
                'Cat Boost':CatBoostClassifier(),
                'XGB':XGBClassifier(),
                'Decision Tree':DecisionTreeClassifier(),
                'Random Forest':RandomForestClassifier()
            }
            model_report:dict = evaluate_model(X_train,y_train,X_test,y_test,models)
            best_score = max(sorted(list(model_report.values())))
            for k,v in model_report.items():
                if v == best_score:
                    best_model_name = k
                    break
                
            best_model = models[best_model_name] # accessing the values
            # let's set a condition such that the accuracy must be above a certain threshold
            if best_score < 0.70:
                raise UserException('No best model found')

            save_object(
                file_path = self.modeltrainerconfig.trained_model_file_path,
                obj = best_model
            )
            predicted = best_model.predict(X_test)
            score = accuracy_score(y_test,predicted)
            
            return score
        except Exception as e:
            raise UserException(e.sys)
            
        
    