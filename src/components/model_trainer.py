import os
import sys 
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import Customexception
from src.logger import logging
from src.utils import save_path
from src.utils import evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("splitting training and test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor()
            }

            params={
                "Decision Tree":{
                    'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },

                "Random Forest":{
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Gradient Boosting":{
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.85,0.9],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "Linear Regression":{},

                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.01,0.05,0.1],
                    'iterations':[30,50,100]
                },

                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },

                "K-Neighbors Regressor":{
                    'n_neighbors':[5,7,9,11],
                    'weights':['uniform','distance'],
                    'algorithm':['auto','ball_tree','kd_tree','brute']
                }
            }

            model_report: dict= evaluate_models(x_train=x_train,y_train= y_train,x_test=x_test,y_test=y_test,
                                              models=models, param=params)
            
            ##to get best model score form dictionary
            best_model_score = max(
                model_report.values(),
                key=lambda x: x["test_r2_score"]
            )["test_r2_score"]

            ##to get best model name from dict

            best_model_name = max(
                model_report,
                key=lambda x: model_report[x]["test_r2_score"]
            )
            best_model= models[best_model_name] 

            if best_model_score<0.6:
                raise Customexception("No Best Model Found")
            logging.info(f"Best found model on both training and testing dataset")

            save_path(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            predicted = best_model.predict(x_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square

        except Exception as e:
            raise Customexception(e,sys)

