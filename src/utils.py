import os
import sys
import numpy as np
import pandas as pd
from src.exception import Customexception
import dill
from sklearn.model_selection import GridSearchCV
import json

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)




def save_path(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path ,"wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise Customexception(e,sys)

def evaluate_models(x_train, y_train, x_test, y_test, models, param):

    try:

        report = {}

        for i in range(len(list(models))):

            model_name = list(models.keys())[i]
            model = list(models.values())[i]

            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3)

            gs.fit(x_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(x_train, y_train)

            # predictions
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            mae = mean_absolute_error(y_test, y_test_pred)

            rmse = mean_squared_error(
                y_test,
                y_test_pred
            ) ** 0.5

            # store metrics
            report[model_name] = {
                "train_r2_score": float(train_r2),
                "test_r2_score": float(test_r2),
                "mae": float(mae),
                "rmse": float(rmse),
                "best_params": gs.best_params_
            }

        # create artifacts folder
        os.makedirs("artifacts", exist_ok=True)

        # save json report
        with open("artifacts/model_report.json", "w") as f:
            json.dump(report, f, indent=4)

        return report

    except Exception as e:
        raise Customexception(e, sys)




