import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump, load

#Get path to data
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(BASE_DIR, "data", "boston.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

#load data
def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}. Please ensure boston.csv is in the /data folder.")
    return pd.read_csv(path)


#build pipeline
def get_pipeline():
        return Pipeline([('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),])

def train_model(data_path):
        housing = load_data(data_path)
        #model = LinearRegression()
        #model = DecisionTreeRegressor()
        pipeline = get_pipeline()

        #split data
        split = StratifiedShuffleSplit(n_splits=1, test_size = 0.2, random_state = 42)
        for train_index, test_index in split.split(housing, housing['CHAS']):
            strat_train_set = housing.loc[train_index]
            strat_test_set = housing.loc[test_index]
        #Adding new TAXRM column to the dataset
        strat_train_set['TAXRM'] = strat_train_set['TAX'] / strat_train_set['RM']

    #Initialize features and labels
        housing_features = strat_train_set.drop("MEDV", axis=1)
        housing_labels = strat_train_set["MEDV"].copy()

        housing_prepared = pipeline.fit_transform(housing_features)
    #Selecting Random Forest Regressor Model
        model = RandomForestRegressor()
        model.fit(housing_prepared, housing_labels)

        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

    #Save and export model and pipeline
        dump(model, os.path.join(MODEL_DIR, 'RealEstate.joblib'))
        dump(pipeline, os.path.join(MODEL_DIR, 'pipeline.joblib'))  

        print(f"Model and Pipeline saved successfully in {MODEL_DIR}")

if __name__ == "__main__":
    train_model(DATA_PATH)
