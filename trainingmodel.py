import numpy as np
import pandas as pd
from src import config
from pickle5 import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


MODELS = {"linearregression": LinearRegression()}
def train(data):
    df=pd.read_csv(data)

    x=df.iloc[:,:-1]
    y=df['Price']

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
    model=MODELS[config.MODEL]

    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)

    print('rmse', np.sqrt(mean_squared_error(y_test,y_pred)))
    print('r2 score is', r2_score(y_test,y_pred))


train(config.PROCESSED_DATA)

#saving model
pickle.dump(MODELS[config.MODEL],open(config.FINAL_MODEL,'wb'))




