import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import tensorflow

passengers = pd.read_csv('train.csv', delimiter=',', header=0)
passengers = passengers.replace({True: 1, False: 0})
dataset = passengers.values

le = OneHotEncoder()

columns_to_encode = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination']

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

ct = ColumnTransformer(
    transformers=[('encoder', ohe, columns_to_encode)],
    remainder='passthrough'
)

passengers_encoded = ct.fit_transform(passengers)

passengers_encoded_df = pd.DataFrame(passengers_encoded)

X = passengers_encoded_df[:,1:-1]
Y = passengers_encoded_df[:,-1]

X = X.astype(str)


