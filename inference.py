import numpy as np
import pandas as pd
import sklearn as sk
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix
import tensorflow as tf
from joblib import dump, load
import os

model = tf.keras.models.load_model("best.h5")
sc = load("std_scaler.bin")
mms = load("mms_scaler.bin")

os.system("cls")
budget = float(input("Enter budegt per week in euros : "))
duration = float(input('Enter duration in weeks: '))
room_type = int(input("Enter the room type from the list below : \nSelect 0 for Blank\nSelect 1 for Ensuite\nSelect 2 for Studio\nSelect 3 for the Entire place\nSelect 4 for Non-Ensuite\nSelect 5 for Twin-Studio : "))

bdr = np.expand_dims(np.array([budget,duration,room_type]),axis=0)

bdr[:,:-1] = sc.transform(bdr[:,:-1].reshape(1,-1))
bdr[:,-1] = mms.transform(bdr[:,-1].reshape(1,-1))


result = model.predict(bdr)

print(result)
result = int(np.round(result))
if(result == 1):
    print("WON")
else:
    print("LOST")