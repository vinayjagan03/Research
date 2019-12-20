# Import Keras Model
from keras.models import model_from_json
json_file = open('model.json', 'r')
model = model_from_json(json_file.read())
json_file.close()
model.load_weights('model.h5')

# Organize imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Create a millisecond time lambda to measure performance
import time
current_milli_time = lambda: int(round(time.time() * 1000))

# Import data
data = pd.read_csv('mitbih_test.csv')

# Split actual output into the specific classifications
Y = data.iloc[:, -1]
Y_0 = np.ndarray.flatten(np.argwhere(Y == 0))
Y_1 = np.ndarray.flatten(np.argwhere(Y == 1))
Y_2 = np.ndarray.flatten(np.argwhere(Y == 2))
Y_3 = np.ndarray.flatten(np.argwhere(Y == 3))
Y_4 = np.ndarray.flatten(np.argwhere(Y == 4))

# Choose 100 samples from each classification
Y_0_test = np.random.choice(Y_0, 100)
Y_1_test = np.random.choice(Y_1, 100)
Y_2_test = np.random.choice(Y_2, 100)
Y_3_test = np.random.choice(Y_3, 100)
Y_4_test = np.random.choice(Y_4, 100)

# Concatenate the different samples and shuffle the array
arr = np.concatenate((Y_0_test, Y_1_test, Y_2_test, Y_3_test, Y_4_test))
np.random.shuffle(arr)

times = []

# Real-time predict by iterating through arr and predicting
for i in range(len(arr)):
    past_time = current_milli_time()
    x = np.expand_dims(data.iloc[i:i+1, :-1], 2)
    y_true = data.iloc[i, -1]
    y_pred = np.argmax(model.predict(x)[0])
    times.append(current_milli_time() - past_time)

print(np.array(times).mean())
print(np.array(times).std())