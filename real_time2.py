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

times = []

for i in range(data.shape[0]):
    past_time = current_milli_time()
    x = np.expand_dims(data.iloc[i:i+1, :-1], 2)
    y_true = data.iloc[i, -1]
    y_pred = np.argmax(model.predict(x)[0])
    times.append(current_milli_time() - past_time)

print(np.array(times).mean())
print(np.array(times).std())