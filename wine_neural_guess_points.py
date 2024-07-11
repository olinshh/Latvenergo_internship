# plan: based of description, predict the score of wine
import os
os.environ["TF_USE_LEGACY_KERAS"]= "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model


df = pd.read_csv('personal_project/winemag-data-130k-v2.csv') # dataset

df = df[["description", "points", "price"]] # choose only desired cols

df = df.dropna() # drop rows with na values

df = df[["description", "points"]]

train, val, test = np.split(df.sample(frac = 1), [int(0.8 * len(df)), int(0.9 * len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    points = df.pop('points')
    df = df["description"]
    ds = tf.data.Dataset.from_tensor_slices((df, points))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

train_data = df_to_dataset(train)
valid_data = df_to_dataset(val)
test_data = df_to_dataset(test)

embedding = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
hub_layer = hub.KerasLayer(embedding, dtype=tf.string, trainable=True)

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = tf.keras.Sequential()
model.add(hub_layer)

model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    loss = tf.keras.losses.MeanSquaredError(),
    metrics = [tf.keras.metrics.MeanAbsoluteError()])

history = model.fit(train_data, epochs = 3, validation_data = valid_data, callbacks = [callback])
model.save('wine_score_predictor.h5')

#model = load_model('wine_score_predictor.h5') !!!!!

loss, mae = model.evaluate(test_data)

print("Test loss: ", loss)
print("test MAE: ", mae)

predictions = model.predict(test_data)

real_values = []
for description, points in test_data.unbatch():
    real_values.append(points.numpy())

real_values = np.array(real_values)

for i in range(len(predictions)):
    print(f"Predicted: {predictions[i][0]}, Real: {real_values[i]}")

