# plan: based of description, predict the score of wine
import os
os.environ["TF_USE_LEGACY_KERAS"]= "1"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub

df = pd.read_csv('personal_project/winemag-data-130k-v2.csv') # dataset

print(df.head())

df = df[["description", "points", "price"]] # choose only desired cols

df = df.dropna() # drop rows with na values

print(df.head())

for col in df.columns:
    na_count = df[col].isna().sum() # check if there are any na values left in dataset
print("na count: ", na_count)
#note to self: na values are the same as null values

plt.hist(df.points, bins = 20)
plt.title("Points histogram")
plt.ylabel("N")
plt.xlabel("Points")
plt.show()

#right now i will test a machine learning model that can only predict whether value is above or below 90 but later
# i want to try a model that actually predicts the score
# in the second scenarion the error will be a number e.g. points off on average rather than a percentage of how many i got correct
df["label"] = (df.points >= 90).astype(int) # returns boolean. true if geq 90, false otherwise
df = df[["description", "label"]]
print("tail", df.tail())
print("head", df.head())

train, val, test = np.split(df.sample(frac = 1), [int(0.8 * len(df)), int(0.9 * len(df))])

def df_to_dataset(dataframe, shuffle=True, batch_size=1024):
    df = dataframe.copy()
    labels = df.pop('label')
    df = df["description"]
    ds = tf.data.Dataset.from_tensor_slices((df, labels))
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
print("flag")
print(hub_layer(list(train_data)[0][0]))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)

model = tf.keras.Sequential()
model.add(hub_layer)
# add early stopping function
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(16, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.4))
model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = tf.keras.losses.BinaryCrossentropy(), metrics = ['accuracy'])
history = model.fit(train_data, epochs = 3, validation_data = valid_data, callbacks = [callback])



loss, accuracy = model.evaluate(test_data)

print("test loss: ", loss)
print("test accuracy: ", accuracy)