import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

train_images = train_images/255
test_images = test_images/255

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation= "softmax")

    ])
model.compile(optimizer="adam", loss= "sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=2)


test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Tested acc: ", test_acc)

predictions = model.predict(test_images)
print(predictions[0])
