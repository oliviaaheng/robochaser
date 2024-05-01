import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_dataset
from finder import Finder
from simple import Simple
from metrics import my_accuracy, my_loss, acc

def getHyperparams():
    epochs = 2
    batch_size = 4

    training_rate = 0.01

    return [epochs, batch_size, training_rate]

def getSimple():
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(False, "data/images", 'data/labels')

    params = getHyperparams()

    locator = Simple()

    locator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params[2]), 
        loss='mse', 
        metrics=[acc],
    )

    return locator, params, x_train, y_train, x_val, y_val, x_test, y_test

def getFull():
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset(True, "data/images", 'data/labels')

    params = getHyperparams()

    locator = Finder()

    locator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params[2]), 
        loss=my_loss, 
        metrics=[my_accuracy],
    )

    return locator, params, x_train, y_train, x_val, y_val, x_test, y_test

def main():
    # Get the model
    locator, params, x_train, y_train, x_val, y_val, x_test, y_test = getSimple()

    # Full model...
    # locator, params, x_train, y_train, x_val, y_val, x_test, y_test = getFull()

    x = tf.ones((1, 416, 416, 3))
    locator.call(x)
    locator.summary()

    trainStats = locator.fit(x_train, y_train,
                    batch_size=params[1],  # Specify your desired batch size
                    epochs=params[0],
                    validation_data=(x_val, y_val))
    
    print("Evaluate on Test Data")
    locator.evaluate(x_test, y_test)

    locator.save('locator.keras')

    return 0

if __name__ == '__main__':
    main()
        