import os
import tensorflow as tf
import numpy as np
import random
import math
from preprocess import get_dataset
from finder import Finder
from metrics import my_accuracy, my_loss

def getHyperparams():
    epochs = 2
    batch_size = 16

    training_rate = 0.01

    return [epochs, batch_size, training_rate]

def main():
    # Get the data

    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset("data/images", 'data/labels')

    params = getHyperparams()

    locator = Finder()

    locator.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=params[2]), 
        loss=my_loss, 
        metrics=[my_accuracy],
    )

    locator.build((1, 416, 416, 3))
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
        