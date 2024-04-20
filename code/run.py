import tensorflow as tf
import numpy as np

class Runner():
    def __init__(self):
        self.model = tf.keras.models.load_model("locator")

    def run(self, inputs):
        return self.model.predict(inputs)
    
def main():
    runner = Runner()

if __name__ == '__main__':
    main()