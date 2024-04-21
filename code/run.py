import tensorflow as tf
import numpy as np

from main import my_accuracy, my_loss

class Runner():
    def __init__(self):
        self.model = tf.keras.models.load_model("locator", 
            custom_objects={'my_accuracy': my_accuracy, 'my_loss': my_loss})

    def run(self, path):
        return self.model.predict(self.loadImage(path))
    
    def loadImage(self, img_path):
        images = []

        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0

        images.append(image)
        images = tf.stack(images)
        return images;
    
def main():
    runner = Runner()
    print("Running model on test image...")
    print(runner.run("test/IMG_0297000 copy 2.jpeg"))

if __name__ == '__main__':
    main()