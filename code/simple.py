# Put new model here
import tensorflow as tf

class Simple(tf.keras.Model):
    
    def __init__(self, **kwargs):

        super().__init__(**kwargs)
        # Feature extraction
        self.resnet = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(416, 416, 3))

        for layer in self.resnet.layers:
            layer.trainable = False

        self.resizeResNet = tf.keras.Sequential([
            tf.keras.layers.UpSampling2D(size=(4, 4)),
            tf.keras.layers.Conv2D(32, (1, 1), padding='same', strides=(1, 1))
        ])

        self.resizeResidual = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (1, 1), padding='same', strides=(4, 4)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding='same')
        ])

        self.mediumDetection = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), padding='same', strides=(16, 16)),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.BatchNormalization(),
        ])

        self.mediumPred = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        self.dense = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(32),
            tf.keras.layers.LeakyReLU(alpha=0.1),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, x):
        residual = x
        residual = self.resizeResidual(residual)

        # Feature extraction
        x = self.resnet(x)
        x = self.resizeResNet(x)

        x = self.adder([x, residual])

        x = self.mediumDetection(x)

        x = self.mediumPred(x)
        
        return tf.keras.activations.sigmoid(x)
    
