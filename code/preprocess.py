import os
import tensorflow as tf
import numpy as np

# generate labels with [confidence level, x center, y center, width, height]
# for no object detected, the label will be all zeros, and treated in the loss

def load_data(image_dir, label_dir):
    image_paths = [os.path.join(image_dir, file) for file in os.listdir(image_dir)]
    label_paths = [os.path.join(label_dir, file) for file in os.listdir(label_dir)]

    images = []
    labels = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        # Load image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_jpeg(image, channels=3)  # Adjust channels as needed
        image = tf.cast(image, tf.float32) / 255.0  # Convert to float32 and normalize

        # Load label
        if os.path.getsize(lbl_path) > 0:
            label = tf.io.read_file(lbl_path)
            label = tf.strings.to_number(tf.strings.split(label, ' '), out_type=tf.float32)
        else:
            label = tf.zeros([5], dtype=tf.float32)

        images.append(image)
        labels.append(label)

    return images, labels

def get_dataset(image_dir, label_dir):
    images, labels = load_data(image_dir, label_dir)

    x = tf.stack(images)
    y = tf.stack(labels)

    # Shuffle the data
    # Create an index to shuffle both x and y in the same order
    indices = tf.range(start=0, limit=tf.shape(x)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    x = tf.gather(x, shuffled_indices)
    y = tf.gather(y, shuffled_indices)

    # Determine the number of data points
    num_examples = x.shape[0]
    train_end = int(num_examples * 0.8)
    val_end = train_end + int(num_examples * 0.1)

    # Split the data into training, validation, and test sets
    x_train = x[:train_end]
    y_train = y[:train_end]
    x_val = x[train_end:val_end]
    y_val = y[train_end:val_end]
    x_test = x[val_end:]
    y_test = y[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def main():
    x_train, y_train, x_val, y_val, x_test, y_test = get_dataset("data/images", 'data/labels')

    print("Shape of image tensor [num samples, width, height, channels]")
    print(x_train.shape)
    print("Shape of labels [num samples, label size]")
    print(y_train.shape)

if __name__ == '__main__':
    main()