# Custom loss and accuracy
import tensorflow as tf

def my_loss(true, pred):
    # Isolate the first element and the remaining elements
    first_element_true = true[:, 0]
    first_element_pred = pred[:, 0]
    remaining_true = true[:, 1:]
    remaining_pred = pred[:, 1:]

    # Compute MSE for the first element
    mse_first = tf.reduce_mean(tf.square(first_element_true - first_element_pred))

    return tf.reduce_mean(mse_first)

def my_accuracy(true, pred):
    # Isolate the first element and the remaining elements
    confidence_label = true[:, 0]
    confidence_pred = pred[:, 0]
    box_label = true[:, 1:]
    box_pred = pred[:, 1:]

    confidence = confidence_pred
    box_acc = 1 - tf.nn.sigmoid(tf.reduce_mean(tf.square(box_label - box_pred), axis=1))
    
    # Accuracy if there is a car in frame
    yes_acc = confidence

    # Accuracy if the car is not in frame
    no_acc = 1 - confidence
    
    custom_acc = confidence_label * (yes_acc) + (1 - confidence_label) * (no_acc)

    # Average this custom loss over the batch
    return tf.reduce_mean(custom_acc)

def acc(true, pred):
    return tf.reduce_mean((true * (pred)) + ((1 - true) * (1 - pred)))