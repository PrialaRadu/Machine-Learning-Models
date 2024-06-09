import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout

# 1. load data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. normalize data
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. reshape data
x_train.reshape(-1, 28, 28, 1)
x_test.reshape(-1, 28, 28, 1)

# 4. one-hot encode labels
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


# 5.1 plot examples of images

def plot_images(x, y):
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap='grey')
        plt.title(f"Label: {y[i].argmax()}")
        plt.axis('off')
    plt.show()


plot_images(x_train, y_train)


# 5.2 plot the distribution of labels
def plot_distribution(y):
    unique, counts = np.unique(y.argmax(axis=1), return_counts=True)
    plt.figure(figsize=(10, 10))
    plt.bar(unique, counts, color='blue')

    plt.xlabel('Digit')
    plt.ylabel('Frequency')
    plt.title('Class Distribution')
    plt.show()


plot_distribution(y_train)

# split the data into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)

# 6. model building
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax'),
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. train and evaluate the model
history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val))

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Accuracy: {test_acc}")


# 8. plot training history
def plot_training_history(history):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.show()


plot_training_history(history)
