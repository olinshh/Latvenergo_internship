import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

fashion_data = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

train_images = train_images / 255.0
test_images = test_images / 255.0 # normalizing the images so that their pixel values are in the range from 0 to 1

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape = (28, 28)), #we areconverting the 28x28 pixel images into a 1d array because the next layers of the neural network expect their input to be a 1d array.
tf.keras.layers.Dense(128, activation = 'softmax'), # 128 neurons
tf.keras.layers.Dense(10, activation = 'softmax') # 10 neurons becuase the data set has 10 different classes to choose from
])

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_images, train_labels, epochs = 5) # epochs specifies the number of times that the model will iterate over the entire dataset. each pass / iteration is called an epoch.
#increasing the number of epoch can increase the accuracy of the model however too many can result in overfitting.

test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Test loss: ", test_loss)
print("Test accuracyL: ", test_acc)

predictions = model.predict(test_images)

predicted_labels = np.argmax(predictions, axis = 1) # indices of predicted labels

fig, axs = plt.subplots(2, 5, figsize = (10, 5))

for i in range(10):
    row = i // 5
    col = i % 5
    
    axs[row, col].imshow(test_images[i], cmap = 'gray')
    axs[row, col].set_title(f"Predicted: {predicted_labels[i]}, Actual: {test_labels[i]}", fontsize = 8)
    axs[row, col].axis('off')
plt.show()
plt.savefig("personal_project/predicted_vs_actual.png")