import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

image_size =128
batch_size = 32
num_classes=4

dataset_test=tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/dohac/Documents/dataset of project/Combined Dataset/test',
    shuffle=True,
    image_size=(image_size,image_size),
    batch_size=batch_size
    color_mode="grayscale"
)

dataset_train=tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/dohac/Documents/dataset of project/Combined Dataset/train',
    shuffle=True,
    image_size=(image_size,image_size),
    batch_size=batch_size
    color_mode="grayscale"
)

class_names_test=dataset_test.class_names
class_names_train=dataset_train.class_names
print("test columns is "+str(class_names_test))
print("train columns is "+str(class_names_train))

plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset_test.take(1):
 for i in range(12):
  ax=plt.subplot(3,4,i+1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  plt.title(class_names_test[label_batch[i]])
  plt.axis("off")

plt.figure(figsize=(10,10))
for image_batch, label_batch in dataset_train.take(1):
 for i in range(12):
  ax=plt.subplot(3,4,i+1)
  plt.imshow(image_batch[i].numpy().astype("uint8"))
  plt.title(class_names_train[label_batch[i]])
  plt.axis("off")

normalization_layer = layers.Rescaling(1.0 / 255)
dataset_train = dataset_train.map(lambda x, y: (normalization_layer(x), y))
dataset_test = dataset_test.map(lambda x, y: (normalization_layer(x), y))

AUTOTUNE = tf.data.AUTOTUNE
dataset_train = dataset_train.prefetch(buffer_size=AUTOTUNE)
dataset_test = dataset_test.prefetch(buffer_size=AUTOTUNE)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_size, image_size,1)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
    ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Print a summary of the model
model.summary()

history = model.fit(
    dataset_train,
    validation_data=dataset_test,
    epochs=50  # Change to a higher number for better results
)

test_loss, test_acc = model.evaluate(dataset_test)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Save the trained model to a file with the .keras extension
model.save('C:/Users/dohac/PycharmProjects/PythonProject1/model.keras')


# Load the saved model
loaded_model = tf.keras.models.load_model('C:/Users/dohac/PycharmProjects/PythonProject1/model.keras')

# Evaluate the loaded model
test_loss, test_acc = loaded_model.evaluate(dataset_test)
print(f"Test accuracy (loaded model): {test_acc * 100:.2f}%")

