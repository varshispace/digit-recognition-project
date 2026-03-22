import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from PIL import Image
from google.colab import files
import os

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

model = keras.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    x_train, y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

test_loss, test_acc = model.evaluate(x_test, y_test)
print("\n✅ Test Accuracy:", test_acc)

plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

plt.imshow(x_test[0].reshape(28,28), cmap='gray')
plt.title("Sample Test Image")
plt.show()

prediction = model.predict(x_test[0].reshape(1,28,28,1))
print("🔍 Predicted Digit:", np.argmax(prediction))

print("\n📂 Upload your custom digit image (digit.png)")
uploaded = files.upload()

for filename in uploaded.keys():
    img = Image.open(filename).convert('L')
    img = img.resize((28,28))
    img = np.array(img)
    img = 255 - img
    img = img / 255.0
    img = img.reshape(1,28,28,1)
    prediction = model.predict(img)
    print(f"🧠 Prediction for {filename}:", np.argmax(prediction))

model.save("digit_model.keras")
print("💾 Model saved successfully as .keras!")

print("\n📁 Files in directory:", os.listdir())
