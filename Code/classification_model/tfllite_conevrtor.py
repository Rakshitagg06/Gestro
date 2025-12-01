import tensorflow as tf

# Load your Keras model (.h5)
model = tf.keras.models.load_model("gesture_model.h5")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save as .tflite
with open("your_model.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Conversion complete: your_model.tflite")
