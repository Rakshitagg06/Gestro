import tensorflow as tf
import os
import sys
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def freeze_keras_model(keras_model_path, output_dir='frozen_model'):
    """
    Loads a .keras model, freezes it into a single .pb file (frozen graph),
    and saves it to the specified directory.

    Args:
        keras_model_path (str): Path to the input .keras model file.
        output_dir (str): Directory to save the frozen_graph.pb file.
    """
    if not os.path.exists(keras_model_path):
        print(f"Error: Model file not found at '{keras_model_path}'")
        sys.exit(1)

    print(f"Loading Keras model from: {keras_model_path}")
    model = tf.keras.models.load_model(keras_model_path)
    model.summary()

    try:
        # Convert the Keras model to a concrete function
        full_model = tf.function(lambda x: model(x))
        concrete_func = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

        # Freeze the model
        frozen_model_func = convert_variables_to_constants_v2(concrete_func)
        frozen_graph_def = frozen_model_func.graph.as_graph_def()

        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'frozen_graph.pb')

        # Save the frozen graph to a .pb file
        with tf.io.gfile.GFile(output_path, 'wb') as f:
            f.write(frozen_graph_def.SerializeToString())

        print("-" * 40)
        print(f"✅ Model successfully frozen.")
        print(f"Saved to: {output_path}")
        print("-" * 40)

    except Exception as e:
        print(f"\nAn error occurred during freezing: {e}")
        sys.exit(1)


if __name__ == '__main__':
    keras_file = input("Enter the path to your .keras model file: ").strip()
    freeze_keras_model(keras_file)
