# This code is a modified version of original code: https://docs.qualcomm.com/bundle/publicresource/topics/80-63442-2/building_and_executing_tutorial_linux_host.html?product=1601111740010412

import tensorflow as tf
import os
import sys

def inspect_frozen_graph(pb_file_path):
    """
    Loads a frozen graph .pb file and prints its input and
    possible output nodes, as shown in the SNPE documentation.

    Args:
        pb_file_path (str): The path to the frozen_graph.pb file.
    """
    if not os.path.exists(pb_file_path):
        print(f"Error: Frozen graph file not found at '{pb_file_path}'")
        sys.exit(1)

    print(f"Inspecting frozen graph: {pb_file_path}")

    try:
        with tf.io.gfile.GFile(pb_file_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name='')

            # Print all placeholder/input nodes:
            print("\n--- Input nodes (Placeholders) ---")
            for op in graph.get_operations():
                if op.type == "Placeholder":
                    print(f"Name: {op.name}, Shape: {op.outputs[0].shape}")

            # Print possible output nodes
            print("\n--- Possible output nodes ---")
            for op in graph.get_operations():
                if op.type in ["Softmax", "Reshape", "Identity"]:
                     print(f"Name: {op.name}, Type: {op.type}, Shape: {op.outputs[0].shape}")
        
        print("\n✅ Inspection complete.")

    except Exception as e:
        print(f"\nAn error occurred during inspection: {e}")
        sys.exit(1)


if __name__ == '__main__':
    pb_file = input("Enter the path to the frozen_graph.pb file: ").strip()
    inspect_frozen_graph(pb_file)
