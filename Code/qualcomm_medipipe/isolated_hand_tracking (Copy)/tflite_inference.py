import numpy as np
import torch
try:
    import tensorflow.lite as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        raise ImportError("Please install 'tensorflow' or 'tflite_runtime' to run these models.")

class TFLiteWrapper:
    def __init__(self, model_path):
        # Initialize Interpreter
        self.interpreter = tflite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()

        # Get details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # --- NEW: Print the output names so we know the order ---
        print(f"\nModel: {model_path}")
        for i, detail in enumerate(self.output_details):
            print(f"  Output {i}: {detail['name']} (Shape: {detail['shape']})")
        # --------------------------------------------------------
        
        # Store input shape
        self.input_shape = tuple(self.input_details[0]['shape'])

    def __call__(self, *args):
        # args[0] is the input image tensor (PyTorch)
        input_tensor = args[0]

        # 1. Convert PyTorch Tensor to Numpy
        input_data = input_tensor.detach().cpu().numpy()

        # 2. Check Layout (NCHW vs NHWC)
        # PyTorch uses NCHW (Batch, Channels, Height, Width)
        # TFLite often uses NHWC. If shapes don't match, we transpose.
        expected_shape = self.input_shape
        if input_data.shape != expected_shape:
            # Try transposing from NCHW (0,1,2,3) to NHWC (0,2,3,1)
            input_data = np.transpose(input_data, (0, 2, 3, 1))
        
        # Ensure types match (often float32)
        if self.input_details[0]['dtype'] == np.float32:
            input_data = input_data.astype(np.float32)

        # 3. Set Input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 4. Run Inference
        self.interpreter.invoke()

        # 5. Gather Outputs
        outputs = []
        # We return outputs in the order the model produces them.
        # Note: We rely on the order in output_details. 
        # Sometimes TFLite orders outputs differently than PyTorch. 
        # If predictions are scrambled, we might need to sort output_details by name.
        for i in range(len(self.output_details)):
            out_data = self.interpreter.get_tensor(self.output_details[i]['index'])
            outputs.append(torch.from_numpy(out_data))

        return tuple(outputs)

    def get_input_spec(self):
        # Helper to satisfy app constructor requirements
        # Returns (H, W) tuple
        if self.input_shape[1] == 3: # NCHW
            return (self.input_shape[2], self.input_shape[3])
        else: # NHWC
            return (self.input_shape[1], self.input_shape[2])