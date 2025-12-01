"""
Real-time Gesture Prediction using Trained Model

This script uses the trained gesture classification model to predict 
hand gestures in real-time from webcam input using MediaPipe landmarks.

Usage:
    python predict.py                          # Use default .h5 model
    python predict.py --model gesture_model.h5 # Specify .h5 model
    python predict.py --model your_model.tflite --tflite  # Use TFLite model
"""

import cv2
import numpy as np
import mediapipe as mp
import argparse
import os

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# --- Configuration ---
DEFAULT_MODEL_PATH = "gesture_model.h5"
DEFAULT_TFLITE_PATH = "your_model.tflite"
LABEL_MAP_PATH = "label_map.npy"
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence to display prediction


def load_keras_model(model_path):
    """Load a Keras .h5 model."""
    from tensorflow.keras.models import load_model
    model = load_model(model_path)
    print(f"✅ Loaded Keras model: {model_path}")
    return model


def load_tflite_model(model_path):
    """Load a TensorFlow Lite model."""
    try:
        import tensorflow.lite as tflite
    except ImportError:
        try:
            import tflite_runtime.interpreter as tflite
        except ImportError:
            raise ImportError("Please install 'tensorflow' or 'tflite_runtime'")
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    print(f"✅ Loaded TFLite model: {model_path}")
    return interpreter


def predict_keras(model, features):
    """Run prediction using Keras model."""
    features = features.reshape(1, -1).astype(np.float32)
    predictions = model.predict(features, verbose=0)
    return predictions[0]


def predict_tflite(interpreter, features):
    """Run prediction using TFLite model."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    features = features.reshape(1, -1).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions[0]


def extract_hand_landmarks(results):
    """
    Extract 3D hand landmarks for Left and Right hands.
    Returns a flat numpy array of 126 values (63 per hand).
    Left hand landmarks first, then Right hand landmarks.
    Returns zeros for missing hands.
    """
    hands_detected = results.multi_hand_landmarks or []
    
    if not hands_detected:
        return None
    
    # Get handedness labels with confidence check
    hand_labels = []
    for h in results.multi_handedness:
        label = h.classification[0].label
        score = h.classification[0].score
        if score < 0.7:
            return None
        hand_labels.append(label)
    
    # For 2 hands, ensure we have exactly one Left and one Right
    if len(hands_detected) == 2 and set(hand_labels) != {"Left", "Right"}:
        return None
    
    # Extract coordinates
    hands_dict = {}
    for lm, handedness in zip(hands_detected, hand_labels):
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y, p.z])
        hands_dict[handedness] = coords
    
    # Build feature vector: Left hand first, then Right hand
    landmarks = []
    landmarks.extend(hands_dict.get("Left", [0.0] * 63))
    landmarks.extend(hands_dict.get("Right", [0.0] * 63))
    
    return np.array(landmarks, dtype=np.float32)


def draw_hand_landmarks(frame, results):
    """Draw hand landmarks on the frame."""
    if results.multi_hand_landmarks:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, 
                                               results.multi_handedness):
            # Different colors for left/right hands
            label = handedness.classification[0].label
            color = (255, 100, 100) if label == "Right" else (100, 100, 255)
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=color, thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1)
            )


def run_prediction(args):
    """Main prediction loop."""
    # Load label map
    if not os.path.exists(LABEL_MAP_PATH):
        print(f"❌ Label map not found: {LABEL_MAP_PATH}")
        print("   Please train a model first using train_model.py")
        return
    
    label_map = np.load(LABEL_MAP_PATH, allow_pickle=True).item()
    inv_label_map = {v: k for k, v in label_map.items()}
    print(f"✅ Loaded {len(label_map)} gesture classes: {list(label_map.keys())}")
    
    # Load model
    if args.tflite:
        model_path = args.model if args.model else DEFAULT_TFLITE_PATH
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        model = load_tflite_model(model_path)
        predict_fn = lambda features: predict_tflite(model, features)
    else:
        model_path = args.model if args.model else DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"❌ Model not found: {model_path}")
            return
        model = load_keras_model(model_path)
        predict_fn = lambda features: predict_keras(model, features)
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        print("❌ Could not open webcam")
        return
    
    print("\n🎥 Starting Real-time Gesture Prediction...")
    print("   Press 'q' or ESC to quit\n")
    
    # Prediction state
    current_gesture = "None"
    current_confidence = 0.0
    frame_count = 0
    
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            
            # Extract landmarks and predict
            frame_count += 1
            if frame_count % 2 == 0:  # Predict every 2nd frame for performance
                landmarks = extract_hand_landmarks(results)
                
                if landmarks is not None:
                    predictions = predict_fn(landmarks)
                    predicted_idx = np.argmax(predictions)
                    confidence = predictions[predicted_idx]
                    
                    if confidence >= CONFIDENCE_THRESHOLD:
                        current_gesture = inv_label_map.get(predicted_idx, f"Class {predicted_idx}")
                        current_confidence = confidence
                    else:
                        current_gesture = "Uncertain"
                        current_confidence = confidence
                else:
                    current_gesture = "No hands detected"
                    current_confidence = 0.0
            
            # Draw hand landmarks
            draw_hand_landmarks(frame, results)
            
            # Draw prediction box
            box_color = (0, 200, 0) if current_confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(frame, (10, 10), (350, 90), (0, 0, 0), -1)
            cv2.rectangle(frame, (10, 10), (350, 90), box_color, 2)
            
            # Draw gesture name
            cv2.putText(frame, f"Gesture: {current_gesture}", (20, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw confidence bar
            bar_width = int(current_confidence * 300)
            cv2.rectangle(frame, (20, 60), (20 + bar_width, 75), box_color, -1)
            cv2.rectangle(frame, (20, 60), (320, 75), (100, 100, 100), 1)
            cv2.putText(frame, f"{current_confidence:.1%}", (325, 73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Show number of hands detected
            num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0
            cv2.putText(frame, f"Hands: {num_hands}", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
            
            # Display frame
            cv2.imshow("Gesture Prediction", frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Prediction stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Real-time gesture prediction using trained model"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default=None,
        help="Path to the model file (.h5 or .tflite)"
    )
    parser.add_argument(
        "--tflite", 
        action="store_true",
        help="Use TensorFlow Lite model instead of Keras"
    )
    
    args = parser.parse_args()
    run_prediction(args)


if __name__ == "__main__":
    main()
