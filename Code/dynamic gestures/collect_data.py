import os
import argparse
import time
import cv2
import numpy as np
import onnxruntime as ort
from abc import ABC

# --- Self-Contained ONNX Model Class ---
class HandDetection(ABC):
    def __init__(self, model_path, image_size=(320, 240)):
        # --- MODIFIED: Automatically select provider ---
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'CUDAExecutionProvider')
        elif 'DmlExecutionProvider' in ort.get_available_providers():
            providers.insert(0, 'DmlExecutionProvider')
        
        self.sess = ort.InferenceSession(model_path, providers=providers)
        print(f"HandDetection is using: {self.sess.get_providers()[0]}")
        # --- END MODIFICATION ---

        self.image_size, self.input_name = image_size, self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)

    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - self.mean) / self.std
        image = np.transpose(image, [2, 0, 1])
        return np.expand_dims(image, axis=0).astype(np.float32)

    def __call__(self, frame):
        input_tensor = self.preprocess(frame)
        boxes, _, _ = self.sess.run(self.output_names, {self.input_name: input_tensor})
        h, w, _ = frame.shape
        boxes[:, [0, 2]] *= w
        boxes[:, [1, 3]] *= h
        return boxes.astype(np.int32)

# --- Main Data Collection Logic ---
def collect_sequences(args):
    SEQUENCE_LENGTH = 30
    # Create nested directories: dynamic_dataset/{hand}/{gesture}
    gesture_path = os.path.join("dynamic_dataset", args.hand, args.gesture)
    images_path = os.path.join(gesture_path, "image_sequences")
    os.makedirs(gesture_path, exist_ok=True)
    os.makedirs(images_path, exist_ok=True)


    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    hand_detector = HandDetection("models/hand_detector.onnx")

    sequence_count = 0
    is_recording = False
    sequence_data = []
    sequence_frames = []


    print(f"Collecting for: HAND={args.hand}, GESTURE={args.gesture}. Press 's' to start, 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        display_frame = frame.copy()

        bboxes = hand_detector(frame)

        box_color = (0, 255, 0) # Green for idle

        if is_recording:
            box_color = (0, 0, 255) # Red for recording
            coords = [0, 0] # x, y for the target hand

            if bboxes is not None and len(bboxes) > 0:
                bboxes = sorted(bboxes, key=lambda box: box[0])

                target_hand = None
                if args.hand == 'left':
                    target_hand = bboxes[0] # Leftmost hand
                elif args.hand == 'right' and len(bboxes) > 1:
                    target_hand = bboxes[-1] # Rightmost hand
                elif args.hand == 'right' and len(bboxes) == 1:
                    target_hand = bboxes[0] # Use the only hand available

                if target_hand is not None:
                    center_x = (target_hand[0] + target_hand[2]) / 2
                    center_y = (target_hand[1] + target_hand[3]) / 2
                    coords = [center_x, center_y]
                    cv2.rectangle(display_frame, (target_hand[0], target_hand[1]), (target_hand[2], target_hand[3]), box_color, 2)

            sequence_data.append(coords)
            sequence_frames.append(frame)


            if len(sequence_data) >= SEQUENCE_LENGTH:
                sequence_timestamp = int(time.time() * 1000)
                sequence_name = f"sequence_{sequence_timestamp}"
                
                # Save coordinate data
                np.save(os.path.join(gesture_path, f"{sequence_name}.npy"), np.array(sequence_data))
                
                # Save image frames
                image_sequence_folder = os.path.join(images_path, sequence_name)
                os.makedirs(image_sequence_folder, exist_ok=True)
                for i, img_frame in enumerate(sequence_frames):
                    cv2.imwrite(os.path.join(image_sequence_folder, f"frame_{i+1}.jpg"), img_frame)

                sequence_count += 1
                is_recording = False
                sequence_data = []
                sequence_frames = []
                print(f"Saved sequence #{sequence_count}")


        elif bboxes is not None and len(bboxes) > 0:
            for box in bboxes:
                cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)

        cv2.putText(display_frame, f"{args.hand.upper()} HAND: {args.gesture} ({sequence_count})", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        if is_recording:
            cv2.putText(display_frame, f"RECORDING... {len(sequence_data)}/{SEQUENCE_LENGTH}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            cv2.putText(display_frame, "Press 's' to start", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Data Collection", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and not is_recording:
            # --- TIMER REMOVED ---
            is_recording = True
            sequence_data = []
            sequence_frames = []
            print("Started recording...")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect dynamic gesture sequences for a specific hand.")
    parser.add_argument("--hand", type=str, required=True, choices=['left', 'right'], help="The hand to collect data for ('left' or 'right').")
    parser.add_argument("--gesture", type=str, required=True, help="Name of the gesture to collect.")
    args = parser.parse_args()
    collect_sequences(args)