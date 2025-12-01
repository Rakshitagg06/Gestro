import cv2
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort
from collections import deque
from abc import ABC
import argparse

# --- Self-Contained Hand Detection Class ---
class HandDetection(ABC):
    def __init__(self, model_path, image_size=(320, 240), use_gpu=True):
        # Set up providers for ONNX Runtime
        providers = []
        if use_gpu:
            try:
                # Check if CUDA is available for ONNX Runtime
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
            except:
                pass
        providers.append("CPUExecutionProvider")
        
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.image_size, self.input_name = image_size, self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)
        
        # Print which provider is being used
        print(f"Hand Detection using: {self.sess.get_providers()}")

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

# --- Self-Contained Hand Classification Class ---
class HandClassification(ABC):
    def __init__(self, model_path, image_size=(128, 128), use_gpu=True):
        # Set up providers for ONNX Runtime
        providers = []
        if use_gpu:
            try:
                # Check if CUDA is available for ONNX Runtime
                available_providers = ort.get_available_providers()
                if "CUDAExecutionProvider" in available_providers:
                    providers.append("CUDAExecutionProvider")
            except:
                pass
        providers.append("CPUExecutionProvider")
        
        self.sess = ort.InferenceSession(model_path, providers=providers)
        self.image_size, self.input_name = image_size, self.sess.get_inputs()[0].name
        self.mean = np.array([127, 127, 127], dtype=np.float32)
        self.std = np.array([128, 128, 128], dtype=np.float32)
        
        # Print which provider is being used
        print(f"Hand Classification using: {self.sess.get_providers()}")

    def preprocess(self, frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.image_size)
        image = (image - self.mean) / self.std
        image = np.transpose(image, [2, 0, 1])
        return np.expand_dims(image, axis=0).astype(np.float32)

    @staticmethod
    def get_square(box, image):
        height, width, _ = image.shape
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        if h < w:
            y0 = y0 - int((w - h) / 2)
            y1 = y0 + w
        if h > w:
            x0 = x0 - int((h - w) / 2)
            x1 = x0 + h
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(width - 1, x1)
        y1 = min(height - 1, y1)
        return x0, y0, x1, y1

    def get_crops(self, frame, bboxes):
        crops = []
        for bbox in bboxes:
            bbox = self.get_square(bbox, frame)
            crop = frame[bbox[1] : bbox[3], bbox[0] : bbox[2]]
            crops.append(crop)
        return crops

    def __call__(self, image, bboxes):
        crops = self.get_crops(image, bboxes)
        if not crops:
            return np.array([])
        crops_preprocessed = [self.preprocess(crop) for crop in crops]
        input_name = self.sess.get_inputs()[0].name
        outputs = self.sess.run(None, {input_name: np.concatenate(crops_preprocessed, axis=0)})[0]
        labels = np.argmax(outputs, axis=1)
        return labels

# --- Self-Contained LSTM Model Class ---
class GestureLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(GestureLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n.squeeze(0))

# --- Main Prediction Logic ---
def predict_dynamic(use_gpu=True):
    SEQUENCE_LENGTH = 30
    
    # Device configuration - use GPU if available and requested
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} - {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Load Left Hand Model and Classes
    with open("custom_models/dynamic_class_names_left_hand.txt", "r") as f:
        left_class_names = [line.strip() for line in f.readlines()]
    left_model = GestureLSTM(input_size=2, hidden_size=64, num_classes=len(left_class_names))
    left_model.load_state_dict(torch.load("custom_models/dynamic_gesture_model_left_hand.pth", map_location=device))
    left_model.to(device)
    left_model.eval()
    
    # Load Right Hand Model and Classes
    with open("custom_models/dynamic_class_names_right_hand.txt", "r") as f:
        right_class_names = [line.strip() for line in f.readlines()]
    right_model = GestureLSTM(input_size=2, hidden_size=64, num_classes=len(right_class_names))
    right_model.load_state_dict(torch.load("custom_models/dynamic_gesture_model_right_hand.pth", map_location=device))
    right_model.to(device)
    right_model.eval()

    # Initialize models with GPU support
    hand_detector = HandDetection("models/hand_detector.onnx", use_gpu=use_gpu)
    hand_classifier = HandClassification("models/crops_classifier.onnx", use_gpu=use_gpu)
    cap = cv2.VideoCapture(0)
    
    # Deques for each hand
    left_hand_sequence = deque(maxlen=SEQUENCE_LENGTH)
    right_hand_sequence = deque(maxlen=SEQUENCE_LENGTH)
    
    left_prediction, right_prediction = "...", "..."

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        bboxes = hand_detector(frame)
        
        left_coords, right_coords = [0, 0], [0, 0]
        left_fist_closed = False
        right_fist_closed = False

        if bboxes is not None and len(bboxes) > 0:
            labels = hand_classifier(frame, bboxes)
            
            # Combine boxes and labels, then sort by x-coordinate
            bboxes_with_labels = sorted(zip(bboxes, labels), key=lambda x: x[0][0])
            
            # Left Hand Logic
            if len(bboxes_with_labels) > 0:
                left_hand_box, left_label = bboxes_with_labels[0]
                if left_label == 25:  # 25 is the label for 'fist'
                    left_fist_closed = True
                left_coords = [(left_hand_box[0] + left_hand_box[2]) / 2, (left_hand_box[1] + left_hand_box[3]) / 2]
                cv2.rectangle(frame, (left_hand_box[0], left_hand_box[1]), (left_hand_box[2], left_hand_box[3]), (0, 255, 0), 2)
            
            # Right Hand Logic
            if len(bboxes_with_labels) > 1:
                right_hand_box, right_label = bboxes_with_labels[-1]
                if right_label == 25: # 25 is the label for 'fist'
                    right_fist_closed = True
                right_coords = [(right_hand_box[0] + right_hand_box[2]) / 2, (right_hand_box[1] + right_hand_box[3]) / 2]
                cv2.rectangle(frame, (right_hand_box[0], right_hand_box[1]), (right_hand_box[2], right_hand_box[3]), (255, 0, 0), 2)

        left_hand_sequence.append(left_coords)
        right_hand_sequence.append(right_coords)

        # Predict for Left Hand
        if len(left_hand_sequence) == SEQUENCE_LENGTH:
            if left_fist_closed:
                seq_tensor = torch.tensor([list(left_hand_sequence)], dtype=torch.float32).to(device)
                normalized_seq = seq_tensor - seq_tensor[:, 0:1, :]
                with torch.no_grad():
                    outputs = left_model(normalized_seq)
                    probs = torch.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                    if conf.item() > 0.9:
                        left_prediction = left_class_names[idx.item()]
                    else:
                        left_prediction = "..."
            else:
                left_prediction = "..."

        # Predict for Right Hand
        if len(right_hand_sequence) == SEQUENCE_LENGTH:
            if right_fist_closed:
                seq_tensor = torch.tensor([list(right_hand_sequence)], dtype=torch.float32).to(device)
                normalized_seq = seq_tensor - seq_tensor[:, 0:1, :]
                with torch.no_grad():
                    outputs = right_model(normalized_seq)
                    probs = torch.softmax(outputs, dim=1)
                    conf, idx = torch.max(probs, 1)
                    if conf.item() > 0.9:
                        right_prediction = right_class_names[idx.item()]
                    else:
                        right_prediction = "..."
            else:
                right_prediction = "..."

        cv2.putText(frame, f"Left: {left_prediction}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Right: {right_prediction}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Concurrent Gesture Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dynamic Gesture Recognition')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage and run on CPU only')
    args = parser.parse_args()
    
    use_gpu = not args.no_gpu
    if use_gpu:
        print("GPU usage enabled (use --no-gpu to disable)")
    else:
        print("GPU usage disabled")
    
    predict_dynamic(use_gpu=use_gpu)