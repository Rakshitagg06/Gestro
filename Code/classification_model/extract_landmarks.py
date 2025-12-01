import os
import csv
import cv2
import random
import mediapipe as mp

mp_hands = mp.solutions.hands
random.seed(42)

def extract_hand_landmarks(results):
    """
    Extract 3D hand landmarks for Left and Right hands.
    Returns a flat list of 126 values (63 per hand) if:
      - 1 hand is detected, or
      - 2 hands are detected and they are exactly one Left and one Right
    Returns None if the frame doesn't meet the above criteria.
    """
    # Get detected hands and labels
    hands_detected = results.multi_hand_landmarks or []

    if not hands_detected:
        return None
    if len(hands_detected) not in [1, 2]:
        return None
    

    # Check handedness confidence and collect labels
    hand_labels = []
    for h in results.multi_handedness:
        label = h.classification[0].label
        score = h.classification[0].score
        if score < 0.7:  # skip low-confidence handedness, set the constant
            return None
        hand_labels.append(label)

    if len(hands_detected) == 2 and set(hand_labels) != {"Left", "Right"}:
        return None
    

    # Extract coordinates
    landmarks = []
    hands_dict = {}
    for lm, handedness in zip(hands_detected, hand_labels):
        coords = []
        for p in lm.landmark:
            coords.extend([p.x, p.y, p.z])
        hands_dict[handedness] = coords

    # Store Left hand first, then Right hand (consistent ordering)
    landmarks.extend(hands_dict.get("Left", [0] * 63))
    landmarks.extend(hands_dict.get("Right", [0] * 63))

    return landmarks

def process_directory(base_dir, train_csv, test_csv, split_ratio=0.7):
    """Recursively process gesture image folders and save train/test CSVs."""
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2) as hands, \
         open(train_csv, "w", newline="") as f_train, \
         open(test_csv, "w", newline="") as f_test:
        
        train_writer = csv.writer(f_train)
        test_writer = csv.writer(f_test)

        print(f"[INFO] Processing dataset recursively from '{base_dir}'")

        # Walk through all subdirectories recursively
        for root, dirs, files in os.walk(base_dir):

            if root == base_dir: continue

            # Determine label name (use parent folder name)
            gesture_dir = os.path.basename(root)
            label = gesture_dir.split("_")[0]
            samples = []

            for filename in files:

                filepath = os.path.join(root, filename)
                image = cv2.imread(filepath)
                if image is None:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
                landmarks = extract_hand_landmarks(results)

                if landmarks:
                    samples.append([label] + landmarks)

            if not samples:
                continue

            random.shuffle(samples)
            split_idx = int(len(samples) * split_ratio)
            train_samples = samples[:split_idx]
            test_samples = samples[split_idx:]

            train_writer.writerows(train_samples)
            test_writer.writerows(test_samples)

            print(f"  ✅ {root}: {len(train_samples)} train, {len(test_samples)} test samples")

    print(f"[INFO] Done — data saved to '{train_csv}' and '{test_csv}'.")


if __name__ == "__main__":
    base_dir = "gestures"
    train_csv = "gestures_train.csv"
    test_csv = "gestures_test.csv"
    process_directory(base_dir, train_csv, test_csv)