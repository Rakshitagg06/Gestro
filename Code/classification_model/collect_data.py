# Will work on this later

import cv2
import mediapipe as mp
import csv
import time

FRAME_SKIP = 5
mp_hands = mp.solutions.hands


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


def record_gestures(gesture_label, output_file, sample_limit):
    """Record landmarks for a single gesture and append to CSV."""

    cap = cv2.VideoCapture(0)

    frame_counter = 0
    samples_saved = 0
    # set the constant , min_detection_confidence and min_tracking_confidence
    with mp_hands.Hands(max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        with open(output_file, "a", newline="") as f:
            writer = csv.writer(f)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                frame_counter += 1
                landmarks = extract_hand_landmarks(results)

                # Overlay info
                status_text = f"Recording: {gesture_label}" if landmarks else "No valid hand(s)"
                cv2.putText(frame, status_text, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0) if landmarks else (0, 0, 255), 2)
                cv2.putText(frame, f"Samples: {samples_saved}", (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("Gesture Recorder", frame)

                if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                    break

                # Skip saving frames if needed
                if landmarks and frame_counter % FRAME_SKIP == 0:
                    if sample_limit < 0 or samples_saved < sample_limit:
                        writer.writerow([gesture_label] + landmarks)
                        samples_saved += 1
                    elif sample_limit >= 0 and samples_saved >= sample_limit:
                        print(f"[INFO] Reached sample limit ({sample_limit}). Stopping.")
                        break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    gesture_label = input("Enter the gesture label to record: ").strip()
    sample_limit = int(input("Enter the number of samples to record (-1 for unlimited): ").strip())
    output_file = input("Enter the CSV output file path: ").strip()
    record_gestures(gesture_label, output_file, sample_limit)

