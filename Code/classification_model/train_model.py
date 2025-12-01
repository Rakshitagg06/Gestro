import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_FILE = "gestures_train.csv"

def train_model():
    df = pd.read_csv(DATA_FILE)
    labels = df.iloc[:, 0].values
    features = df.iloc[:, 1:].values

    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    # Split using raw labels to keep alignment for CSVs
    X_train, X_test, lbl_train, lbl_test = train_test_split(
        features, labels, test_size=0.3, random_state=42, stratify=labels
    )

    # One-hot encode AFTER split for training
    y_train = to_categorical([label_map[l] for l in lbl_train], num_classes=len(unique_labels))
    y_test = to_categorical([label_map[l] for l in lbl_test], num_classes=len(unique_labels))

    # Write CSVs: label first column, then features with original column names
    feature_cols = df.columns[1:]
    label_col = df.columns[0]

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df.insert(0, label_col, lbl_train)
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df.insert(0, label_col, lbl_test)

    train_df.to_csv('training_dataset.csv', index=False)
    test_df.to_csv('testing_dataset.csv', index=False)
    model = Sequential([
        Dense(128, activation='relu', input_shape=(features.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20)

    model.save("gesture_model.h5")
    np.save("label_map.npy", label_map)
    print("✅ Model trained and saved as gesture_model.h5")

if __name__ == "__main__":
    train_model()
