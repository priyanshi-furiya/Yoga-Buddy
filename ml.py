from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import re
import csv
import numpy as np
from sklearn.metrics import accuracy_score
import pickle


def read_landmarks_from_csv(file_path):
    landmarks_data = []
    labels = []

    with open(file_path, "r") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            landmarks_str = row[0]
            label = row[1] if len(row) > 1 else ""

            landmarks = []
            for match in re.finditer(
                r"x: (\d+\.\d+)\s+y: (\d+\.\d+)\s+z: (-?\d+\.\d+)", landmarks_str
            ):
                x = float(match.group(1))
                y = float(match.group(2))
                z = float(match.group(3))
                landmarks.extend([x, y, z])

            landmarks_data.append(landmarks)
            labels.append(label)

    max_length = max(len(landmarks) for landmarks in landmarks_data)
    landmarks_data = [
        landmarks + [0] * (max_length - len(landmarks)) for landmarks in landmarks_data
    ]

    return np.array(landmarks_data), labels
