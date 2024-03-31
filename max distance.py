from ml import read_landmarks_from_csv
from scipy.spatial import distance

# Read landmarks data from the CSV file using the provided function
training_landmarks, _ = read_landmarks_from_csv("landmarks.csv")


def calculate_max_distance(training_landmarks):
    global max_distance
    max_distance = 0
    for i in range(len(training_landmarks)):
        for j in range(i + 1, len(training_landmarks)):
            # Calculate the Euclidean distance between the two landmarks
            dist = distance.euclidean(
                training_landmarks[i], training_landmarks[j])
            # Update max_distance if the calculated distance is greater
            if dist > max_distance:
                max_distance = dist
    return max_distance


# Call the function to calculate the maximum distance
MAX_DISTANCE = calculate_max_distance(training_landmarks)
print(MAX_DISTANCE)
