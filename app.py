from scipy.spatial import distance
from flask import Flask, request, jsonify, render_template
import cv2
import traceback
from sip import *
import pickle
import warnings
from ml import *
import joblib
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
# from ml import *
global label
global certainty
global training_landmarks
training_landmarks, _ = read_landmarks_from_csv("landmarks.csv")
model = pickle.load(
    open(
        r"C:\Users\Priyanshi Furiya\Downloads\ML_SIP FINAL\ML_SIP FINAL\xgboost_model.pkl",
        "rb",
    )
)
label_encoder = joblib.load("label_encoder.pkl")
app = Flask(__name__, template_folder="templates", static_folder="static")


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/try_now")
def try_now():
    return render_template("trynow.html")


@app.route("/accuracy")
def try1():
    return render_template("accuracy.html")


def calculate_similarity(predicted_landmarks, training_landmarks):
    # Calculate the average Euclidean distance between corresponding landmarks
    avg_distance = np.mean(
        [
            distance.euclidean(pred_lm, train_lm)
            for pred_lm, train_lm in zip(predicted_landmarks, training_landmarks)
        ]
    )
    # Normalize the distance to a scale of 0 to 1
    MAX_DISTANCE = 74.7514876201743
    similarity_score = 1 - (
        avg_distance / MAX_DISTANCE
    )  # MAX_DISTANCE is the maximum possible distance between landmarks
    # Convert similarity score to percentage (0 to 100)
    similarity_percentage = similarity_score * 100
    return similarity_percentage


@app.route("/process_frame", methods=["POST"])
def predict():
    global label
    all_landmarks = []
    try:
        try:
            selected_pose = request.form["pose"]
            mapping_dict = {
                "0": "anantasana",
                "1": "ardha uttanasana",
                "2": "bhujangasana",
                "3": "chakravakasana",
                "4": "dandasana",
                "5": "makarasana",
                "6": "sukhasana",
                "7": "urdhva mukha svanasana",
                "8": "utkatasana",
                "9": "uttana shishosana",
                "10": "utthita trikonasana",
                "11": "vajrasana",
                "12": "virbhadrasana",
                "13": "vriksasana",
                "14": "vrischikasana",
                "15": "malasana",
            }
            pose = mapping_dict[selected_pose]
            frame_file = request.files["frame"]
            frame_data = bytearray(frame_file.read())
            frame_np = np.asarray(frame_data, dtype=np.uint8)
            frame_np = np.array(frame_np)
            frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            image = equalize(image)
            image, landmarkss = landmarks(image)
            all_landmarks.append(landmarkss)

            # Perform pose prediction based on the selected pose name
            label = model.predict(landmarkss)
            label1 = label_encoder.inverse_transform([label[0]])[0]
            print("Predicted Label:", label1)

            # Calculate similarity percentage for the selected pose
            similarity_percentage = calculate_similarity(
                landmarkss, training_landmarks)

            # If the predicted label matches the selected pose, return the certainty percentage, otherwise set it to 0
            if label1.lower() == pose.lower():
                return jsonify({"label": label1, "certainty": similarity_percentage})
            else:
                return jsonify({"label": "Not correct Pose", "certainty": 0})
        except:
            return jsonify({"label": "Please select a pose"})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/process_frames", methods=["POST"])
def predict1():
    global label
    global certainty
    all_landmarks = []
    try:
        frame_file = request.files["frame"]
        frame_data = bytearray(frame_file.read())
        frame_np = np.asarray(frame_data, dtype=np.uint8)
        frame_np = np.array(frame_np)
        frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        image = equalize(image)
        image, landmarkss = landmarks(image)
        all_landmarks.append(landmarkss)
        # print(all_landmarks)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        label = model.predict(landmarkss)
        label1 = label_encoder.inverse_transform([label[0]])[0]
        print(label1)
        certainity = model.predict_proba(landmarkss)
        certainity = np.argmax(certainity)
        certainity = int(certainity)
        print(certainity)
        return jsonify({"label": label1, "certainty": certainity})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/update_labels_certainty", methods=["POST"])
def update_labels_certainty():
    global label, certainty
    return jsonify({"label": label, "certainty": round(certainty * 100, 2)})


if __name__ == "__main__":
    app.run(debug=True)
