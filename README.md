# Yoga Buddy

Yoga Buddy is an innovative application that utilizes machine learning and image processing techniques to classify real-time yoga poses. Whether you're a seasoned yogi or just starting your yoga journey, Yoga Buddy is here to assist you in perfecting your poses and improving your practice.

## Features

- Real-time yoga pose classification using an XGBoost machine learning model.
- User-friendly GUI powered by Flask, HTML, CSS, and JavaScript.
- **Try Now:** Practice different yoga poses and receive instant feedback on your form.
- **Accuracy Checker:** Test the accuracy of your yoga poses by selecting a specific pose and receiving feedback on your form.

## Instructions for Usage

1. **Clone the Repository:**
   - Clone the repository to your local machine:

     bash
     git clone https://github.com/priyanshi-furiya/yoga-buddy.git
     

2. **Install Dependencies:**
   - Navigate to the project directory:

     bash
     cd yoga-buddy
     

   - Install the required dependencies:

     bash
     pip install -r requirements.txt
     

3. **Run the Application:**
   - Execute the following command to start the application:

     bash
     python app.py
     

4. **Access Yoga Buddy:**
   - Open your web browser and go to `http://localhost:5000`.

## Pages

### 1. Try Now

- **Description:** Practice different yoga poses and receive instant feedback on your form.
- **Instructions:**
  1. Ensure your webcam is connected and functioning properly.
  2. Visit the Yoga Buddy website.
  3. Click on the "Try Now" section.
  4. Follow the on-screen instructions to perform various yoga poses.
  5. Receive instant feedback on your form and pose accuracy.

### 2. Accuracy Checker

- **Description:** Test the accuracy of your yoga poses by selecting a specific pose and receiving feedback on your form.
- **Instructions:**
  1. Visit the Yoga Buddy website.
  2. Navigate to the "Accuracy Checker" section.
  3. Select a specific yoga pose you want to test.
  4. Follow the on-screen instructions to perform the selected pose.
  5. Receive feedback on the accuracy of your pose, including suggestions for improvement.

## Dataset and Model

- **Dataset:** Yoga Buddy utilizes a yoga pose dataset for training the XGBoost model.
- **Model:** The application runs on an XGBoost machine learning model for real-time yoga pose classification.
