import cv2
import numpy as np
import math
import streamlit as st
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

# Function for the sign detection feature
def sign_detection():
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Camera not accessible. Please check your camera.")
        return

    # Initialize the hand detector and classifier
    detector = HandDetector(maxHands=1)
    classifier = Classifier("model/keras_model.h5", "model/labels.txt")
    offset = 20
    imgSize = 300

    # Define labels for prediction
    labels = ["Hello", "I love you", "No", "Please", "Thank you", "Yes"]

    # Create a placeholder for the video feed
    frame_placeholder = st.empty()

    # Add a button to stop the stream
    if st.button('Stop'):
        st.write("Stream stopped.")
        cap.release()
        return

    while True:
        success, img = cap.read()
        if not success:
            st.error("Camera not accessible. Please check your camera.")
            break  # Stop if the camera is not working

        imgOutput = img.copy()
        hands, img = detector.findHands(img)

        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            # Ensure valid crop dimensions
            imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]
            imgCropShape = imgCrop.shape

            if imgCropShape[0] == 0 or imgCropShape[1] == 0:
                st.error("Invalid crop size. Hand might be out of bounds.")
                continue

            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
                prediction, index = classifier.getPrediction(imgWhite, draw=False)

            # Draw rectangles and labels on the output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        # Update the video frame in Streamlit
        frame_placeholder.image(cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB), channels="RGB")

    # Release the webcam and cleanup
    cap.release()
    cv2.destroyAllWindows()

# Main function to run the Streamlit app
def main():
    st.title("Sign Language Detection")

    # Navigation sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Home", "Sign Detection"])

    if page == "Home":
        st.image("img/5.jpg", caption="Welcome to Sign Language Detection", use_column_width=True)
        st.header("About the Project")
        
        # Custom CSS to make the text scrollable
        st.markdown("""
            <style>
            .scrollable-text {
                height: 300px;
                overflow-y: scroll;
                padding-right: 15px;
            }
            </style>
            """, unsafe_allow_html=True)

        # Using a scrollable div for the "About the Project" text
        st.markdown("""
            <div class="scrollable-text">
                This project is designed to facilitate communication through sign language detection using advanced computer vision techniques. The app leverages a webcam to capture hand gestures, which are then classified into specific sign language phrases, making it an intuitive tool for translating gestures into meaningful communication. By using machine learning algorithms and real-time hand tracking, it accurately identifies and interprets various hand signs to convey common phrases such as “Hello,” “Thank you,” or “Please.”
                <br><br>
                Our goal is to bridge the communication gap for hearing-impaired individuals and those unfamiliar with sign language. This interactive tool not only promotes inclusivity but also serves as an accessible platform for learning and practicing sign language. Users can gain confidence in expressing themselves through gestures while receiving instant feedback, making it an ideal solution for educational environments, healthcare settings, and social interactions where clear communication is essential.
                <br><br>
                In addition, this project highlights the power of AI-driven solutions in enhancing accessibility and empowering individuals to connect across language barriers. The modular design of the app also allows for future expansions, including support for additional signs, more complex gestures, and potentially even personalized sign language training modules, further advancing its role as a comprehensive communication aid.
            </div>
            """, unsafe_allow_html=True)

    elif page == "Sign Detection":
        st.header("You can try with this signs")
        st.image("img/3.png", caption="Copyright@2024 by NIRANJAN JHA", width=500)
        st.write("Click the button below to start the sign detection:")
        if st.button("Start Sign Detection"):
            sign_detection()

# Run the Streamlit app
if __name__ == "__main__":
    main()
