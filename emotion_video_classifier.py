from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image

#detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
#emotion_model_path = 'final_model.h5'
#face_detection = cv2.CascadeClassifier(detection_model_path)
#emotion_classifier = load_model(emotion_model_path, compile=False)
#EMOTIONS = ["happy", "sad"]


def emotion_testing():
    # Load the pre-trained model
    model = load_model('final_model.h5')
    # OpenCV initialization
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    while True:
        ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
        if not ret:
            continue
        gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
        faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
        for (x, y, w, h) in faces_detected:
            cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
            roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
            roi_gray = cv2.resize(roi_gray, (48, 48))
            img_pixels = image.img_to_array(roi_gray)
            img_pixels = np.expand_dims(img_pixels, axis=0)
            img_pixels /= 255
            predictions = model.predict(img_pixels)
            # Find the emotion with maximum score  
            max_index = np.argmax(predictions[0])
            emotions = ['happy','sad']
            predicted_emotion = emotions[max_index]
            # Display the emotion on the screen
            cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        resized_img = cv2.resize(test_img, (1000, 700))
        cv2.imshow('Facial emotion analysis ', resized_img)
        if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
            break
    cap.release()
    cv2.destroyAllWindows()
    return predicted_emotion  # Make sure to define and assign a default value to `predicted_emotion` before this line
