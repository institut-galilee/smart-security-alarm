####################################################
# Modified by Nazmi Asri                           #
# Original code: http://thecodacus.com/            #
# All right reserved to the respective owner       #
####################################################

# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np

import os
import serial


def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


ser = serial.Serial('/dev/ttyACM0', 9600)  # open serial port

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainner/")

# Load the trained mode
recognizer.read('trainner/trainner.yml')

# Load prebuilt model for Frontal Face
cascadePath = "/home/nabil/IOT/Arduino/projet/alarm_security/faceReconize/haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im = cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    # For each face in faces
    for (x, y, w, h) in faces:

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if (confidence < 50):

            # Create rectangle around the face
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 255, 0), 4)

            if (Id == 1):
                Id = "Nabil {0:.2f}%".format(round(100 - confidence, 2))

                cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
                cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 2)
                if (100 - confidence > 60):
                   ser.write(1)
                else:
                   ser.write(0)

            elif (Id == 2):
                Id = "Abder {0:.2f}%".format(round(100 - confidence, 2))

                cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 255, 0), -1)
                cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 2)
                if (100 - confidence > 50):
                   ser.write(1)
                else:
                   ser.write(0)
        else:

            # Create rectangle around the face
            cv2.rectangle(im, (x - 20, y - 20), (x + w + 20, y + h + 20), (0, 0, 255), 4)
            Id = "Inconnu"
            cv2.rectangle(im, (x - 22, y - 90), (x + w + 22, y - 22), (0, 0, 255), -1)
            cv2.putText(im, str(Id), (x, y - 40), font, 1, (255, 255, 255), 2)
            ser.write(0)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im', im)

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('q'):
        ser.close()
        break

# Stop the camera
cam.release()

# Close all windows
cv2.destroyAllWindows()
