import cv2
import numpy as np
import pickle
import os
import sys
# Load the cascade
face_cascade = cv2.CascadeClassifier('FR/trainer/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("FR/trainer/trainer.yml")

labels = {"person_name": 1}
with open("FR/model/labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

# To capture video from webcam. 
cap = cv2.VideoCapture(0)

while True:
    # Read the frame
    ret, frame = cap.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.03, 4, 60, (75,75))
    # Draw the rectangle around each face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y: y+h ,x: x+w ]   
        roi_color = frame[y: y+h,x: x+w]

        #recognizer
        id_, conf = recognizer.predict(roi_gray)
        if conf>=45 and conf<=65:
            #print(id_)
            #print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
        i = 0
        while os.path.exists("FR/img_saved/saved%s.jpg" % i):
            i+=1
        filename = 'FR/img_saved/saved%s.jpg' % i
        cv2.imwrite(filename, roi_gray)
    # Display
    cv2.imshow('frame', frame)
    # Stop if escape key is pressed
    if cv2.waitKey(60) & 0xFF == ord('q'):
        break
# Release the VideoCapture object
cap.release()
cv2.destroyAllWindows()