import cv2 as cv
import numpy as np
import pickle
import os
import sys
from PIL import Image

class recognize():
    def __init__(self):
        self.cascade = cv.CascadeClassifier('FR/trainer/haarcascade_frontalface_alt2.xml')
        self.reg = cv.face.LBPHFaceRecognizer_create()
        self.reg.read("FR/trainer/trainer.yml")
        self.labels = {"person_name": 1}
        with open("FR/model/labels.pickle", 'rb') as f:
            og_labels = pickle.load(f)
            self.labels = {v:k for k,v in og_labels.items()}


    def train(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        image_dir = os.path.join(base_dir, "images")
        current_id = 0
        label_ids = {}
        y_labels = []
        x_train = []
        for root, dirs, files in os.walk(image_dir):
            for file in files:
                if file.endswith("png") or file.endswith("jpg") or file.endswith("jpeg"):
                    path = os.path.join(root,file)
                    label = os.path.basename(root).replace(" ", "-").lower()
                    print(label,file)
                    if label in label_ids:
                        pass
                    else:
                        label_ids[label] = current_id
                        current_id +=1
                    id_ = label_ids[label]
                    #y_labels.append(label) #some number
                    #x_train.append(path) #verify image, turn into NUMPY array, GRAY
                    pil_image = Image.open(path).convert("L" ) #grayscale
                    size = (550,550)
                    final_image = pil_image.resize(size, Image.ANTIALIAS)
                    image_array = np.array(final_image, "uint8")
                    faces = self.cascade.detectMultiScale(image_array, 1.03, 4, 60, (75,75))

                    for (x,y,w,h) in faces:
                        roi = image_array[y: y+h ,x: x+w]
                        x_train.append(roi)
                        y_labels.append(id_)

        with open("FR/model/labels.pickle", 'wb') as f:
            pickle.dump(label_ids, f)
        self.reg.train(x_train, np.array(y_labels))
        self.reg.save("FR/trainer/trainer.yml")
        print("Training completed")

    def live_testing(self):
        cam = cv.VideoCapture(0)
        while True:
            ret, frame = cam.read()
            # Convert to grayscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            # Detect the faces
            faces = self.cascade.detectMultiScale(gray, 1.03, 4, 60, (75,75))
            # Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi_gray = gray[y: y+h ,x: x+w ]   
                roi_color = frame[y: y+h,x: x+w]

                #recognizer
                id_, conf = self.reg.predict(roi_gray)
                if conf>=45 and conf<=65:
                    #print(id_)
                    #print(labels[id_])
                    font = cv.FONT_HERSHEY_SIMPLEX
                    name = self.labels[id_]
                    color = (255,255,255)
                    stroke = 2
                    cv.putText(frame, name, (x,y), font, 1, color, stroke, cv.LINE_AA)
                i = 0
                while os.path.exists("FR/img_saved/saved%s.jpg" % i):
                    i+=1
                filename = 'FR/img_saved/saved%s.jpg' % i
                cv.imwrite(filename, roi_gray)
            # Display
            cv.imshow('frame', frame)
            # Stop if escape key is pressed
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
        # Release the VideoCapture object
        cam.release()
        cv.destroyAllWindows()

def main():
    a = recognize()
    train = int(input("Do you want to re-train models?\n1. Yes\n2. No\nYour command: "))
    if train == 1:
        a.train()
    test = int(input("Do you want to live testing?\n1. Yes\n2. No\nYour commannd: "))
    if test == 1:
        a.live_testing()

if __name__ == "__main__":
    main()