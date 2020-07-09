import cv2 as cv
import numpy as np
import time

class detection():
    def __init__(self):
        self.net = cv.dnn.readNet("Object_Detection/models/yolov3.weights", 'Object_Detection/models/yolov3.cfg')
        self.image = None
        self.classes = []

    def get_input(self):
        # path = str(input("What image you wanna test: "))
        # path = "Object_Detection/images/input/input.jpg"
        path = "Object_Detection/images/input/input.jpg"
        try:
            self.image = cv.imread("Object_Detection/images/input/input.jpg")
        except:
            # self.get_input()
            print("Wrong path")
        self.detecting()

    def detecting(self):
        Width = self.image.shape[1]
        Height = self.image.shape[0]
        scale = 0.00392
        self.classes = None
        with open("Object_Detection/models/classes.data", 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

        # Detecting objects
        blob = cv.dnn.blobFromImage(self.image, scale, (416,416), (0,0,0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())
        # Creating and label boxes
        self.class_ids = []
        self.confidences = []
        self.boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    self.class_ids.append(class_id)
                    self.confidences.append(float(confidence))
                    self.boxes.append([x, y, w, h])
        self.draw()

    def draw(self):
        conf_threshold = 0.5
        nms_threshold = 0.4
        indices = cv.dnn.NMSBoxes(self.boxes, self.confidences, conf_threshold, nms_threshold)

        for i in indices:
            i = i[0]
            box = self.boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(self.class_ids[i],self.confidences[i],round(x), round(y), round(x+w), round(y+h))
        
        self.show()

    def draw_prediction(self,class_id, confidence, x,y,xw,yh):
        label = str(self.classes[class_id])
        color = self.colors[class_id]
        cv.rectangle(self.image, (x,y), (xw,yh), color, 2)
        cv.putText(self.image, label, (x-10,y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2) 

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()    
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        return output_layers

    def show(self):
        cv.imwrite("Object_Detection/images/output/detection.jpg", self.image)
        cv.imshow("object detection", self.image)
        cv.waitKey()
        cv.destroyAllWindows()

    def live_testing(self):
        with open("Object_Detection/models/classes.data","r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        layer_names = self.net.getLayerNames()
        outputlayers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        self.colors= np.random.uniform(0,255,size=(len(self.classes),3))

        # Loading stream
        cam=cv.VideoCapture(0)
        font = cv.FONT_HERSHEY_PLAIN
        starting_time= time.time()
        frame_id = 0

        while True:
            _,frame= cam.read() 
            frame_id+=1
            
            height,width,channels = frame.shape
            # Detect objects
            blob = cv.dnn.blobFromImage(frame,0.00392,(320,320),(0,0,0),True,crop=False)

                
            self.net.setInput(blob)
            outs = self.net.forward(outputlayers)
            # Add and label boxes
            class_ids=[]
            confidences=[]
            boxes=[]
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3:
                        center_x= int(detection[0]*width)
                        center_y= int(detection[1]*height)
                        w = int(detection[2]*width)
                        h = int(detection[3]*height)

                        x=int(center_x - w/2)
                        y=int(center_y - h/2)

                        boxes.append([x,y,w,h])
                        confidences.append(float(confidence)) 
                        class_ids.append(class_id)
            indexes = cv.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
            for i in range(len(boxes)):
                if i in indexes:
                    x,y,w,h = boxes[i]
                    label = str(self.classes[class_ids[i]])
                    confidence= confidences[i]
                    color = self.colors[class_ids[i]]
                    cv.rectangle(frame,(x,y),(x+w,y+h),color,2)
                    cv.putText(frame,label+" "+(str(round(confidence,2)*100)+"%"),(x,y+30),font,1,(255,255,255),2)
                    

            # elapsed_time = time.time() - starting_time
            # fps=frame_id/elapsed_time
            # cv.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
            
            #Show result
            cv.imshow("Image",frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            
        cam.release()    
        cv.destroyAllWindows() 

def main():
    obj = detection()
    obj.get_input()

if __name__ == "__main__":
    main()
