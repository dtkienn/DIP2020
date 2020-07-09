<h1>Usage:</h2>

1. Install requirements:
```python
pip install -r requirements.txt
```
2. Download YOLOv3 pre-trained weights file:
```
wget https://pjreddie.com/media/files/yolov3.weights
```
3. Object detection with images:
```python
python image.py
```
4. Live object detection with webcam and videos:
```python
python live.py
```
This particular model is trained on [COCO](https://cocodataset.org/#home) dataset (common objects in context) from Microsoft. It is capable of detecting 80 common objects. See the full list  [here](https://github.com/Sm00thiee/DIP2020/blob/master/Object_Recognition/classes.data)

For more information: [YOLO](https://pjreddie.com/darknet/yolo/)

<h2>Future plans:</h2>

- Re-train weights file with [OID](https://storage.googleapis.com/openimages/web/index.html) (Google Image Dataset)
- Build UX/UI
- Graduate from USTH