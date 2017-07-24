# Simple face recognition system

![demo](demo.gif)

***

Simple face recognition system based on OpenCV 3.2.0 and Python 3.5.2.

 - face detection using Cascade Classifier
 - recognition with LBPH (Local Binary Patterns Histogram)

To add new person:
 - open app.py and uncomment the following line
```python
op.take_images(webcam, detector);
```

To start:
 - execute app.py (make sure the following line is uncommented):
```python
op.train_and_run(webcam, detector)
```

***
