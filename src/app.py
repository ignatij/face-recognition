import cv2;
from video_camera import VideoCamera
from face_detector import Detector
import operations as op

webcam = VideoCamera();
detector = Detector();

#uncomment the following line to add new person in the database
#op.take_images(webcam, detector);

#start live recognition
op.train_and_run(webcam, detector)
 

