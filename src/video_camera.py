import cv2;

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0);
    
    # free the camera after finishing        
    def __del__(self):
        self.video.release();
        
    # capture frame and convert to gray           
    def get_frame(self):
        _, frame = self.video.read();
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY);
        return frame, gray; 