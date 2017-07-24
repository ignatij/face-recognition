import cv2;

class Detector(object):
    
    # setting the cascade classifier to detect faces
    def __init__(self):
        self.classifier = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml");
    
    # detect the coordinates of the face in image        
    def detect(self, image):
        scale_factor = 1.2;
        min_neighbors = 5;
        min_size = (30, 30);
        flags = cv2.CASCADE_FIND_BIGGEST_OBJECT | \
        cv2.CASCADE_SCALE_IMAGE;
        face_coord = self.classifier.detectMultiScale(
            image,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=flags
        );
        return face_coord;
