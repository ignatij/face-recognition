import cv2;
import os;
import numpy as np;
from click._compat import raw_input

# get the face of an image using the coordinates of the face
def get_face(image, face_coord):
    face = [];
    x = face_coord[0];
    y = face_coord[1];
    w = face_coord[2];
    h = face_coord[3];
    w_rm = int(0.2 * w / 2);
    face.append(image[y: y + h, x: x + w - w_rm])    
    return face;

# normalize the images
def normalize_intensity(images):
    images_norm = [];
    for image in images:
        is_color = len(image.shape) == 3;
        if is_color:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
        images_norm.append(cv2.equalizeHist(image));
    return images_norm;
  

# resize the image to 50x50    
def resize(images, size=(50, 50)):
    images_resized = [];
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_AREA);
        else:
            image_norm = cv2.resize(image, size, interpolation=cv2.INTER_CUBIC);
        images_resized.append(image_norm);
    return images_resized;
    
# draw rectangle around the face coordinates    
def draw_rectangle(frame, face_coord):
    x = face_coord[0];
    y = face_coord[1];
    w = face_coord[2];
    h = face_coord[3];
    cv2.rectangle(frame, (x, y), (x + w, y + h), (150, 150, 0), 8);

# collect all the persons in the database
def collect():
    images = [];
    labels = [];
    labels_dic = [];
    people = [person for person in os.listdir("people/")];
    for i, person in enumerate(people):
        labels_dic.append(person);
        for image in os.listdir("people/" + person):
            images.append(cv2.imread("people/" + person + '/' + image, 0));
            labels.append(i);
    return (images, np.array(labels), labels_dic);

# enter name of the person and then capture 10 images and store them in the database
def take_images(webcam, detector):
    folder = "people/" + raw_input('Person: ').lower();
    cv2.namedWindow("Capturing...", cv2.WINDOW_AUTOSIZE);
    if not os.path.exists(folder):
        os.mkdir(folder);
        counter = 0;
        while counter < 10:
            frame, gray = webcam.get_frame();
            face_coord = detector.detect(gray);
            if len(face_coord):
                face_coord = face_coord[0];
                faces = get_face(frame, face_coord);
                faces = normalize_intensity(faces);
                faces = resize(faces);
                cv2.imwrite(folder + '/' + str(counter) + '.jpg', faces[0]);
                counter += 1;
                draw_rectangle(frame, face_coord);
            cv2.imshow("Capturing...", frame);
            cv2.waitKey(40);
        cv2.destroyAllWindows();
    else:
        print("This name already exists!");
        
# train the model and start the live recognition system        
def train_and_run(webcam, detector):
        
    images, labels, labels_dic = collect();
    rec = cv2.face.createLBPHFaceRecognizer();
    rec.train(images, labels);
    cv2.namedWindow("Live face recognition", cv2.WINDOW_AUTOSIZE);
     
    while True:
            frame, gray = webcam.get_frame();
            face_coords = detector.detect(gray);
            if len(face_coords):
                for face_coord in face_coords:
                    face = get_face(frame, face_coord);
                    face = normalize_intensity(face);
                    face = resize(face);
                    pred = [];
                    conf = [];
                    pred, conf = rec.predict(face[0]);
                    print("Prediction: " + labels_dic[pred].capitalize() + " Confidence: " + str(round(conf)));
                    cv2.putText(frame, labels_dic[pred].capitalize(), (face_coord[0], face_coord[1] - 10), 
                                cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243, 2));
                    draw_rectangle(frame, face_coord);
            cv2.imshow("Live face recognition", frame);
            if cv2.waitKey(40) & 0xFF == 27:
                break 
