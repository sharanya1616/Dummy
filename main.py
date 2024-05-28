import face_recognition
import os
import sys
import cv2
import numpy as np
import math
import time
import serial

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)

class FaceRecognition:

    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = Trueimport face_recognition
import os
import sys
import cv2
import numpy as np
import math
import time
import serial

def face_confidence(face_distance, face_match_threshold=0.6):
    range_val = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range_val * 2.0)

    if face_distance > face_match_threshold:
        return round(linear_val * 100, 2)
    else:
        value = (linear_val + ((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return round(value, 2)

class FaceRecognition:

    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('FACES'):
            face_image = face_recognition.load_image_file(f'FACES/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)

        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        time.sleep(2)

        face_detected_time = None
        unknown_face_detected_time = None

        #Open serial port
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)  # Allow some time for Arduino to initialize

        while True:
            ret, frame = video_capture.read()

            if self.process_current_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]

                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:

                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'

                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        confidence = face_confidence(face_distances[best_match_index])
                        name = self.known_face_names[best_match_index]
                        if confidence > 80:
                            if face_detected_time is None:
                                face_detected_time = time.time()
                                ser.write(b'1')  # Send '1' to Arduino
                            elif time.time() - face_detected_time >= 5:  # Wait at least 5 seconds
                                print(1)
                                ser.write(b'0')  # Send '0' to reset Arduino
                                face_detected_time = None  # Reset the timer
                        else:
                            print(0)
                    else:  # Unknown face detected
                        if unknown_face_detected_time is None:
                            unknown_face_detected_time = time.time()
                        elif time.time() - unknown_face_detected_time >= 10:  # Wait at least 10 seconds
                            print(0)
                            unknown_face_detected_time = None  # Reset the timer

                    self.face_names.append(f'{name} ({confidence}%)')

            self.process_current_frame = not self.process_current_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

        # Close serial port
        ser.close()
        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
