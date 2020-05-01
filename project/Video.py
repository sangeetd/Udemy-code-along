
import cv2
import numpy as np
import matplotlib.pyplot as plt

print (cv2.__file__)

face_casacde = cv2.CascadeClassifier('E:\\pythonProjectsDir\\udemy-exploratory_data_analysis\\venv\\lib\\site-packages\\cv2\\data\\haarcascade_frontalface_alt2.xml')

videoCap = cv2.VideoCapture(0)

def open_video():
    while True:
        (isConnected, frame) = videoCap.read()
        if (isConnected):
            # frame_np = np.asarray(frame)
            # print(type(frame_np))
            # frame_np = frame_np.reshape(1, frame_np.shape[0], frame_np.shape[1], frame_np.shape[2])
            # print (frame_np.shape)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            #detecting faces from the frame
            faces = face_casacde.detectMultiScale(gray_frame, scaleFactor=1.5, minNeighbors=5)

            for (x, y, w, h) in faces:
                # print (x, y, w, h)
                roi_gray= gray_frame[y:y+h+10, x:x+w+10]
                roi_color = frame[y:y + h + 10, x:x + w + 10]
                # cv2.imwrite("frame-cap.png", roi_gray)

                #creating rectangle around the detected face
                rect_color = (255, 0, 0) #BGR
                rect_stroke = 2
                #params (image source, start_coordinate, end_coordinate, color, rect_width)
                cv2.rectangle(frame, (x, y), (x+w+10, y+h+10), rect_color, rect_stroke)

            cv2.imshow("frame ", frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
            # plt.imshow(frame_np)
            # plt.show()
            # videoCap.release()
            # return frame_np

open_video()
cv2.destroyAllWindows()
