import cv2 as cv
# from functions import face
from joblib import load
import numpy as np
import pywt

print('loading model.....')
model = load('SVM_3.joblib')


def face_recognition(img):
    Face = cv.CascadeClassifier('opencv_files/face.xml')
    Eyes = cv.CascadeClassifier('opencv_files/eyes.xml')
    face = Face.detectMultiScale(img, 1.1, 5)
    eyes = Eyes.detectMultiScale(img, 1.2, 3)

    for (x, y, w, h) in face:
        for (x_e, y_e, w_e, h_e) in eyes:
            if (x < x_e < x+h) and ((0.5*y) < y_e < y+h):
                try:
                    img = img[y:y+h, x:x+w]
                    return img, x, y, w, h
                except:
                    pass


def w2d(img, mode='haar', level=1):
    imArray = img
    # Datatype conversions
    # convert to grayscale
    imArray = cv.cvtColor(imArray, cv.COLOR_RGB2GRAY)
    # convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    # compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Process Coefficients
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruction
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def get_data(color_img, wave_img):
    color_row = color_img.reshape(1, 30*30*3)
    wave_row = wave_img.reshape(1, 30*30)

    data_row = np.concatenate((color_row, wave_row), axis=1)
    return data_row


cam = 0
cap = cv.VideoCapture(cam)
cap = cv.VideoCapture(cam, cv.CAP_DSHOW)
# cap.set(cv. cv.CAP_PROP_FPS, 60)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:

    ret, frame = cap.read()

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    try:
        face,x,y,w,h = face_recognition(frame)
        wave_img = w2d(face,'db1',5)
        
        face = cv.resize(face,(30,30))
        wave_img = cv.resize(wave_img,(30,30))
        
        data_row = get_data(face,wave_img)

        font = cv.FONT_HERSHEY_SIMPLEX
        prediction = model.predict(data_row)
        if prediction[0] == 0:
            cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)
            cv.putText(frame, 'Save', (x, y), font,
                       2, (255, 0, 0), 2, cv.LINE_AA)
        elif prediction[0] == 1:
            cv.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 3)
            cv.putText(frame, 'Unsave', (x, y), font,
                       2, (0, 0, 255), 2, cv.LINE_AA)
    except:
        pass

    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
