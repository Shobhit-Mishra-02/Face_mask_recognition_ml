'''
This will extract the image of face from each frame.
'''
import cv2 as cv
import numpy as np
import pywt


class face_extractor:
    def __init__(self,img):
        self.img = img

    # This will extract the face from each frame or image.
    def face_recognition(self):
        Face = cv.CascadeClassifier('opencv_files/face.xml')
        Eyes = cv.CascadeClassifier('opencv_files/eyes.xml')
        face = Face.detectMultiScale(self.img, 1.1, 5)
        eyes = Eyes.detectMultiScale(self.img, 1.2, 3)

        for (x, y, w, h) in face:
            for (x_e, y_e, w_e, h_e) in eyes:
                if (x < x_e < x+h) and ((0.5*y) < y_e < y+h):
                    try:
                        
                        self.img = self.img[y:y+h,x:x+w]
                        
                        # self.img = cv.resize(self.img, (30,30))
                        # wave = w2d(img,'db1',5)
                        
                        # col = img_to_colordata(img)
                        
                        # final_data = np.concatenate((col,wave), axis=1)
                        # prediction = model.predict(final_data)
                        # print(prediction)  
                        return self.img ,x,y,w,h
                    except:
                        pass
    def get_wavelet(self):
        
        def w2d(img, mode='haar', level=1):
            imArray = img
            #Datatype conversions
            #convert to grayscale
            imArray = cv.cvtColor( imArray,cv.COLOR_RGB2GRAY )
            #convert to float
            imArray =  np.float32(imArray)   
            imArray /= 255;
            # compute coefficients 
            coeffs=pywt.wavedec2(imArray, mode, level=level)

            #Process Coefficients
            coeffs_H=list(coeffs)  
            coeffs_H[0] *= 0;  

            # reconstruction
            imArray_H=pywt.waverec2(coeffs_H, mode);
            imArray_H *= 255;
            imArray_H =  np.uint8(imArray_H)

            return imArray_H

        wave = w2d(self.img,'db1',5)
        # wave = cv.resize(wave,(30,30))
        return wave


class data_creator:
    def __init__(self, color_img, wave_img):
        self.color_img = np.array(color_img)
        self.wave_img = np.array(wave_img)

    def get_data(self):
        color_row = self.color_img.reshape(1,30*30*3)
        wave_row = self.wave_img.reshape(1,30*30)

        data_row = np.concatenate((color_row,wave_row), axis=1)
        return data_row