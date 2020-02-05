#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
based on https://gist.github.com/tilfin/98bbba47fdc4ac10c4069cce5fabd834
and parameters: https://stackoverflow.com/questions/20801015/recommended-values-for-opencv-detectmultiscale-parameters
"""

import cv2
import sys
import os

class FaceCropper(object):
    # CASCADE_PATH = "/home/xin/Documents/celeba_preprocessing/haarcascade_frontalface_default.xml"

    def __init__(self, CASCADE_PATH):
        self.face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

    def generate(self, raw_img_path, crop_img_path, log_path, out_heigth=64, out_width=64):
        img = cv2.imread(raw_img_path)
        if (img is None):
            print("Can't open image file")
            return 0
        
        # get the filename of this image
        img_filename = raw_img_path.split("/")[-1]
        

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.05, 3, minSize=(30, 30))
        if (faces is None):
            print('Failed to detect face')
            
            f=open(log_path+"/FaceCropper_log.txt", "a+")
            f.write(img_filename+"\r\n")
            f.close()
            
            return 0
            

        #facecnt = len(faces)
        #print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            r = max(w, h) / 2
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (out_heigth, out_width))
            i += 1
            cv2.imwrite(crop_img_path+"/"+img_filename, lastimg)