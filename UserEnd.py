# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 20:47:26 2024

@author: atrij
"""

import numpy as np
import cv2
import time 
import PoseModule as pm

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

# Storing the angle data of 6 yoga poses   
angle_data = [
    {'Name': 'tadasan', 'right_arm': 201, 'left_arm': 162, 'right_leg': 177, 'left_leg': 182},
    {'Name': 'vrksana', 'right_arm': 207, 'left_arm': 158, 'right_leg': 180, 'left_leg': 329},
    {'Name': 'balasana', 'right_arm': 155, 'left_arm': 167, 'right_leg': 337, 'left_leg': 335},
    {'Name': 'trikonasana', 'right_arm': 181, 'left_arm': 184, 'right_leg': 176, 'left_leg': 182},
    {'Name': 'virabhadrasana', 'right_arm': 167, 'left_arm': 166, 'right_leg': 273, 'left_leg': 178},
    {'Name': 'adhomukha', 'right_arm': 176, 'left_arm': 171, 'right_leg': 177, 'left_leg': 179}
]

while True:
    success, img = cap.read()
    img = cv2.resize(img, (1280, 728))
    
    img = detector.findPose(img, False)
    lmlist = detector.getPosition(img, False)

    if len(lmlist) != 0:
        right_arm = detector.findAngle(img, 12, 14, 16)
        left_arm = detector.findAngle(img, 11, 13, 15)
        right_leg = detector.findAngle(img, 24, 26, 28)
        left_leg = detector.findAngle(img, 23, 25, 27)

        # Iterate through angle_data to find matching yoga pose
        for pose in angle_data:
            if (
                abs(right_arm - pose['right_arm']) < 10 and
                abs(left_arm - pose['left_arm']) < 10 and
                abs(right_leg - pose['right_leg']) < 10 and
                abs(left_leg - pose['left_leg']) < 10
            ):
                cv2.putText(img, pose['Name'], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                break  # Exit loop once the matching yoga pose is found

    cv2.imshow("Image", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
