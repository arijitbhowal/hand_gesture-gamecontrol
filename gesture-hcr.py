# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 21:05:39 2023

@author: ariji
"""
import numpy as np
import cv2
import pyautogui
import mediapipe as mp
import time

mp_drawing=mp.solutions.drawing_utils
mp_drawing_styles=mp.solutions.drawing_styles
mp_hands=mp.solutions.hands

cap=cv2.VideoCapture(0)

mp_hands=mp.solutions.hands
hands=mp_hands.Hands(static_image_mode=False,
                     max_num_hands=1,
                     min_detection_confidence=0.5,
                     min_tracking_confidence=0.5)


while True:
    ret,frame=cap.read()
    h,w,c =frame.shape
    start=time.perf_counter()
    if not ret:
        break
    
    #Flip the image for selfie view display
    # Convert BGR to RGB (Mediapipe takes input in RGB)
    image_rgb=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    #Mark image as not writable to improve performance
    image_rgb.flags.writeable=False
    
    #Process the image
    results=hands.process(image_rgb)
    
    image_rgb.flags.writeable=True
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame,hand_landmarks,mp_hands.HAND_CONNECTIONS)
            
            thumb_tip=hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip=hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip=hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip=hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip=hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            cv2.putText(frame, f"Thumb tip y: {round(thumb_tip.y, 2)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Index tip y: {round(index_tip.y, 2)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Middle tip y: {round(middle_tip.y, 2)}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Ring tip y: {round(ring_tip.y, 2)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Pinky tip y: {round(pinky_tip.y, 2)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            is_hand_closed=(
                index_tip.y>thumb_tip.y and
                middle_tip.y>thumb_tip.y and
                ring_tip.y>thumb_tip.y and
                pinky_tip.y>thumb_tip.y
                )
            
            if is_hand_closed:
                pyautogui.keyDown("left")
                pyautogui.keyUp("right")
            else:
                pyautogui.keyDown("right")
                pyautogui.keyUp("left")

    cv2.imshow('CV',frame)
    
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
    
    