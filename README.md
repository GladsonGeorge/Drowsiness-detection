# Drowsiness-detection-mediapipe-part
--README--
## STEPS TO RUN CODE##

The yawn counter code is written in the python file DrowsinessCounter.py

This code runs on uses the modules of opencv, mesiapipe and math packages to run.

%%INSTALLING PACKAGES%%

Open CMD on the python workspace directory and use pipinstall:

py -m pip install opencv-python
py -m pip install mediapipe
py -m pip install python-math

%%PLEASE NOTE THAT PYTHON 3.10 WAS USED FOR RUNNING THE CODE%%

##YAWN COUNTER##

In the packages, opencv was used to get the live feed as frames, and
the module of FaceMesh from mediapipe has been used to recognise faces with 468 landmarks, out of which
16 were mapped for each eye, and 4 were mapped for each significant point on the mouth:
LIPS = [11,18,62,375]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]

The calculations done on these were to find the aspect ratios of eyes and lips, which gives the amount of
opening/closing of each, revealing yawns when mouth is open to a large degree, and eyes are closing
simultaneously by checking for breach of threshold values.
The formulae and threshold have been given:

right_eye_ratio = ((right_eye_vertical_distance1 + right_eye_vertical_distance2) / (2*right_eye_horizontal_distance))*100
left_eye_ratio = ((left_eye_vertical_distance1 + left_eye_vertical_distance2) /(2*left_eye_horizontal_distance))*100
eye_ratio= (left_eye_ratio+right_eye_ratio)/2 # Threshold = less than 20
Lip_Ratio = (Lip_vertical_distance / (2 * Lip_horizontal_distance)) * 100  # Threshold = more than 45

For the eyes, two vertical distances are calculated and the average is taken.
The distances taken are of the euclidean nature.
The outputs are the count of yawns printed onto the live feed.
