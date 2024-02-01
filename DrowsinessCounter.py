import cv2
import mediapipe
from math import sqrt



colors=[0,100,0]
eye_ratio_change = 25
flag = 0
counter_yawn = 0
counter_blink = 0
TOTAL_BLINKS = 0
TOTAL_YAWNS = 0
color = (255, 0, 255)
eye_ratioList = []
eye_ratioAvg = [0,0]

Lip_RatioList = []
FONT = cv2.FONT_HERSHEY_SIMPLEX

#Lip_Up = 11, Lip_ Down = 18 Lip_Left = 62 Lip_Right = 375
# landmarks from mesh_map.jpg
LIPS = [11,18,62,375]
LEFT_EYE = [ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]
EYES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
DRAWS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246,11 ,18,62,375 ]
#[362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398, 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246, 1,19,62,375 ]
mediapipe_face_mesh = mediapipe.solutions.face_mesh
face_mesh = mediapipe_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence =0.6, min_tracking_confidence=0.7)

video_capture = cv2.VideoCapture(0)

def landmarksDetection(image, results, draw=False):
    image_height, image_width= image.shape[:2]
    mesh_coordinates = [(int(point.x * image_width), int(point.y * image_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv2.circle(image, i, 2, (0, 255, 0), -1) for i in mesh_coordinates]
    return mesh_coordinates

# Euclidean distance to calculate the distance between the two points
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance


while True:
    frame_grab_check_bool, frame = video_capture.read()

    frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    frame_height, frame_width= frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results  = face_mesh.process(rgb_frame)
    if results.multi_face_landmarks:
        mesh_coordinates = landmarksDetection(frame, results, False)
        single_face_landmarks = results.multi_face_landmarks[0].landmark
        for id in DRAWS:
            if id in EYES:
                cv2.circle(frame, mesh_coordinates[id], 2,color, cv2.FILLED)
            else:
                cv2.circle(frame, mesh_coordinates[id], 5, color, cv2.FILLED)

        Lip_Up = mesh_coordinates[LIPS[0]]
        Lip_Down = mesh_coordinates[LIPS[1]]
        Lip_vertical_distance=euclideanDistance(Lip_Up,Lip_Down)
        cv2.line(frame, Lip_Up, Lip_Down, (0, 200, 0), 3)
        Lip_Left = mesh_coordinates[LIPS[2]]
        Lip_Right = mesh_coordinates[LIPS[3]]
        Lip_horizontal_distance=euclideanDistance(Lip_Right,Lip_Left)


        cv2.line(frame, Lip_Left, Lip_Right, (0, 200, 0), 3)
        right_eye_landmarkLeft = mesh_coordinates[RIGHT_EYE[0]]
        right_eye_landmarkRight = mesh_coordinates[RIGHT_EYE[8]]
        cv2.line(frame, right_eye_landmarkLeft, right_eye_landmarkRight, (0, 200, 0), 3)
        right_eye_landmarkUp1 = mesh_coordinates[RIGHT_EYE[12]]
        right_eye_landmarkUp2 = mesh_coordinates[RIGHT_EYE[11]]
        right_eye_landmarkDown2 = mesh_coordinates[RIGHT_EYE[5]]
        right_eye_landmarkDown1 = mesh_coordinates[RIGHT_EYE[4]]
        cv2.line(frame, right_eye_landmarkUp1, right_eye_landmarkDown1, (0, 200, 0), 3)
        cv2.line(frame, right_eye_landmarkUp2, right_eye_landmarkDown2, (0, 200, 0), 3)
        left_eye_landmarkLeft = mesh_coordinates[LEFT_EYE[0]]
        left_eye_landmarkRight = mesh_coordinates[LEFT_EYE[8]]
        cv2.line(frame, left_eye_landmarkLeft, left_eye_landmarkRight, (0, 200, 0), 3)
        left_eye_landmarkUp1 = mesh_coordinates[LEFT_EYE[12]]
        left_eye_landmarkUp2 = mesh_coordinates[LEFT_EYE[11]]
        left_eye_landmarkDown1 = mesh_coordinates[LEFT_EYE[4]]
        left_eye_landmarkDown2 = mesh_coordinates[LEFT_EYE[5]]
        cv2.line(frame, left_eye_landmarkUp1, left_eye_landmarkDown1, (0, 200, 0), 3)
        cv2.line(frame, left_eye_landmarkUp2, left_eye_landmarkDown2, (0, 200, 0), 3)

        right_eye_horizontal_distance = euclideanDistance(right_eye_landmarkLeft, right_eye_landmarkRight)
        right_eye_vertical_distance1 = euclideanDistance(right_eye_landmarkUp1, right_eye_landmarkDown1)
        right_eye_vertical_distance2 = euclideanDistance(right_eye_landmarkUp2, right_eye_landmarkDown2)

        left_eye_horizontal_distance = euclideanDistance(left_eye_landmarkLeft, left_eye_landmarkRight)
        left_eye_vertical_distance1 = euclideanDistance(left_eye_landmarkUp1, left_eye_landmarkDown1)
        left_eye_vertical_distance2 = euclideanDistance(left_eye_landmarkUp2, left_eye_landmarkDown2)

        right_eye_ratio = ((right_eye_vertical_distance1 + right_eye_vertical_distance2) / (2*right_eye_horizontal_distance))*100
        left_eye_ratio = ((left_eye_vertical_distance1 + left_eye_vertical_distance2) /(2*left_eye_horizontal_distance))*100
        eye_ratio= (left_eye_ratio+right_eye_ratio)/2 # Threshold = less than 20
        Lip_Ratio = (Lip_vertical_distance / (2 * Lip_horizontal_distance)) * 100  # Threshold = more than 45
        cv2.putText(frame, "Please blink your eyes",(int(frame_height/2), 100), FONT, 1, (0, 255, 0), 2)

        eye_ratioList.append(eye_ratio)
        if len(eye_ratioList) > 3:
            eye_ratioList.pop(0)
        eye_ratioAvg = sum(eye_ratioList) / len(eye_ratioList)
        #print(eye_ratioAvg)
        Lip_RatioList.append(Lip_Ratio)
        if len(Lip_RatioList) >3:
            Lip_RatioList.pop(0)
        Lip_RatioAvg = sum(Lip_RatioList)/len(Lip_RatioList)
        #print(Lip_RatioAvg)
        if Lip_RatioAvg < 40:
            if flag == 1:
                TOTAL_YAWNS += 1
                flag = 0
                color = (255, 0, 255)

        else:
            if eye_ratioAvg < 20 and counter_yawn == 0 and flag == 0:
                color = (0, 0, 255)
                flag = 1

        cv2.rectangle(frame, (20, 120), (290, 160), (0, 0, 0), -1)
        cv2.putText(frame, f'Total Yawns: {TOTAL_YAWNS}', (30, 150), FONT, 1, (0, 255, 0), 2)



    cv2.imshow('Drowsiness Detection', frame)

    if cv2.waitKey(2) == 27:
        break

cv2.destroyAllWindows()
video_capture.release()