from unittest import result
import mediapipe as mp 
import numpy as np 
import cv2 
 
cap = cv2.VideoCapture(0)

name = input("Enter the type of mood to train : ")

holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

X = []
data_size = 0

while True:
	about_read_mark = []
	_, frame_get = cap.read()
	frame_get = cv2.flip(frame_get, 1)
	result_process = holis.process(cv2.cvtColor(frame_get, cv2.COLOR_BGR2RGB))

	if result_process.face_landmarks:
		for i in result_process.face_landmarks.landmark:
			about_read_mark.append(i.x - result_process.face_landmarks.landmark[1].x)
			about_read_mark.append(i.y - result_process.face_landmarks.landmark[1].y)
            

		if result_process.left_hand_landmarks:
			for i in result_process.left_hand_landmarks.landmark:
				about_read_mark.append(i.x - result_process.left_hand_landmarks.landmark[8].x)
				about_read_mark.append(i.y - result_process.left_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				about_read_mark.append(0.0)

		if result_process.right_hand_landmarks:
			for i in result_process.right_hand_landmarks.landmark:
				about_read_mark.append(i.x - result_process.right_hand_landmarks.landmark[8].x)
				about_read_mark.append(i.y - result_process.right_hand_landmarks.landmark[8].y)
		else:
			for i in range(42):
				about_read_mark.append(0.0)

        
		X.append(about_read_mark)
		data_size = data_size+1



	drawing.draw_landmarks(frame_get, result_process.face_landmarks, holistic.FACEMESH_CONTOURS)
	drawing.draw_landmarks(frame_get, result_process.left_hand_landmarks, hands.HAND_CONNECTIONS)
	drawing.draw_landmarks(frame_get, result_process.right_hand_landmarks, hands.HAND_CONNECTIONS)

	cv2.putText(frame_get, str(data_size), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

	cv2.imshow("window", frame_get)

	if cv2.waitKey(1) == 27 or data_size>99:
		cv2.destroyAllWindows()
		cap.release()
		break


np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)