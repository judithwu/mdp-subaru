# USAGE 
# python dlib_predict_video.py --input video/2_0.avi --models  models/ --upsample 1 --output demo/output.mp4

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import numpy as np
import argparse
import time
import dlib
import cv2
import os
from scipy.spatial import distance

awake_path = "data/awake"
drowsy_path = "data/drowsy"

# load the face detector (HOG-SVM)
print("[INFO] loading dlib thermal face detector...")
detector = dlib.simple_object_detector(os.path.join("thermal-facial-landmarks-detection", "models", "dlib_face_detector.svm"))

# load the facial landmarks predictor
print("[INFO] loading facial landmark predictor...")
predictor = dlib.shape_predictor(os.path.join("thermal-facial-landmarks-detection", "models", "dlib_landmark_predictor.dat"))

# initialize the video stream
vs = cv2.VideoCapture(0)

while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()

	# break the loop if the frame 
	# was not grabbed
	if not grabbed:
		break
	
	# resize the frame
	frame = imutils.resize(frame, width=500)

	# copy the frame
	frame_copy = frame.copy()
	frame_show = frame.copy()

	# convert the frame to grayscale
	frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)

	# detect faces in the frame
	rects = detector(frame, upsample_num_times=0)

	# loop over the bounding boxes
	for rect in rects:
		# convert the dlib rectangle into an OpenCV bounding box and
		# draw a bounding box surrounding the face
		(x, y, w, h) = face_utils.rect_to_bb(rect)
		cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.rectangle(frame_copy, (x-60, y-60), (x + w+60, y + h+60), (0, 0, 255), 2)

		# predict the location of facial landmark coordinates then 
		# convert the prediction to an easily parsable NumPy array
		shape = predictor(frame, rect)
		shape = face_utils.shape_to_np(shape)
		#print(shape[17])
		#print(len(shape))

		# loop over the (x, y)-coordinates from our dlib shape
		# predictor model draw them on the image
		#coords = []
		#exclude = [17, 18, 19, 20, 21, 36, 37, 38, 39, 40, 41]
		#for i, (sx, sy) in enumerate(shape):
			#if i not in exclude:
				#cv2.circle(frame_copy, (sx, sy), 2, (255, 0, 0), -1)
			#cv2.circle(frame_copy, (sx, sy), 2, (255, 0, 0), -1)
			#coords.append((sx, sy))
		#point18_x, point18_y = int(shape[17][0]-10), int(shape[17][1]+10) #left eyebrow top corner
		#w = int(distance.euclidean(shape[21], shape[17])+15)
		#h = int(distance.euclidean(shape[41], shape[19])+7)
		#cv2.rectangle(frame_show, (point18_x, point18_y), (point18_x + w, point18_y + h), (252, 227, 3), 1)
		
		#roi = frame_copy[point18_y:point18_y+h, point18_x:point18_x+w ] #[y1:y2, x1:x2]
		roi = frame[y-20:y+h+20, x-20:x+w+20]
		cv2.imwrite("roi.png", roi)

	# show the image
	cv2.imshow("Frame", frame_copy)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit cleanup 
cv2.destroyAllWindows()
vs.release()
