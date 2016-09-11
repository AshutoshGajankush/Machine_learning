from __future__ import print_function
import numpy as np
import argparse
import imutils
import cv2
from imutils.object_detection import non_max_suppression
from imutils import paths


h = cv2.HOGDescriptor()
h.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

ar = argparse.ArgumentParser()
ar.add_argument("-i", "--images", required=True, help="It specifies the path to images directory")
ar1 = vars(ar.parse_args())



i = list(paths.list_images(ar1["images"]))

for value in i:
	image = cv2.imread(value)
	image = imutils.resize(image, width=min(400, image.shape[1]))
	orig = image.copy()

	
	(outline, weights) = h.detectMultiScale(image, winStride=(4, 4),
		padding=(8, 8), scale=1.06)

	
	for (a, b, c, d) in outline:
		cv2.rectangle(orig, (a, b), (a + c, b + d), (0, 0, 255), 2)

	
	outline = np.array([[a, b, a + c, b + d] for (a, b, c, d) in outline])
	final = non_max_suppression(outline, probs=None, overlapThresh=0.65)
	

	
	for (aA, bA, aB, bB) in final:
		cv2.rectangle(image, (aA, bA), (aB, bB), (0, 255, 0), 2)

	
	filename = value[value.rfind("/") + 1:]
	print("There are total",len(final),"people in the image")	
	print(" {} original boxes, {} after suppression".format(
		 len(outline), len(final)))
	
	cv2.imshow("Before suppression", orig)
	cv2.imshow("After suppression", image)
	cv2.waitKey(0)

	
