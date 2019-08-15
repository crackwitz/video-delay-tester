import os
import sys
import time
import numpy as np
import cv2 as cv

from utils import *

shape = (nrows, ncols) = (8,8)
nbits = nrows * ncols

drawscale = 32 # size of a pixel
delay = 10

if __name__ == '__main__':
	cv.namedWindow("Timecode", cv.WINDOW_NORMAL)

	while True:
		now = time.time()
		timestamp = int(now * 1e3)

		pattern = build_marker(shape, timestamp)
		resized = upscale(drawscale, pattern * np.uint8(255))

		cv.imshow("Timecode", resized)

		key = cv.waitKey(delay)
		if key == 27:
			break

	cv.destroyWindow("Timecode")

