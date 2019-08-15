#!/usr/bin/env python3
import os
import sys
import time
import numpy as np
import cv2 as cv
import threading

from utils import *

import showtc # for nrows, ncols, nbits


def simplify_contour(c):
	length = cv.arcLength(c, True)
	approx = cv.approxPolyDP(c, length * 0.01, True)

	# length changed noticeably?
	if length > 10 and abs(cv.arcLength(approx, True) / length - 1) > 0.1:
		return None

	return approx

def refine_corners(contour, image):
	#return contour
	return cv.cornerSubPix(image,
		corners=contour.astype(np.float32),
		winSize=(7,7),
		zeroZone=(2,2),
		criteria=(cv.TERM_CRITERIA_COUNT*0 | cv.TERM_CRITERIA_EPS, 10, 0.1))

def filter_and_refine_quads(monochrome, mask, minarea=70**2):
	(contours, hierarchy) = cv.findContours(mask, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

	# simplify runs of straight lines
	contours = [simplify_contour(c) for c in contours]

	# find quads
	contours = [c for c in contours if c is not None and len(c) == 4]

	# that are white (contour goes counterclockwise in screen coords)
	contours = [c for c in contours if contour_sense(c) < -np.pi]

	contours = [refine_corners(c, monochrome) for c in contours]

	# drop small ones right here
	contours = [c for c in contours if cv.contourArea(c) >= minarea]

	# sort by size
	contours.sort(key=lambda c: -cv.contourArea(c))
	return contours


#ft2 = cv.freetype.createFreeType2()
#ft2.loadFontData("C:\\Windows\\Fonts\\times.ttf", 0)
#ft2.loadFontData("C:\\Windows\\Fonts\\consola.ttf", 0)

def centeredText(im, text, origin, fontScale, color, thickness, background=None, *args, **kwargs):
	fontFace = cv.FONT_HERSHEY_SIMPLEX

	((w,h), baseLine) = cv.getTextSize(text, fontFace, fontScale, thickness)
	ox,oy = origin

	if background is not None:
		cv.rectangle(im,
			fixn((ox - w/2 - 10, oy - h/2 - 10, w+20, h+20), 4),
			color=background,
			thickness=cv.FILLED, lineType=cv.LINE_AA, shift=4)

	cv.putText(im,
		text,
		fixn((ox - w/2, oy + h/2), 0),
		fontFace, fontScale, color, thickness)

	#ft2.putText(im,
	#	text,
	#	(ox - w//2, oy + h//2),
	#	fontHeight=fontHeight,
	#	color=color,
	#	thickness=-1,
	#	line_type=cv.LINE_AA,
	#	bottomLeftOrigin=False)

	pass


drawscale = 32

# counterclockwise, order of findcontours for white blobs
# screen space coordinates
# content + 1px white border around
tch,tcw = (showtc.nrows+2, showtc.ncols+2)
tc_model = (np.array([
	[0,0], # top left
	[0,1], # bottom left
	[1,1],
	[1,0],
]) * (tcw,tch)).astype(np.float32)


cap = cv.VideoCapture(int(sys.argv[1]) if len(sys.argv) >= 2 else 0)
cap.set(cv.CAP_PROP_FPS, 30)
capw = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
caph = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

cap = FreshestFrame(cap)
cv.namedWindow("camera", cv.WINDOW_NORMAL) # resizable
cv.resizeWindow("camera", capw, caph)

maxquads = 3

alpha = 0.05
meanval = [0.0] * maxquads
meanerr = [0.0] * maxquads

while True:
	if not cap.running: break

	rv,im = cap.read()
	frameh,framew = im.shape[:2]
	frame_capture_time = time.time()
	if not rv: break

	# image analysis...
	monochrome = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

	th_effective, mask = cv.threshold(monochrome, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
	cv.imshow("thresholded", mask)

	quads = filter_and_refine_quads(monochrome, mask)[:maxquads]

	print(f"{len(quads)} quads")

	for index,quad in enumerate(quads):
		#quad = rotate_topleft(quad)
		#quad = np.float32(quad)

		# some measures of size
		quadarea = cv.contourArea(quad)
		quadsize = np.sqrt(quadarea)

		# sense: from screen space (where quad is) to quad space (where bits in the marker are)
		H = cv.getPerspectiveTransform(quad, tc_model)

		# sense: from quad space to screen space
		Hinv = np.linalg.inv(H)

		# if you wanna see more of what these homographies do, look below

		# sample cells inside the quad/marker
		# a sampling grid, sample center of each cell
		rx = np.arange(tcw) + 0.5
		ry = np.arange(tch) + 0.5
		sgrid_qs = np.dstack(np.meshgrid(rx, ry)) # sample grid in quad space

		sgrid_ss = cv.perspectiveTransform(sgrid_qs.reshape((-1, 1, 2)), Hinv).astype(np.float32) # sample grid in screen space
		sgrid_ss.shape = (tch, tcw, 2)

		# use cv.remap to sample at these coordinates
		# split into x coordinates and y coordinates, each 2d array
		xmap,ymap = np.moveaxis(sgrid_ss, 2, 0)
		sampled = cv.remap(mask, xmap, ymap, cv.INTER_AREA)
		# or simply take mask[xmap, ymap] with integer xmap/ymap

		# binarize
		sampled = (sampled >= 128)

		# try all four orientations
		for _ in range(4):
			if (sampled[-2,1:-1] == 0).all(): # top 8 bits should be zero (row -1 is white border, -2 has content)
				break
			else:
				sampled = np.rot90(sampled)
				quad = np.roll(quad, 1, axis=0)
		else:
			print("quad has no rotation where upper 8 bits are zero")
			continue

		try:
			# this can throw an assertion too, for the border
			codevalue = decode_marker(sampled)

			assert codevalue >> (showtc.nbits-8) == 0, "invalid marker contents; upper 8 bits expected to be zero"

		except AssertionError as e:
			#print(e)
			continue
			
		timecode = codevalue * 1e-3 # ms -> s

		# some time statistics

		delta = timecode - frame_capture_time
		err = abs(delta - meanval[index])
		meanval[index] += (delta - meanval[index]) * alpha
		meanerr[index] += (err - meanerr[index]) * alpha
		if err > 1.0:
			meanval[index] = meanerr[index] = 0

		print(f"#{index}: {timecode:.3f} s, delta {delta:+.3f} s")
		print(f"mean {meanval[index]:.3f}, mean err {meanerr[index]:.3f}")


		# mark the quad

		cv.polylines(im,
			pts=fixn(np.array([quad]), 4),
			isClosed=True,
			color=(255,0,255),
			thickness=1,
			lineType=cv.LINE_AA,
			shift=4)

		# top left corner
		cv.circle(im,
			center=fixn(quad[0,0], 4),
			radius=fixn(10, 4),
			color=(255, 255, 0),
			thickness=2,
			lineType=cv.LINE_AA,
			shift=4)

		# sample grid
		for coord in sgrid_ss.reshape((-1,2)):
			cv.circle(im,
				center=fixn(coord, 4),
				radius=fixn(2, 4),
				color=(0,0,255),
				thickness=1,
				lineType=cv.LINE_AA,
				shift=4)

		# text: time delay
		fontScale = 1.5
		fontScale = quadsize / 150
		thickness = int(np.ceil(fontScale * 2))
		centeredText(im,
			f"{meanval[index]:.2f}s",
			quad.mean(axis=(0,1)),
			fontScale=fontScale, color=(255,255,255), thickness=thickness, background=(0, 201, 106))

		# demonstration of homographies
		if 1:
			# (x,y) is a point
			# (x,y,1)*w for any w are homogeneous coords representing (x,y)
			# to get (x,y) from (xw,yw,w), divide by w
			# it can be understood as a projective space,
			# where any (x,y,z) is projected onto a z=1 screen to become (x/z,y/z,1)

			# p' = M p is how you apply a transformation matrix to a point

			# H transforms from screen space coords to quad space coords ("cells")
			# we'll add some translation and scaling to it
			# matrixes are 3x3
			# top left 2x2 is scaling
			# right column topmost 2 are translation
			# bottom row is the w-row, involved in perspective (think "divide by z")

			# scaling for display
			S = np.matrix(np.diag([drawscale, drawscale, 1]))
			
			# shift down and to the right by one
			T = np.matrix(np.eye(3))
			T[0,2] = +1
			T[1,2] = +1

			Hdisp = S * T * np.matrix(H)
			# matrix multiplication, read it right to left for the succession of transformations
			# enlarged output space coordinate, each cell is `drawscale` wide
			# ^ (S)
			# "quad" space, but now origin is a cell to the left and up of the corner
			# ^ (T)
			# quad space coordinate/pixel, origin is the top left white corner
			# ^ (H)
			# screen space coordinate

			# creates a canvas of (tcw+2,tch+2)*drawscale pixels
			# that's enough space for the quad including black quiet zone
			# fills each output pixel with what corresponds to the data in screen space (input)
			# implicitly inverts the matrix because it has to calculate backwards from output space coordinates
			outsize = ((tcw+2)*drawscale, (tch+2)*drawscale)
			output = cv.warpPerspective(mask, Hdisp, outsize, flags=cv.INTER_AREA)

			output = cv.cvtColor(output, cv.COLOR_GRAY2BGR)

			# draw sample grid
			for coord in ((sgrid_qs + (1,1)) * drawscale).reshape((-1,2)):
				cv.circle(output,
					center=fixn(coord, 4),
					radius=fixn(2, 4),
					color=(0,0,255),
					thickness=1,
					lineType=cv.LINE_AA,
					shift=4)

			cv.imshow(f"straight quad {index}", output)


	print()

	cv.imshow("camera", im)

	k = cv.waitKey(1)
	if k == 27:
		break

cap.release()
cv.destroyAllWindows()

