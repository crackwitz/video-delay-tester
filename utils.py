import numpy as np
import cv2 as cv
import threading


def RoundUpPowerOf2(num):
	# https://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
	num -= 1

	k = 1
	while True:
		mask = num >> k
		if not mask: break
		num |= mask
		k *= 2

	num += 1
	return num

def linear2gray(num):
	return num ^ (num >> 1)

def gray2linear(num, nbits=64):
	nbits = RoundUpPowerOf2(nbits)

	while nbits >= 2:
		nbits //= 2
		num ^= num >> nbits

	return num

def num2bits(num, nbits=64):
	assert num >= 0
	assert num.bit_length() <= nbits

	result = [(num >> k) & 1 for k in range(nbits)]

	# little endian, index = position
	return result

def bits2num(bits):
	return sum(bool(d) << p for p,d in enumerate(bits))

def upscale(factor, im):
	h,w = im.shape[:2]
	return cv.resize(im, (w*factor, h*factor), interpolation=cv.INTER_NEAREST)



def build_marker(shape, value):
	(nrows, ncols) = shape
	nbits = nrows * ncols

	code = linear2gray(value)

	code_bits = num2bits(code, nbits=nbits)
	code_bits = np.array(code_bits, dtype=np.bool).reshape(shape)

	im = np.zeros((4+nrows, 4+ncols), dtype=np.bool)
	# quiet zone: black
	im[1:-1, 1:-1] = 1 # white border
	im[2:-2, 2:-2] = code_bits # content

	return im

def decode_marker(samples):
	# check border is white
	assert samples[0,:].all() and samples[-1,:].all() and samples[:,0].all() and samples[:,-1].all()

	(nrows, ncols) = samples.shape
	nrows -= 2 # subtract border, check comes later
	ncols -= 2
	nbits = nrows * ncols

	bits = samples[1:-1, 1:-1].flatten()

	value = bits2num(bits)
	value = gray2linear(value, nbits=nbits)

	return value

def fixn(value, shift):
	# drawing utility, opencv drawing functions only take integers,
	# but they have a shift parameter for "fixed point" format
	factor = 1<<shift
	if isinstance(value, (int, float)):
		return int(round(value * factor))
	elif isinstance(value, (tuple, list)):
		return tuple(int(round(v * factor)) for v in value)
	elif isinstance(value, np.ndarray):
		result = np.round(value * factor).astype(np.int)
		if len(result.shape) == 1:
			result = tuple(result)
		# otherwise: see polylines
		return result
	else:
		assert False, f"fixn() doesn't know how to deal with type {type(value)}: {value!r}"

def contour_sense(contour):
	"signed sum of all angles. should be +- 2pi for single turns. use to determine if clockwise or ccw."
	# sum angles. positive -> clockwise
	# cross product of successive vectors
	contour = contour.reshape((-1, 2)).astype(np.float32)
	vectors = np.roll(contour, -1, axis=0) - contour
	vectors /= np.linalg.norm(vectors, axis=1).reshape((-1, 1))
	crossed = np.arcsin(np.cross(vectors, np.roll(vectors, -1, axis=0)))
	return crossed.sum()

def rotate_topleft(contour):
	distances = np.linalg.norm(contour, axis=(1,2))
	shift = np.argmin(distances)
	return np.roll(contour, -shift, axis=0)
	#return np.vstack([
	#	contour[shift:],
	#	contour[:shift]
	#])



# also acts (partly) like a cv.VideoCapture
class FreshestFrame(threading.Thread):
	def __init__(self, capture, name='FreshestFrame'):
		self.capture = capture
		assert self.capture.isOpened()

		# this lets the read() method block until there's a new frame
		self.cond = threading.Condition()

		# this allows us to stop the thread gracefully
		self.running = False

		# keeping the newest frame around
		self.frame = None

		# passing a sequence number allows read() to NOT block
		# if the currently available one is exactly the one you ask for
		self.latestnum = 0

		# this is just for demo purposes		
		self.callback = None
		
		super().__init__(name=name)
		self.start()

	def start(self):
		self.running = True
		super().start()

	def release(self, timeout=None):
		self.running = False
		self.join(timeout=timeout)
		self.capture.release()

	def run(self):
		counter = 0
		while self.running:
			# block for fresh frame
			(rv, img) = self.capture.read()
			assert rv
			counter += 1

			# publish the frame
			with self.cond: # lock the condition for this operation
				self.frame = img if rv else None
				self.latestnum = counter
				self.cond.notify_all()

			if self.callback:
				self.callback(img)

	def read(self, wait=True, seqnumber=None, timeout=None):
		# with no arguments (wait=True), it always blocks for a fresh frame
		# with wait=False it returns the current frame immediately (polling)
		# with a seqnumber, it blocks until that frame is available (if it even needs to wait)
		# with timeout argument, may return an earlier frame;
		#   may even be (0,None) if nothing received yet

		with self.cond:
			if wait:
				if seqnumber is None:
					seqnumber = self.latestnum+1
				if seqnumber < 1:
					seqnumber = 1
				
				rv = self.cond.wait_for(lambda: self.latestnum >= seqnumber, timeout=timeout)
				if not rv:
					return (self.latestnum, self.frame)

			return (self.latestnum, self.frame)

