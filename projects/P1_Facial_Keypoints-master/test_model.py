import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from models import Net


def normalize(image):
	image_copy = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

	# scale color range from [0, 255] to [0, 1]
	image_copy = image_copy / 255.0

	return image_copy


def rescale(image, output_size):
	h, w = image.shape[:2]
	if isinstance(output_size, int):
		if h > w:
			new_h, new_w = output_size * h / w, output_size
		else:
			new_h, new_w = output_size, output_size * w / h
	else:
		new_h, new_w = output_size

	new_h, new_w = int(new_h), int(new_w)
	img = cv2.resize(image, (new_w, new_h))

	return img


def toTensor(image):
		if (len(image.shape) == 2):
			image = image.reshape(1, image.shape[0], image.shape[1], 1)
		image = image.transpose((0, 3, 1, 2))

		return torch.from_numpy(image)

net = Net()
net.load_state_dict(torch.load('saved_models/keypoints_model_cpu.pt'))
net.eval()

cascPath = 'detector_architectures/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascPath)
cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640, 480))

last_y = 0
while cv2.waitKey(1) != 27:
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(
		gray,
		scaleFactor=1.1,
		minNeighbors=5,
		minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE
	)
	if len(faces) == 0:
		# cv2.imshow("Whole frame", frame)
		# cv2.waitKey(1)
		continue

	x, y, w, h = faces[0]
	if w > h:
		offset = int((w - h) / 2)
		y -= offset
	elif h > w:
		offset = int((h - w) / 2)
		x -= offset
	if x < 0 or y < 0:
		print('too close to boundary!!')
		continue
	w = h
	y += 10

	if abs(last_y - y > 10):
		last_y = y
		continue
	last_y = y

	# cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
	face = frame[y : y + h, x : x + w, :]
	face = rescale(face, 224)

	face_copy = np.copy(face)
	face_copy = normalize(face_copy)
	face_copy = toTensor(face_copy)
	if (torch.cuda.is_available()):
		images = face_copy.type(torch.cuda.FloatTensor)
	else:
		images = face_copy.type(torch.FloatTensor)
	output_pt = net(images)
	output_pt = output_pt.view(68, -1)
	predicted_key_pts = output_pt.data.cpu().numpy() * 50.0 + 100
	for i in range(68):
		cv2.circle(face, (predicted_key_pts[i][0], predicted_key_pts[i][1]), 2, (0, 255, 0), -1)
	face = rescale(face, int(w))
	frame[y: y + h, x: x + w, :] = face
	# face = cv2.resize(face, (480,480))
	out.write(frame)
	#
	# cv2.imshow("face", face)
	cv2.imshow("Whole frame", frame)
	cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()