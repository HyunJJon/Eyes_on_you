import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
#CUDAExecutionProvider

app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

faces = app.get(frame)
rimg = app.draw_on(frame, faces)
cv2.imwrite("./t1_output.jpg", rimg)