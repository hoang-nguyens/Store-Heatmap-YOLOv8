import cv2
from yolo_model import YoloHeatMap
from imutils.video import VideoStream
from skimage.transform import resize
import numpy as np
import argparse

parser = argparse.ArgumentParser(description= "Process video with YOLO and heatmap")
parser.add_argument('--video', type = str, default="video.store video.mp4", help = 'Path to the input video file')
parser.add_argument('--model', type = str, default = 'yolov8s.pt', help = 'Path to YOLO model file')
args = parser.parse_args()

video_file = args.video
model_file = args.model

alpha = 0.4
video_height = 640
video_width = 640
cell_size = 40
num_cols = video_width// cell_size
num_rows = video_height// cell_size

custom_model = YoloHeatMap(model_file, num_rows, num_cols, cell_size)



video = cv2.VideoCapture(video_file)
#video = VideoStream(video_file).start()

while True:
    ret, frame = video.read()
    if not ret:
        break

    frame = cv2.resize(frame, (video_width, video_height))


    boxes, classes, names, _ = custom_model.detection(False, frame)

    for i in range(len(boxes[0])):
        x1, y1, x2, y2 = boxes[0][i]
        cls = classes[0][i]
        name = names[0][i]
        custom_model.draw_prediction(frame, round(x1), round(y1), round(x2), round(y2), cls, name)
        custom_model.heat_increase(round(x1), round(y1), round(x2), round(y2))


    temp_heat_matrix = custom_model.heat_matrix.copy()
    temp_heat_matrix = resize(temp_heat_matrix, (video_height, video_width))
    temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
    temp_heat_matrix = np.uint8(temp_heat_matrix * 255)

    image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

    cv2.addWeighted(image_heat, alpha, frame, 1- alpha, 0, frame)

    cv2.imshow("Store", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()