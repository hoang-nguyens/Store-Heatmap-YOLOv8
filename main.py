import cv2
from yolo_model import YoloHeatMap
from imutils.video import VideoStream
from skimage.transform import resize
import numpy as np

alpha = 0.3
video_height = 360
video_width = 634
cell_size = 20
num_cols = video_width//20
num_rows = video_height//20

custom_model = YoloHeatMap("yolov8n.pt", num_rows, num_cols, cell_size)



video_file = "video/store video.mp4"
video = cv2.VideoCapture(video_file)

while True:
    ret, frame = video.read()

    if ret:
        cv2.imshow('video',frame)
        if cv2.waitKey(1) == ord('q'):
            break


        boxes, classes, names, _ = custom_model.detection(False, frame)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[0][i]
            cls = classes[0][i]
            name = names[0][i]
            custom_model.draw_prediction(frame, round(x1), round(y1), round(x2), round(y2), cls, name)
            custom_model.heat_increase(round(x1), round(y1), round(x2), round(y2))


        temp_heat_matrix = custom_model.heat_matrix.copy()
        temp_heat_matrix = resize(temp_heat_matrix, (video_height, video_width))
        temp_heat_matrix = temp_heat_matrix / np.max(temp_heat_matrix)
        temp_heat_matrix = np.uint(temp_heat_matrix*255)

        image_heat = cv2.applyColorMap(temp_heat_matrix, cv2.COLORMAP_JET)

        cv2.addWeighted(image_heat, alpha,frame, 1- alpha, 0, frame)




video.release()
cv2.destroyAllWindows()