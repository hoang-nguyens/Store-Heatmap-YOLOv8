from ultralytics import YOLO
import cv2
import numpy as np




class YoloHeatMap:
    def __init__(self, model, num_rows, num_cols, cell_size):
        self.yolo = YOLO(model)
        self.heat_matrix = np.zeros((num_rows, num_cols))
        self.COLORS = np.random.uniform(0,255,size = (80, 3))
        self.cell_size = cell_size

    def detection(self,is_show = False, image = None):
        result = self.yolo.predict(show = is_show, source = image)#, classes = [0])

        boxes = []
        classes = []
        names = []
        confidence = []

        for re in result:
            box = re.boxes.xyxy.tolist()
            cls = re.boxes.cls.tolist()
            name = re.names
            conf = re.boxes.conf.tolist()

            boxes.append(box)
            classes.append(cls)
            names.append(name)
            confidence.append(conf)
        return boxes, classes, names, confidence

    def draw_prediction(self, image, x_top, y_top, x_bottom, y_bottom, cls, name): # pass integer x, y
        color = self.COLORS[int(cls)]
        cv2.rectangle(image, (x_top, y_top), (x_bottom, y_bottom), color, 2)
        cv2.putText(image, name, (x_top, y_top), cv2.FONT_HERSHEY_PLAIN, 1,color,2)


    def heat_increase(self, x_top, y_top, x_bottom, y_bottom): # pass integer x, y
        y_center = (y_top + y_bottom)//2
        x_center = (x_top + x_bottom)//2
        row = y_center // self.cell_size
        col = x_center // self.cell_size
        self.heat_matrix[row, col] +=1

