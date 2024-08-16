
# [YOLO-OPENCV] Store HeatMap with YOLOv8 and OpenCV
## Introdution
In a store, some areas receive more customer traffic than others. We want to identify the best places to arrange our grocery items to increase sales. One way to do this is by using data from our security cameras to create a heat map. The area with higher customer traffic will appear in hotter colors, helping us indentify those spots.<br>
We use YOLOv8 for object dectection, then divide the frames into grid and calculate the time each cell is visited by customers. The color intensy of each cell increases based on the time spent by people in that area.

https://github.com/user-attachments/assets/71dab92c-f9c7-4f9a-9b20-0ef8e65c9c05

<video src="https://github.com/hoang-nguyens/Store-Heatmap-YOLOv8/blob/main/demo/demo1.mp4" autoplay loop>

## How to use
Run pip install -r requirement.txt<br>
Run python main.py --video [your testing video] --model [your model/default is YOLOv8s]

## Requirement
* cv2<br>
* python 3.12<br>
* scikit-image<br>
* ultralytics<br>
* imutils
