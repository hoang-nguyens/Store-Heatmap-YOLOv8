# [YOLO-OPENCV] Store HeatMap with YOLOv8 and OpenCV
## Introdution
In a store, some areas receive more customer traffic than others. We want to identify the best places to arrange our grocery items to increase sales. One way to do this is by using data from our security cameeras to create a heat map. The area with higher customer traffic will appear in hotter colors, helping us indentify those spots.<br>
We use YOLOv8 for object dectection, then divide the frames into grid and calculate the time each cell is visited by customers. The color intensy of each cell increases based on the time spent by people in that area.
