import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import numpy as np
import threading
import queue
import math
from collections import OrderedDict, defaultdict

class VehicleInfo:
    def __init__(self):
        self.positions = [] 
        self.speeds = []    
        self.warning_level = 0 
        self.warning_frames = 0 
        self.direction = None 
        self.speed = 0      
        self.danger_zone = [] 
        self.collision_frame = 0
        self.stable_frames = 0 

class CentroidTracker:
    def __init__(self, max_disappeared=50, history_length=10):
        self.nextObjectID = 0
        self.objects = OrderedDict()  
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.vehicle_info = defaultdict(VehicleInfo)
        self.history_length = history_length
        self.frame_count = 0

    def register(self, centroid, box):
        self.objects[self.nextObjectID] = (centroid, box)
        self.disappeared[self.nextObjectID] = 0
        self.vehicle_info[self.nextObjectID] = VehicleInfo()
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        del self.vehicle_info[objectID]

    def calculate_danger_zone(self, box, direction):
        x, y, w, h = box
        if direction is None or (direction[0] == 0 and direction[1] == 0):
            return []
        
        front_length = min(w, h) * 0.5 
        
        if abs(direction[0]) > abs(direction[1]): 
            if direction[0] > 0: 
                return [(x+w, y), (x+w+front_length, y), (x+w+front_length, y+h), (x+w, y+h)]
            else: 
                return [(x-front_length, y), (x, y), (x, y+h), (x-front_length, y+h)]
        else: 
            if direction[1] > 0: 
                return [(x, y+h), (x+w, y+h), (x+w, y+h+front_length), (x, y+h+front_length)]
            else:  
                return [(x, y-front_length), (x+w, y-front_length), (x+w, y), (x, y)]

    def update_movement(self, objectID, new_position, box):
        info = self.vehicle_info[objectID]
        
        if len(info.positions) > 0:
            last_pos = info.positions[-1]
            dx = new_position[0] - last_pos[0]
            dy = new_position[1] - last_pos[1]
            distance = math.sqrt(dx**2 + dy**2)
            
            info.speeds.append(distance)
            info.speed = distance
            
            if distance > 2: 
                if info.direction is None:
                    info.direction = (dx/distance, dy/distance)
                else:
                    smooth_factor = 0.3
                    info.direction = (
                        info.direction[0]*(1-smooth_factor) + (dx/distance)*smooth_factor,
                        info.direction[1]*(1-smooth_factor) + (dy/distance)*smooth_factor
                    )
                    norm = math.sqrt(info.direction[0]**2 + info.direction[1]**2)
                    if norm > 0:
                        info.direction = (info.direction[0]/norm, info.direction[1]/norm)
                
                info.stable_frames += 1
            else:
                info.stable_frames = max(0, info.stable_frames-1)
            
            if info.stable_frames > 3:
                info.danger_zone = self.calculate_danger_zone(box, info.direction)
            else:
                info.danger_zone = []
        
        info.positions.append(new_position)
        if len(info.positions) > self.history_length:
            info.positions.pop(0)
        if len(info.speeds) > self.history_length:
            info.speeds.pop(0)

    def check_collision_risk(self, id1, id2):
        info1 = self.vehicle_info[id1]
        info2 = self.vehicle_info[id2]
        box1 = self.objects[id1][1]
        box2 = self.objects[id2][1]
        
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1+w1, x2+w2)
        y_bottom = min(y1+h1, y2+h2)
        
        if x_right < x_left or y_bottom < y_top:
            overlap_area = 0
        else:
            overlap_area = (x_right - x_left) * (y_bottom - y_top)
        
        min_area = min(w1*h1, w2*h2)
        significant_overlap = overlap_area > min_area * 0.4 if min_area > 0 else False
        
        danger_overlap = False
        if info1.danger_zone and info2.danger_zone and len(info1.danger_zone) == 4 and len(info2.danger_zone) == 4:
            dz1 = np.array(info1.danger_zone, dtype=np.int32).reshape((-1,1,2))
            dz2 = np.array(info2.danger_zone, dtype=np.int32).reshape((-1,1,2))
            
            temp_shape = (1080, 1920)
            mask1 = np.zeros(temp_shape, dtype=np.uint8)
            mask2 = np.zeros(temp_shape, dtype=np.uint8)
            
            cv2.fillPoly(mask1, [dz1], 1)
            cv2.fillPoly(mask2, [dz2], 1)
            
            overlap_pixels = np.sum(mask1 & mask2)
            min_pixels = min(np.sum(mask1), np.sum(mask2))
            danger_overlap = (overlap_pixels / min_pixels) > 0.3 if min_pixels > 10 else False
        
        avg_speed1 = np.mean(info1.speeds[-3:]) if len(info1.speeds) >= 3 else info1.speed
        avg_speed2 = np.mean(info2.speeds[-3:]) if len(info2.speeds) >= 3 else info2.speed
        speed_diff = abs(avg_speed1 - avg_speed2)
        speed_threshold = max(5, 0.6 * max(avg_speed1, avg_speed2))  
        high_speed_diff = speed_diff > speed_threshold
        
        converging = False
        if info1.direction and info2.direction and len(info1.positions) > 4 and len(info2.positions) > 4:
            pts1 = np.array(info1.positions[-5:])
            pts2 = np.array(info2.positions[-5:])
            
            vec1 = np.mean(np.diff(pts1, axis=0), axis=0)
            vec2 = np.mean(np.diff(pts2, axis=0), axis=0)
            
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 > 0 and norm2 > 0:
                vec1 = vec1 / norm1
                vec2 = vec2 / norm2
                
                centroid1 = self.objects[id1][0]
                centroid2 = self.objects[id2][0]
                between_vec = np.array(centroid2) - np.array(centroid1)
                between_norm = np.linalg.norm(between_vec)
                
                if between_norm > 0:
                    between_vec = between_vec / between_norm
                    
                    dot1 = np.dot(vec1, between_vec)
                    dot2 = np.dot(vec2, -between_vec)
                    
                    converging = dot1 > 0.9 or dot2 > 0.9  
        
        if significant_overlap:
            if high_speed_diff or converging:
                self.vehicle_info[id1].collision_frame = 10  
                self.vehicle_info[id2].collision_frame = 10
                return 3
            else:
                return 2
        elif danger_overlap and high_speed_diff and converging:
            return 3
        elif (danger_overlap and high_speed_diff) or (danger_overlap and converging):
            return 2
        elif danger_overlap or (converging and speed_diff > 5):
            return 1
        return 0

    def update(self, rects):
        self.frame_count += 1
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = []
        for (x, y, w, h) in rects:
            cX = int(x + w/2)
            cY = int(y + h/2)
            inputCentroids.append((cX, cY))

        if len(self.objects) == 0:
            for i in range(len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = [v[0] for v in self.objects.values()]
            D = np.zeros((len(objectCentroids), len(inputCentroids)))
            for i in range(len(objectCentroids)):
                for j in range(len(inputCentroids)):
                    D[i, j] = math.dist(objectCentroids[i], inputCentroids[j])

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > 50:
                    continue
                objectID = objectIDs[row]
                new_centroid = inputCentroids[col]
                new_box = rects[col]
                self.objects[objectID] = (new_centroid, new_box)
                self.disappeared[objectID] = 0
                self.update_movement(objectID, new_centroid, new_box)
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(D.shape[0])).difference(usedRows)
            unusedCols = set(range(D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col], rects[col])

        return self.objects

def load_yolo():
    net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
    with open('coco.names', 'r') as f:
        classes = f.read().strip().split('\n')
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers().flatten()]
    return net, classes, output_layers

def detect_vehicles(net, output_layers, frame, classes, vehicle_classes):
    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416,416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in vehicle_classes:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    vehicles = []
    if len(indices) > 0:
        for i in indices.flatten():
            vehicles.append({'box': boxes[i], 'class': classes[class_ids[i]], 'confidence': confidences[i]})
    return vehicles

def capture_frames(video_path, frame_queue, stop_flag):
    cap = cv2.VideoCapture(video_path)
    while not stop_flag.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    stop_flag.set()
    cap.release()

def process_frames(input_queue, output_queue, stop_flag):
    net, classes, output_layers = load_yolo()
    vehicle_classes = ['car', 'motorbike', 'bus', 'truck']
    tracker = CentroidTracker(max_disappeared=30, history_length=10)

    while not stop_flag.is_set() or not input_queue.empty():
        try:
            frame = input_queue.get(timeout=0.5)
        except queue.Empty:
            continue

        vehicles = detect_vehicles(net, output_layers, frame, classes, vehicle_classes)
        rects = [v['box'] for v in vehicles]
        objects = tracker.update(rects)

        object_ids = list(objects.keys())
        
        for i in range(len(object_ids)):
            for j in range(i+1, len(object_ids)):
                id1 = object_ids[i]
                id2 = object_ids[j]
                
                risk_level = tracker.check_collision_risk(id1, id2)
                
                if risk_level >= 3:  
                    tracker.vehicle_info[id1].warning_level = max(tracker.vehicle_info[id1].warning_level, 3)
                    tracker.vehicle_info[id2].warning_level = max(tracker.vehicle_info[id2].warning_level, 3)
                    tracker.vehicle_info[id1].warning_frames += 2  
                    tracker.vehicle_info[id2].warning_frames += 2
                elif risk_level == 2: 
                    tracker.vehicle_info[id1].warning_level = max(tracker.vehicle_info[id1].warning_level, 2)
                    tracker.vehicle_info[id2].warning_level = max(tracker.vehicle_info[id2].warning_level, 2)
                    tracker.vehicle_info[id1].warning_frames += 1
                    tracker.vehicle_info[id2].warning_frames += 1
                elif risk_level == 1:  
                    tracker.vehicle_info[id1].warning_level = max(tracker.vehicle_info[id1].warning_level, 1)
                    tracker.vehicle_info[id2].warning_level = max(tracker.vehicle_info[id2].warning_level, 1)

        for id in object_ids:
            centroid, box = objects[id]
            x, y, w, h = box
            info = tracker.vehicle_info[id]
            
            if info.collision_frame > 0:
                color = (0, 0, 255)  
                cv2.rectangle(frame, (x,y), (x+w, y+h), color, 1)
                
                warning_text = "COLISAO!"
                text_size = cv2.getTextSize(warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                text_x = x + (w - text_size[0]) // 2
                cv2.putText(frame, warning_text, (text_x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                info.collision_frame -= 1
            elif info.warning_level >= 3 and info.warning_frames > 3:
                color = (0, 0, 255)  
                warning_text = "ATENCAO!"
                cv2.putText(frame, warning_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
                if info.danger_zone:
                    dz = np.array(info.danger_zone, dtype=np.int32).reshape((-1,1,2))
                    cv2.polylines(frame, [dz], True, color, 1)
            elif info.warning_level >= 2:
                color = (0, 165, 255) 
                if info.danger_zone:
                    dz = np.array(info.danger_zone, dtype=np.int32).reshape((-1,1,2))
                    cv2.polylines(frame, [dz], True, color, 1)
            elif info.warning_level == 1:
                color = (0, 255, 255) 
            else:
                color = (0, 255, 0)  
            
            cv2.rectangle(frame, (x,y), (x+w, y+h), color, 1)
            
            cv2.putText(frame, f'{id}', (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            
            if info.warning_frames > 0:
                info.warning_frames -= 1
            if info.warning_frames == 0:
                info.warning_level = 0

        output_queue.put(frame)

def main(video_path):
    input_queue = queue.Queue(maxsize=30)  
    output_queue = queue.Queue(maxsize=30)
    stop_flag = threading.Event()

    t1 = threading.Thread(target=capture_frames, args=(video_path, input_queue, stop_flag))
    t2 = threading.Thread(target=process_frames, args=(input_queue, output_queue, stop_flag))

    t1.start()
    t2.start()

    while True:
        try:
            frame = output_queue.get(timeout=1)
            cv2.imshow('Enhanced Collision Detection', frame)
            if cv2.waitKey(1) & 0xFF == 27: 
                stop_flag.set()
                break
        except queue.Empty:
            if stop_flag.is_set() and input_queue.empty() and output_queue.empty():
                break

    t1.join()
    t2.join()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video_path = 'acidente.mp4'  
    main(video_path)