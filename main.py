import os

from  datetime import datetime
from nets.unet import Unet

# from colormap import Color


#from detectron2.structures import Boxes
from sklearn.decomposition import PCA

from tqdm import tqdm
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

from glob import glob
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from glob import glob
from nets.unet import Unet as unet
from utils.utils import resize_image
import copy
import torch.nn.functional as F
from skimage.morphology import skeletonize
from skimage import img_as_ubyte
from sklearn.neighbors import KDTree
from PIL import Image

import scipy
import pandas as pd
import json,sys
from numba import jit
from utils.utils import cvtColor, preprocess_input, resize_image, show_config

sys.setrecursionlimit(200000)
def resize_and_pad(img, target_size=640):
    h, w, _ = img.shape
    scale = target_size / max(h, w)
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    h_resized, w_resized, _ = img_resized.shape
    img_padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    pad_y = (target_size - h_resized) // 2
    pad_x = (target_size - w_resized) // 2
    img_padded[pad_y:pad_y+h_resized, pad_x:pad_x+w_resized] = img_resized
    return img_padded,w_resized,h_resized
def get_adjacent(point, kdtree, points):
    adjacent_points_index = kdtree.query_radius([point], r=2**0.5)[0]
    adjacent_points = [points[i] for i in adjacent_points_index]
    return adjacent_points

def dfs(node, end_node, visited, kdtree, points, path):

    visited.add(node)
    path.append(node)
    if node == end_node:
        return True, path
    adjacent_nodes = get_adjacent(node, kdtree, points)
    
    for adj_node in adjacent_nodes:
        if adj_node not in visited:
            found, path = dfs(adj_node, end_node, visited, kdtree, points, path)
            if found:
                return True, path
    path.pop() 
    return False, path

def find_path(points):
    kdtree = KDTree(points)
    visited = set()
    start_node = min(points, key=lambda x: x[0])
    end_node = max(points, key=lambda x: x[0])
    _, path = dfs(start_node, end_node, visited, kdtree, points, [])
    return path

def get_angle(cross_point,curve_points,index):
    x_c,y_c=cross_point
    x=[i[0] for i in curve_points]
    y=[i[1] for i in curve_points]
    start_idx = max(index - 5, 0)
    end_idx = min(index + 6, len(x))

    x_nearby = x[start_idx:end_idx]
    y_nearby = y[start_idx:end_idx]
    coefficients = np.polyfit(x_nearby, y_nearby, 1)
    return coefficients[0]
def get_tangent_line(x,y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    if m==0:
        return -1/(m+1e-8)
    return -1/m
@jit(nopython=True)
def calculate_x_from_y(y, m, x1, y1):
    x = (y - y1 + m * x1) / m
    return x
    
def find_cross_point(x1, y1, x2, y2, curve_points):
    m = (y2 - y1) / (x2 - x1)
    min_distance = float("inf")
    cross_point = None
    mark_index=None
    for index,(x, y) in enumerate(curve_points):
        y_line = m * (x - x1) + y1
        distance = abs(y - y_line)
        if distance < min_distance:
            min_distance = distance
            cross_point = (x, y)
            mark_index = index
    if min_distance <= 1 :
        return cross_point,mark_index
    return None,None
def find_min_distance(target, curve_points):
    distances = np.sqrt(np.sum((curve_points - target) ** 2, axis=1))
    min_distance = np.min(distances)
    return min_distance
def get_width_fit(centerPoint,k,wid,stride=1):  #
    res_width=0
    h,w=wid.shape
    m=centerPoint[1]
    while True:
        #search_point1=(y2x(crd[i],m,temp_y1,temp_y2),m)  #（点0,点0y坐标，点-1坐标，点1坐标)
        search_point1=(calculate_x_from_y(m,k,centerPoint[0],centerPoint[1]),m)
        if round(search_point1[0])<0 or round(search_point1[0])>=w or round(search_point1[1])<0 or round(search_point1[1])>=h:
            break
        if wid[round(search_point1[1]),round(search_point1[0])]==0:
            break
        m+=1
    m=centerPoint[1]
    while True:
        #search_point2=(y2x(crd[i],m,temp_y1,temp_y2),m)
        
        search_point2=(calculate_x_from_y(m,k,centerPoint[0],centerPoint[1]),m)
        if round(search_point2[0])<0 or round(search_point2[0])>=w or round(search_point2[1])<0 or round(search_point2[1])>=h:
            break
        if wid[round(search_point2[1]),round(search_point2[0])]==0:
            break
        m-=1
        
    width=get_line_length(list(map(round,search_point1)),list(map(round,search_point2)))
    return width,search_point1,search_point2
def extend_to_edge(pt1, pt2, shape):
    x1, y1 = pt1
    x2, y2 = pt2
    slope = (y2 - y1) / (x2 - x1 + 1e-10)
    
    new_x1 = 0
    new_y1 = int(y1 - slope * (x1 - 0))
    
    new_x2 = shape[1] - 1
    new_y2 = int(slope * (new_x2 - x1) + y1)
    
    return (new_x1, new_y1), (new_x2, new_y2)
def get_mask_left(output):
    mask=(output==1).astype(np.uint8)*255
    stem=(output==2).astype(np.uint8)*255
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) 
    max_contour1=contours[0]
    contours, hierarchy = cv2.findContours(stem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True) 
    if len(contours)==0:
        height,width=mask.shape[:2]
        x1,y1, w1, h1 = cv2.boundingRect(max_contour1)
        left=(x1<(width-(x1+w1)))
    else:
        max_contour2=contours[0]
        x1,y1, w1, h1 = cv2.boundingRect(max_contour1)
        x2, _, _, _ = cv2.boundingRect(max_contour2)
        left=x1>x2
    top = 2*w1<h1
    return mask,left,max_contour1,top

def rotate_back(point, H, W,rotate_mark):
    x_prime, y_prime = point
    if rotate_mark ==90:
        x = y_prime - 1
        y = H - x_prime
        return x, y
    elif rotate_mark ==180:
        x = W - x_prime
        y = H - y_prime
        return x, y
    return x_prime,y_prime
def get_direction(output):
    
    mask=(output==2).astype(np.uint8)*255
    height, width = mask.shape
    upper_half = mask[0:height//20, :]
    lower_half = mask[height-height//20:, :]
    left_half = mask[:, 0:width//20]
    right_half = mask[:, width-width//20:]
    # 计算每个区域中白色像素的数量
    upper_white = np.sum(upper_half == 255)
    lower_white = np.sum(lower_half == 255)
    left_white = np.sum(left_half == 255)
    right_white = np.sum(right_half == 255)

    # 判断白色区域在哪个边缘
    if upper_white > lower_white and upper_white > left_white and upper_white > right_white:
        return 0
    elif lower_white > upper_white and lower_white > left_white and lower_white > right_white:
        return 1
    elif left_white > upper_white and left_white > lower_white and left_white > right_white:
        return 3
    # elif right_white > upper_white and right_white > lower_white and right_white > left_white:
    return 4

def get_pixel_length(img):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    average_brightness = 5*np.mean(hsv[:,:,-1])
    thresh = cv2.inRange(hsv[:,:,-1], average_brightness, 255)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask=np.zeros_like(img[:,:,0])
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if area>400 and perimeter>400 and (4*np.pi*area)/perimeter**2>0.8:
            mask=cv2.fillPoly(mask,[contour],(255))
            res=cv2.bitwise_and(img,img,mask=mask)
            
            return 1/np.sqrt(area/np.pi)
        
def mask_nms(masks, scores, iou_threshold=0.1, score_threshold=0.1):
    _, indices = torch.sort(scores, descending=True)
    keep = []
    while indices.numel() > 0:
        i = indices[0]
        if scores[i] < score_threshold:
            break
        keep.append(i)
        ious = calc_mask_iou(masks[i], masks[indices[1:]])
        indices = indices[1:][ious <= iou_threshold]     
    return torch.tensor(keep, dtype=torch.long)   
def calc_mask_iou(mask1, masks2):
    mask1=mask1.long()
    masks2=masks2.long()
    inter_area = (mask1 & masks2).sum(dim=(1, 2))
    mask1_area = mask1.sum()
    mask2_area = masks2.sum(dim=(1, 2))
    iou = inter_area / (mask1_area + mask2_area - inter_area)
    return iou
def split_and_calculate_area(m, point, mask):
    height, width = mask.shape
    x1, y1 = point
    c = y1 - m * x1
    y, x = np.ogrid[:height, :width]
    mask1 = y > m * x + c
    mask2 = ~mask1
    area1 = np.sum(mask[mask1] == 255)
    area2 = np.sum(mask[mask2] == 255)
    return area1, area2

@jit(nopython=True)
def compute_distance(point1, point2):
    return ((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)**0.5

@jit(nopython=True)
def compute_curve_length(coords):
    length = 0.0
    for i in range(1, len(coords)):
        length += compute_distance(coords[i - 1], coords[i])
    return length

@jit(nopython=True)
def y2x(point,y,a,b):
    if a[0]!=b[0]:
        return -(a[1]-b[1])/(a[0]-b[0])*(y-point[1])+point[0]
    return point[0]

@jit(nopython=True)
def get_line(x1, y1, x2, y2):
    points = []
    issteep = abs(y2-y1) > abs(x2-x1)
    if issteep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    rev = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        rev = True
    deltax = x2 - x1
    deltay = abs(y2-y1)
    error = int(deltax / 2)
    y = y2
    if y1 < y2:
        ystep = 1
    else:
        ystep = -1
    for x in range(x2, x1-1, -1):
        if issteep:
            points.append((y, x))
        else:
            points.append((x, y))
        error -= deltay
        if error < 0:
            y -= ystep
            error += deltax
    if rev:
        points.reverse()
    return points

@jit(nopython=True)
def get_line_length(a,b):
    return ((a[1]-b[1])**2+(a[0]-b[0])**2)**0.5

def get_width(crd,wid,pixel_length): #
    res_width=0
    h,w=wid.shape
    for i in range(len(crd)//4,len(crd)//4*3):
        points_list=[]
        for pi_n in range(-5,6):
            points_list.append(crd[i+pi_n])
        m=get_tangent_line([i[0] for i in points_list],[i[1] for i in points_list])
        try:
            temp_width=calculate_white_length(crd[i],m,wid)*pixel_length 
        except:
            continue
        if temp_width>res_width and temp_width<3:
            res_width=temp_width
    return res_width


@jit(nopython=True)
def calculate_white_length(center_point, slope, mask):
    x0, y0 = center_point
    height, width = mask.shape
    white_x = np.empty(height, dtype=np.int32)
    white_y = np.empty(height, dtype=np.int32)
    count = 0
    if slope != 0:
        for y in range(height):
            x = int(x0 + (y - y0) / slope)
            if 0 <= x < width:
                if mask[y, x] == 255:
                    white_x[count] = x
                    white_y[count] = y
                    count += 1
    else:
        x = int(x0)
        for y in range(height):
            if mask[y, x] == 255:
                white_x[count] = x
                white_y[count] = y
                count += 1
    if count > 0:
        white_x = white_x[:count]
        white_y = white_y[:count]
        first_point = (white_x[0], white_y[0])
        last_point = (white_x[-1], white_y[-1])
        length = np.sqrt((first_point[0] - last_point[0])**2 + (first_point[1] - last_point[1])**2)
    else:
        length = 0
    return length

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    elif isinstance(o, np.float32):
        return float(o)
    elif isinstance(o, np.float64):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.bool_):
        return bool(o)
    else:
        raise TypeError(f"Unserializable object {o} of type {type(o)}")

        
from ultralytics import YOLO
# import numpy as np
# import cv2
# from colormap import Color
# import matplotlib.pyplot as plt

model = YOLO('spikelet.yolo.segm.pt')
model1 = Unet(num_classes=3, pretrained=False, backbone='resnet50').cuda()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model1.load_state_dict(torch.load('spike.segm.best.pth', map_location=device),strict=False)
model1 = model1.eval()


from glob import glob
import copy
files=glob('test_images/*S*.PNG')
output_file='wheat.spike.pheno.output.json'

files.sort()

def euclidean_distance(point1,points):
    min_distance=float('inf')
    for point2 in points:
        distance=((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
        if min_distance>distance:
            min_distance=distance
            point=point2
    return min_distance,point
    
result={}
findex=0
for file in tqdm(files[:]):
    image = Image.open(file)
    filename=os.path.basename(file)
    result[filename]={}
    #old_img = copy.deepcopy(image)
    orininal_h  = np.array(image).shape[0]
    orininal_w  = np.array(image).shape[1]
    old_img = copy.deepcopy(np.array(image))
    pixel_length=get_pixel_length(old_img)
    image_data, nw, nh  = resize_image(image, (1024,1024))
    image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)
    with torch.no_grad():
        images = torch.from_numpy(image_data)
        images = images.cuda()
        pr = model1(images)[0]
        pr = F.softmax(pr.permute(1,2,0),dim = -1).cpu().numpy()
        pr = pr[int((1024 - nh) // 2) : int((1024 - nh) // 2 + nh), \
                int((1024 - nw) // 2) : int((1024 - nw) // 2 + nw)]
        pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation = cv2.INTER_LINEAR)
        pr = pr.argmax(axis=-1)
        pr = pr==1
        pr=pr.astype(np.uint8)*255
        mask=cv2.merge([pr,pr,pr]).astype(np.uint8)
        contours, hierarchy = cv2.findContours(mask[:,:,0], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contours=list(contours)
        contours.sort(key = cv2.contourArea, reverse=True)
        contour=contours[0]
        x,y,w,h=cv2.boundingRect(contour)
        mask=cv2.bitwise_and(old_img,old_img,mask=pr)
        max_edge=max(h,w)
        newpr=np.zeros((max_edge,max_edge))
        newpr[(max_edge-h)//2:(max_edge-h)//2+h,(max_edge-w)//2:(max_edge-w)//2+w]=pr[y:y+h,x:x+w]
        mask=mask[y:y+h,x:x+w]
        contours, _ = cv2.findContours(newpr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        newpr=np.zeros((max_edge,max_edge))
        cv2.drawContours(newpr, [max_contour], -1, color=255, thickness=cv2.FILLED)
        skeleton = skeletonize(newpr)
        y, x = np.where(skeleton)
        skeleton_coords = list(zip(x, y))
        skeleton_coords.sort()
        skeleton_coords = find_path(skeleton_coords)
        x=[i[0] for i in skeleton_coords]
        y=[i[1] for i in skeleton_coords]
        left_points = [skeleton_coords[0],skeleton_coords[10]]
        right_points = [skeleton_coords[-11],skeleton_coords[-1]] #skeleton_coords[-2:]
        left_slope, left_intercept = np.polyfit([point[0] for point in left_points], [point[1] for point in left_points], 1)
        right_slope, right_intercept = np.polyfit([point[0] for point in right_points], [point[1] for point in right_points], 1)
        leftmost_point = min(skeleton_coords, key=lambda coord: coord[0])
        rightmost_point = max(skeleton_coords, key=lambda coord: coord[0])
        extended_left = [(x, int(left_slope * x + left_intercept)) for x in range(0, leftmost_point[0])]
        extended_left = [coord for coord in extended_left if 0 <= coord[1] < newpr.shape[0]]
        extended_right = [(x, int(right_slope * x + right_intercept)) for x in range(rightmost_point[0], newpr.shape[1])]
        extended_right = [coord for coord in extended_right if 0 <= coord[1] < newpr.shape[0]]
        extended_skeleton_coords = extended_left + skeleton_coords + extended_right
        extended_skeleton_coords.sort()
        x=[i[0] for i in extended_skeleton_coords]
        y=[i[1] for i in extended_skeleton_coords]
        try:
            y=scipy.signal.savgol_filter(y,len(y)//2,1)
        except:
            continue
        raw_extended_skeleton_coords=list(zip(x,y))

        width=get_width(raw_extended_skeleton_coords, newpr, pixel_length)

        result[filename]['spike']={}
        result[filename]['spike']['width']=width
        length=compute_curve_length(np.array(raw_extended_skeleton_coords))*pixel_length



        points_index=[(i+1)*len(raw_extended_skeleton_coords)//21 for i in range(21)][:-1]
        grading_width=[]
        for pi in points_index:
            center_point=raw_extended_skeleton_coords[pi]
            points_list=[]
            for pi_n in range(-5,6):
                points_list.append(raw_extended_skeleton_coords[pi+pi_n])
            m=get_tangent_line([i[0] for i in points_list],[i[1] for i in points_list])

            width=calculate_white_length(center_point,m,newpr)
            grading_width.append(width*pixel_length)

        #面积、周长
        perimeter = cv2.arcLength(max_contour, True) *pixel_length   #周长
        area = cv2.contourArea(max_contour)  * pixel_length**2     #面积
        #面积/周长
        area_peri=area/perimeter

        #面积/最小外接矩面积
        rect = cv2.minAreaRect(max_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        width, height = rect[1]
        area_rect = width * height
        area_rect_ratio=area/(area_rect* pixel_length**2)
        #椭圆形
        ellipse = cv2.fitEllipse(max_contour)
        major_axis, minor_axis = ellipse[1]
        axis_ratio = major_axis / minor_axis

        #前后面积比
        mid_point=raw_extended_skeleton_coords[len(raw_extended_skeleton_coords)//2]
        area1,area2=split_and_calculate_area(m,mid_point,newpr)
        area_ratio=area1/area2
        
        result[filename]['spike']['length']=length
        result[filename]['spike']['grading_width']=grading_width
        result[filename]['spike']['perimeter']=perimeter
        result[filename]['spike']['area']=area
        result[filename]['spike']['area_peri']=area_peri
        result[filename]['spike']['axis_ratio']=axis_ratio
        result[filename]['spike']['area_ratio']=area_ratio
        result[filename]['spike']['area_rect_ratio']=area_rect_ratio

        
        
        
        curve_points = raw_extended_skeleton_coords #list(map(lambda p: rotate_back(p, orininal_h, orininal_w,0), raw_extended_skeleton_coords))
        max_edge=max(h,w)
        new=np.zeros((max_edge,max_edge,3))
        new[(max_edge-h)//2:(max_edge-h)//2+h,(max_edge-w)//2:(max_edge-w)//2+w]=mask[:,:,::-1]
        new=new.astype(np.uint8)
        orininal_h,orininal_w  = new.shape[:2]
        draw_img=new.copy()
        new=cv2.resize(new,(960,960))
        output=model.predict(new,verbose=False)[0]
        pred_masks=output.masks.data#.cpu().numpy()
        pred_boxes=output.boxes.data#.cpu().numpy()
        scores = output.boxes.conf#.data.cpu().numpy()
        pred_classes = output.boxes.cls.data#.cpu().numpy()



        
        indices = mask_nms(pred_masks, scores,iou_threshold=0.1,score_threshold=0.1)
        pred_masks = pred_masks[indices]#.cpu().numpy()
        pred_boxes = pred_boxes[indices]#.cpu().numpy()
        scores = scores[indices]#.cpu().numpy()
        pred_classes = pred_classes[indices]#.cpu().numpy()
        
        if len(pred_masks)==0:
            continue
        
        #对实例排序
        bboxes = pred_boxes
        curve_points=curve_points[::-1]
        centers_x = (bboxes[:, 0] + bboxes[:, 2]) / 2
        centers_y = (bboxes[:, 1] + bboxes[:, 3]) / 2
        centers = torch.stack((centers_x, centers_y), dim=1)
        curve_points_tensor = torch.tensor(curve_points, dtype=torch.float32).cuda()
        nearest_curve_indices = []
        for center in centers:
            center_reshaped = center.unsqueeze(0)
            distances = torch.norm(curve_points_tensor - center_reshaped, dim=1)
            nearest_curve_idx = torch.argmin(distances).item()
            nearest_curve_indices.append(nearest_curve_idx)
        sorted_indices = [x for _, x in sorted(zip(nearest_curve_indices, list(range(len(centers)))))]
        sorted_bboxes = bboxes[sorted_indices]
        sorted_masks = pred_masks[sorted_indices]
        sorted_classes = pred_classes[sorted_indices]
        sorted_scores = scores[sorted_indices]
        
        spikelet=[]
        seed_mask=sorted_masks.cpu().numpy()
        last_base_point=None
        for sm_index,sm in enumerate(seed_mask):
            rst={}
            #计算间距
            
            try:
                contours, hierarchy = cv2.findContours(sm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                max_contour = max(contours, key=cv2.contourArea)
                sm = np.zeros_like(sm,dtype=np.uint8)
                cv2.drawContours(sm, [max_contour], 0, (255), thickness=cv2.FILLED)
                sm=cv2.resize(sm,(draw_img.shape[1],draw_img.shape[0]))
                
                y, x = np.nonzero(sm)
                coords = np.column_stack((x, y))
                
                
                pca = PCA(n_components=2)
                pca.fit(coords)
                direction = pca.components_[0]
                mean = pca.mean_
                projection = np.dot(coords - mean, direction)
                min_proj_point = coords[np.argmin(projection)]
                max_proj_point = coords[np.argmax(projection)]
                x1,y1=min_proj_point
                x2,y2=max_proj_point
                # min_distance_to_curve_from_min_proj = find_min_distance(min_proj_point, curve_points)
                # min_distance_to_curve_from_max_proj = find_min_distance(max_proj_point, curve_points)
                if min_proj_point[0] > max_proj_point[0]:
                    base_point=min_proj_point
                    top_point=max_proj_point
                else:
                    base_point=max_proj_point
                    top_point=min_proj_point
                if sm_index==0:
                    distance=None
                else:
                    distance=((base_point[0]-last_base_point[0])**2+(base_point[1]-last_base_point[1])**2)**0.5*pixel_length
                last_base_point=base_point
                # print(distance)
                rst['distance']=distance
            except:
                continue
            #类别数
            
            rst['class']=sorted_classes.cpu().numpy()[sm_index]
            #计算夹角、面积
            if sm_index==len(seed_mask)-1:   #顶部的不算
                rst['angle']=None
                rst['area']=None
                rst['perimeter']=None
                rst['length']=None
                spikelet.append(rst)
                continue
            cross_point,point_index=find_cross_point(x1, y1, x2, y2, curve_points)
            if cross_point==None:
                dist_to_start = ((x1 - curve_points[0][0])**2 + (y1 - curve_points[0][1])**2) ** 0.5
                dist_to_end =   ((x1 - curve_points[-1][0])**2 + (y1 - curve_points[-1][1])**2) ** 0.5
                if dist_to_start < dist_to_end:
                    point_index = 0
                else:
                    point_index = len(curve_points)-1
                cross_point=curve_points[point_index]
            if x2 - x1==0:
                slope1=float('inf')
            else:
                slope1= (y2 - y1) / (x2 - x1)
                
            theta_radians1 = np.arctan(slope1)
            theta_degrees1 = np.degrees(theta_radians1)
            curve_angle_slope = get_angle(cross_point,curve_points,point_index)
            theta_radians2 = np.arctan(curve_angle_slope)
            theta_degrees2 = np.degrees(theta_radians2)
            theta=theta_degrees1-theta_degrees2
            
                
            sm=sm.astype(np.uint8)
            seed_area = cv2.countNonZero(sm)*pixel_length**2
            contours, _ = cv2.findContours(sm, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            perimeter = cv2.arcLength(contours[0], True)*pixel_length
            length=  ((min_proj_point[0]-max_proj_point[0])**2+(min_proj_point[1]-max_proj_point[1])**2)**0.5*pixel_length
            semiwid,point=euclidean_distance(top_point,raw_extended_skeleton_coords)
            
            rst['semi_width']=semiwid*pixel_length
            rst['angle']=theta
            rst['area']=seed_area
            rst['perimeter']=perimeter
            rst['length']=length
            
            spikelet.append(rst)
        result[filename]['spikelet']=spikelet
        findex+=1
        if findex%10==0:
            with open(output_file,'w') as f:
                json.dump(result,f, default=convert)
        
with open(output_file,'w') as f:
    json.dump(result,f, default=convert)

    
    