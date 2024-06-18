import numpy as np
import torch
import time
import os
import sys
import argparse
from typing import List
import cv2

from concurrent.futures import thread
from sqlalchemy import null
from torchvision import transforms
import time
from threading import Thread

import warnings

# Ignore the specific warning by category
warnings.filterwarnings("ignore", category=UserWarning)


sys.path.insert(0, 'Character-Time-series-Matching/yolov5/')
from utils_lp.general_lp import non_max_suppression, scale_coords
from models_lp.experimental_lp import attempt_load






class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        # cwd = os.path.dirname(__file__)
        self.device=device
        self.char_model, self.names = self.load_model(weights_path)
        self.size=size
        
        self.iou_thres=iou_thres
        self.conf_thres=conf_thres

    def detect(self, frame):
        
        results, resized_img = self.char_detection_yolo(frame)

        return results, resized_img
    
    def preprocess_image(self, original_image):

        resized_img = self.ResizeImg(original_image,size=self.size)
        # resized_img = original_image.copy()
        image = resized_img.copy()[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        image = np.ascontiguousarray(image)

        image = torch.from_numpy(image).to(self.device)
        image = image.float()
        image = image / 255.0
        if image.ndimension() == 3:
            image = image.unsqueeze(0)
        return image, resized_img
    
    def char_detection_yolo(self, image, classes=None, \
                            agnostic_nms=True, max_det=1000):

        img,resized_img = self.preprocess_image(image.copy())
        # print(resized_img.shape, image.shape)
        pred = self.char_model(img, augment=False)[0]
        
        detections = non_max_suppression(pred, conf_thres=self.conf_thres,
                                            iou_thres=self.iou_thres,
                                            classes=classes,
                                            agnostic=agnostic_nms,
                                            multi_label=True,
                                            labels=(),
                                            max_det=max_det)
        results=[]
        for i, det in enumerate(detections):
            det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
            det=det.tolist()
            if len(det):
                for *xyxy, conf, cls in det:
                    # xc,yc,w_,h_=(xyxy[0]+xyxy[2])/2,(xyxy[1]+xyxy[3])/2,(xyxy[2]-xyxy[0]),(xyxy[3]-xyxy[1])
                    result=[self.names[int(cls)], str(conf), (xyxy[0],xyxy[1],xyxy[2],xyxy[3])]
                    results.append(result)
        # print(results)
        return results, resized_img
        
    def ResizeImg(self, img, size):
        h1, w1, _ = img.shape
        # print(h1, w1, _)
        h, w = size
        if w1 < h1 * (w / h):
            # print(w1/h1)
            img_rs = cv2.resize(img, (int(float(w1 / h1) * h), h))
            mask = np.zeros((h, w - (int(float(w1 / h1) * h)), 3), np.uint8)
            img = cv2.hconcat([img_rs, mask])
            trans_x = int(w / 2) - int(int(float(w1 / h1) * h) / 2)
            trans_y = 0
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
        else:
            img_rs = cv2.resize(img, (w, int(float(h1 / w1) * w)))
            mask = np.zeros((h - int(float(h1 / w1) * w), w, 3), np.uint8)
            img = cv2.vconcat([img_rs, mask])
            trans_x = 0
            trans_y = int(h / 2) - int(int(float(h1 / w1) * w) / 2)
            trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
            height, width = img.shape[:2]
            img = cv2.warpAffine(img, trans_m, (width, height))
            return img
    def load_model(self,path, train = False):
        # print(self.device)
        model = attempt_load(path, map_location=self.device)  # load FP32 model
        names = model.module.names if hasattr(model, 'module') else model.names  # get class names
        if train:
            model.train()
        else:
            model.eval()
        return model, names
    def xyxytoxywh(self, x):
        # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[0] = (x[0] + x[2]) / 2  # x center
        y[1] = (x[1] + x[3]) / 2  # y center
        y[2] = x[2] - x[0]  # width
        y[3] = x[3] - x[1]  # height
        return y
    





def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--obj-weights', nargs='+', type=str, default='Character-Time-series-Matching/Vietnamese/object.pt', help='model path or triton URL')
    parser.add_argument('--char-weights', nargs='+', type=str, default='Character-Time-series-Matching/Vietnamese/char.pt', help='model path or triton URL')
    # parser.add_argument('--source', type=str, default='Vietnamese_imgs', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[256], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt


opt = parse_opt()
# imgsz = (640,640)
# obj_weights = 'Character-Time-series-Matching/Vietnamese/object.pt'
# char_weights = 'Character-Time-series-Matching/Vietnamese/char.pt'
# device = 'cpu'
# iou_thres = 0.5
# conf_thres = 0.25

    
obj_model=Detection(size=opt.imgsz,weights_path=opt.obj_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
char_model=Detection(size=opt.imgsz,weights_path=opt.char_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
# path=opt.source
char_model.size=(256,256)



sys.path.insert(0, "face-recognition/yolov5_face/")
# sys.path.append(os.path.abspath('face-recognition'))
from models.experimental import attempt_load_face
from utils.datasets import letterbox
from utils.general import non_max_suppression_face, check_img_size, scale_coords
sys.path.insert(0, 'face-recognition')

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get model detect
## Case 1:
# model = attempt_load("yolov5_face/yolov5s-face.pt", map_location=device)

## Case 2:
model = attempt_load_face("face-recognition/yolov5_face/yolov5n-0.5.pt", map_location=device)

# Get model recognition
## Case 1: 

from insightface.insight_face import iresnet100
weight = torch.load("face-recognition/insightface/resnet100_backbone.pth", map_location = device)
model_emb = iresnet100()

model_emb.load_state_dict(weight)
model_emb.to(device)
model_emb.eval()

face_preprocess = transforms.Compose([
                                    transforms.ToTensor(), # input PIL => (3,56,56), /255.0
                                    transforms.Resize((112, 112)),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                                    ])



def resize_image(img0, img_size):
    h0, w0 = img0.shape[:2]  # orig hw
    r = img_size / max(h0, w0)  # resize image to img_size

    if r != 1:  # always resize down, only resize up if training with augmentation
        interp = cv2.INTER_AREA if r < 1  else cv2.INTER_LINEAR
        img0 = cv2.resize(img0, (int(w0 * r), int(h0 * r)), interpolation=interp)

    imgsz = check_img_size(img_size, s=model.stride.max())  # check img_size
    img = letterbox(img0, new_shape=imgsz)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1).copy()  # BGR to RGB, to 3x416x416

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    
    return img

def scale_coords_landmarks(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2, 4, 6, 8]] -= pad[0]  # x padding
    coords[:, [1, 3, 5, 7, 9]] -= pad[1]  # y padding
    coords[:, :10] /= gain
    #clip_coords(coords, img0_shape)
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    coords[:, 4].clamp_(0, img0_shape[1])  # x3
    coords[:, 5].clamp_(0, img0_shape[0])  # y3
    coords[:, 6].clamp_(0, img0_shape[1])  # x4
    coords[:, 7].clamp_(0, img0_shape[0])  # y4
    coords[:, 8].clamp_(0, img0_shape[1])  # x5
    coords[:, 9].clamp_(0, img0_shape[0])  # y5
    return coords

def get_face(input_image):
    # Parameters
    size_convert = 128
    conf_thres = 0.4
    iou_thres = 0.5
    
    # Resize image
    img = resize_image(input_image.copy(), size_convert)

    # Via yolov5-face
    with torch.no_grad():
        pred = model(img[None, :])[0]

    # Apply NMS
    det = non_max_suppression_face(pred, conf_thres, iou_thres)[0]
    bboxs = np.int32(scale_coords(img.shape[1:], det[:, :4], input_image.shape).round().cpu().numpy())
    
    landmarks = np.int32(scale_coords_landmarks(img.shape[1:], det[:, 5:15], input_image.shape).round().cpu().numpy())    
    
    return bboxs, landmarks

def get_feature(face_image, training = True): 
    # Convert to RGB
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing image BGR
    face_image = face_preprocess(face_image).to(device)
    
    # Via model to get feature
    with torch.no_grad():
        if training:
            emb_img_face = model_emb(face_image[None, :])[0].cpu().numpy()
        else:
            emb_img_face = model_emb(face_image[None, :]).cpu().numpy()
    
    # Convert to array
    images_emb = emb_img_face/np.linalg.norm(emb_img_face)
    return images_emb

def read_features(root_fearure_path = "face-recognition/static/feature/face_features.npz"):
    data = np.load(root_fearure_path, allow_pickle=True)
    images_name = data["arr1"]
    images_emb = data["arr2"]
    
    return images_name, images_emb

def recognition(face_image):
    global isThread, score, name
    
    # Get feature from face
    query_emb = (get_feature(face_image, training=False))
    
    # Read features
    images_names, images_embs = read_features()   

    scores = (query_emb @ images_embs.T)[0]

    id_min = np.argmax(scores)
    score = scores[id_min]
    name = images_names[id_min]
    isThread = True
    # print("successful")



global isThread, score, name
isThread = True
score = 0
name = null

# Open camera 
cap = cv2.VideoCapture(0)
start = time.time_ns()
frame_count = 0
fps = -1

# Create a named window (you can give it any name)
# cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("frame", 1280, 720)

# Save video
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)
# video = cv2.VideoWriter('face-recognition/static/results/alfpr-demo.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 6, size)

# Read until video is completed
while(True):
    # Capture frame-by-frame
    _, frame = cap.read()

    na = []
    
    # Get faces
    bboxs, landmarks = get_face(frame)
    h, w, c = frame.shape
    
    tl = 1 or round(0.002 * (h + w) / 2) + 1  # line/font thickness
    clors = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255)]
    
    # Get boxs
    for i in range(len(bboxs)):
        # Get location face
        x1, y1, x2, y2 = bboxs[i]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 146, 230), 2)
        
        # Landmarks
        for x in range(5):
            point_x = int(landmarks[i][2 * x])
            point_y = int(landmarks[i][2 * x + 1])
            cv2.circle(frame, (point_x, point_y), tl+1, clors[x], -1)
        
        # Get face from location
        if isThread == True:
            isThread = False
            
            # Recognition
            face_image = frame[y1:y2, x1:x2]
            thread = Thread(target=recognition, args=(face_image,))
            thread.start()
    
        if name == null:
            continue
        else:
            if score < 0.25:
                caption= "UN_KNOWN"
            else:
                caption = f"{name.split('_')[0].upper()}:{score:.2f}"

            t_size = cv2.getTextSize(caption, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            na.append(caption)
            cv2.rectangle(frame, (x1, y1), (x1 + t_size[0], y1 + t_size[1]), (0, 146, 230), -1)
            cv2.putText(frame, caption, (x1, y1 + t_size[1]), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)       
        
    
    # object detection
    results, resized_img=obj_model.detect(frame.copy())
    cropped_image = None
    x1,y1,x2,y2 = 0,0,0,0
    plate_type = ""
    
    for names,conf,box in results:
        # if(names!='rectangle license plate' or names!='square license plate'):
        #     frame=cv2.putText(frame, "{}".format(names), (int(box[0]), int(box[1])-3),
        #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                         (255, 0, 255), 2)
        #     frame = cv2.rectangle(frame, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
        if(names=='rectangle license plate' or names=='square license plate'):
            # Crop the image using the ROI coordinates
            na.append(names)
            plate_type = names
            cropped_image = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
            # print(cropped_image.shape)


    lpr = ""
    # character detection
    if(cropped_image is not None and cropped_image.size > 0):
        results2, resized_img2=char_model.detect(cropped_image.copy())
        char_info = []
        for names,conf,box in results2:
            # lpr += names
            cropped_image=cv2.putText(cropped_image, "{}".format(names), (int(box[0]), int(box[1])-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 2)
            cropped_image = cv2.rectangle(cropped_image, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
            # print(names, box)
            char_info.append([names, box])
        # print(char_info)  
        # Sort the list based on x1 values
        if(plate_type == "square license plate" or plate_type == "rectangle license plate"):
            group1 = []
            group2 = []

            for i in range(len(char_info)):
                for j in range(i+1, len(char_info)):
                    item1 = char_info[i]
                    item2 = char_info[j]
                    y1_diff = abs(item1[1][1] - item2[1][1])
                    if y1_diff > 20.0:
                        if item1[1][1] < item2[1][1]:
                            group1.append(item1)
                            group2.append(item2)
                        else:
                            group1.append(item2)
                            group2.append(item1)
            
            # print(group1, group2)

            if(len(group1)!=0 and len(group2)!=0):
                group1 = list(set(map(tuple, group1)))
                group2 = list(set(map(tuple, group2)))

                ord_char_info = []
                for group in [group1, group2]:
                    g_ord = sorted(group, key=lambda item: item[1][0])
                    ord_char_info += g_ord

            else:
                ord_char_info = sorted(char_info, key=lambda item: item[1][0])
            

        else:
            ord_char_info = sorted(char_info, key=lambda item: item[1][0])

        # Extract the sorted names
        ord_char_names = [item[0] for item in ord_char_info]
        
        # print(ord_char_names)
        lpr += "".join(ord_char_names)
        if(len(lpr)>=10):
            lpr = lpr[:2].replace("1","T").replace("0","O").replace("2","Z").replace("3","E").replace("4","A").replace("5","S").replace("6","G").replace("7","T").replace("8","B").replace("9","P")+lpr[2:]
            lpr = lpr[:4]+lpr[4:6].replace("1","T").replace("0","O").replace("2","Z").replace("3","E").replace("4","A").replace("5","S").replace("6","G").replace("7","T").replace("8","B").replace("9","P")+lpr[6:]

        lpr = lpr.upper()
        if(len(cropped_image)>0 and x1!=0 and y1!=0 and x2!=0 and y2!=0):
            # pass
            # resized_img[y1:y2, x1:x2] = cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1])
            frame[y1:y2, x1:x2] = cropped_image
    
    if(len(lpr)>=9):
        na.append(lpr)
    if all(item != '' for item in na if isinstance(item, str)) and len(na)>2:
        print(na)

    # Count fps 
    frame_count += 1
    
    if frame_count >= 30:
        end = time.time_ns()
        fps = 1e9 * frame_count / (end - start)
        frame_count = 0
        start = time.time_ns()

    if fps > 0:
        fps_label = "FPS: %.2f" % fps
        cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    
    # video.write(frame)
    cv2.imshow("AFLPR system", frame)
    
    # Press Q on keyboard to  exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break  

# video.release()
cap.release()
cv2.destroyAllWindows()
cv2.waitKey(0)
