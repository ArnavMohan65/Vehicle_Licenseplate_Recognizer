import numpy as np
import torch
import time
import os
import sys
import argparse
sys.path.append(os.path.abspath('../yolov5'))
from utils.general import non_max_suppression, scale_coords
# from ai_core.object_detection.yolov5_custom.od.data.datasets import letterbox
from typing import List
# from dynaconf import settings
from models.experimental import attempt_load
import cv2

class Detection:
    def __init__(self, weights_path='.pt',size=(640,640),device='cpu',iou_thres=None,conf_thres=None):
        cwd = os.path.dirname(__file__)
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
            # det[:, :4]=scale_coords(resized_img.shape,det[:, :4],image.shape).round()
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
    parser.add_argument('--obj-weights', nargs='+', type=str, default='object.pt', help='model path or triton URL')
    parser.add_argument('--char-weights', nargs='+', type=str, default='char.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='Vietnamese_imgs', help='file/dir')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[1280], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.1, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

    return opt

def crop_with_argwhere(image):
    # Mask of non-black pixels (assuming image has a single channel).
    mask = image > 0
    
    # Coordinates of non-black pixels.
    coords = np.argwhere(mask)
    
    # Bounding box of non-black pixels.
    x0, y0 = coords.min(axis=0)
    x1, y1 = coords.max(axis=0) + 1   # slices are exclusive at the top
    
    # Get the contents of the bounding box.
    cropped = image[x0:x1, y0:y1]
    return x0,y0,x1,y1

def crop_rgb_with_argwhere(image):
    # Split the RGB image into its individual channels (R, G, B).
    r_channel, g_channel, b_channel = cv2.split(image)
    
    # Apply the cropping function to each channel.
    x0,y0,x1,y1 = crop_with_argwhere(r_channel)
    r_cropped = r_channel[x0:x1, y0:y1]
    g_cropped = g_channel[x0:x1, y0:y1]
    b_cropped = b_channel[x0:x1, y0:y1]

    
    # Merge the cropped channels back into an RGB image.
    cropped_rgb_image = cv2.merge((r_cropped, g_cropped, b_cropped))
    
    return cropped_rgb_image


# def process_frames(frame):
#     results, resized_img = obj_model.detect(frame)
#     # Display the results using OpenCV
#     for name, conf, box in results:
#         resized_img = cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
#                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                  (255, 0, 255), 2)
#         resized_img = cv2.rectangle(resized_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 1)

#     cv2.imshow('frame', resized_img)




# if __name__ == '__main__':
#     opt = parse_opt()
#     obj_model = Detection(size=opt.imgsz, weights_path=opt.obj_weights, device=opt.device, iou_thres=opt.iou_thres, conf_thres=opt.conf_thres)
#     path = opt.source

#     # Create a pool of worker processes
#     pool = multiprocessing.Pool(1)  # You can adjust the number of worker processes as needed

#     # Open the webcam
#     cap = cv2.VideoCapture(0)

#     count = 0
#     while True:
#         ret, frame = cap.read()
#         count += 1

#         # Process the frame in a separate process
#         pool.apply_async(process_frames, args=(frame,))

#         # Perform any other tasks on the frame or in the main thread

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Close the pool and release the webcam
#     pool.close()
#     pool.join()
#     cap.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     opt = parse_opt()
    
#     obj_model=Detection(size=opt.imgsz,weights_path=opt.obj_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
#     char_model=Detection(size=opt.imgsz,weights_path=opt.char_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
#     path=opt.source
#     char_model.size=(128,128)


#     # test this on webcam
#     cap = cv2.VideoCapture(0)
#     count = 0
#     fps = -1
#     start = time.time_ns()

#     while True:
#         ret, frame = cap.read()

#         # object detection
#         results, resized_img=obj_model.detect(frame)
#         for name,conf,box in results:
#             resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                                     (255, 0, 255), 2)
#             resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)

#         count += 1
#         if count >= 30:
#             end = time.time_ns()
#             fps = 1e9 * count / (end - start)
#             count = 0
#             start = time.time_ns()
    
#         if fps > 0:
#             fps_label = "FPS: %.2f" % fps
#             cv2.putText(resized_img, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1*(opt.imgsz[0]/640), (0, 0, 255), 2)

#         cv2.imshow('frame', resized_img)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()



if __name__ == '__main__':
    opt = parse_opt()
    
    obj_model=Detection(size=opt.imgsz,weights_path=opt.obj_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    char_model=Detection(size=opt.imgsz,weights_path=opt.char_weights,device=opt.device,iou_thres=opt.iou_thres,conf_thres=opt.conf_thres)
    path=opt.source
    char_model.size=(128,128)


    # test this on webcam
    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()

        # object detection
        results, resized_img=obj_model.detect(frame.copy())
        cropped_image = None
        x1,y1,x2,y2 = 0,0,0,0
        for name,conf,box in results:
            resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (255, 0, 255), 2)
            resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
            if(name=='rectangle license plate' or name=='square license plate'):
    #             # Crop the image using the ROI coordinates
                cropped_image = resized_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])


        # character detection
        if(cropped_image is not None and cropped_image.size > 0):
            results2, resized_img2=char_model.detect(cropped_image.copy())
            for name,conf,box in results2:
                resized_img2=cv2.putText(resized_img2, "{}".format(name), (int(box[0]), int(box[1])-3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (255, 0, 255), 2)
                resized_img2 = cv2.rectangle(resized_img2, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
                
            if(len(resized_img2)>0 and x1!=0 and y1!=0 and x2!=0 and y2!=0):
                # pass
                resized_img[y1:y2, x1:x2] = cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1])
        

        cv2.imshow('frame', resized_img)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()








    # while True:
    #     ret, frame = cap.read()
    #     # object detection
    #     results, resized_img=obj_model.detect(frame)
    #     # don't use resized_img to show, use frame
    #     if(len(results)!=0):
    #         if(len(results[0])!=0):
    #             # print(results[0])
    #             if(results[0][0]=='rectangle license plate'):
    #                 pass
    #                 # print(results[0])
        
    #     cropped_image = None
    #     x1,y1,x2,y2 = 0,0,0,0

    #     for name,conf,box in results:
    #         resized_img=cv2.putText(resized_img, "{}".format(name), (int(box[0]), int(box[1])-3),
    #                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                                 (255, 0, 255), 2)
    #         resized_img = cv2.rectangle(resized_img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
    #         if(name=='rectangle license plate' or name=='square license plate'):
    #             # Crop the image using the ROI coordinates
    #             cropped_image = resized_img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
    #             # results2, resized_img2=char_model.detect(cropped_image.copy())
    #             x1,y1,x2,y2 = int(box[0]),int(box[1]),int(box[2]),int(box[3])
    #             # # cv2.imwrite('cropped_license_plate.jpg', cropped_image)
    #             # # break

    #             # for name2,conf2,box2 in results2:
    #             #     resized_img2=cv2.putText(frame, "{}".format(name2), (int(box2[0]), int(box2[1])-3),
    #             #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #             #                             (255, 0, 255), 2)
    #             #     resized_img2 = cv2.rectangle(frame, (int(box2[0]),int(box2[1])), (int(box2[2]),int([3])), (0,0,255), 1)
                
    #             # if(len(resized_img2)>0 and x1!=0 and y1!=0 and x2!=0 and y2!=0):
    #             #     # pass
    #             #     resized_img[y1:y2, x1:x2] = cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1])
    #             #     # cv2.imwrite('frames/{:04d}.jpg'.format(count), resized_img[y1:y2, x1:x2])
            
    #     # cv2.imshow('frame', resized_img)

    #     # character detection
    #     if(cropped_image is not None and cropped_image.size > 0):
    #         results2, resized_img2=char_model.detect(cropped_image)
    #     # else:
    #         # results2, resized_img2=char_model.detect(frame.copy())
    #         # char_model.size=(128,128)
    #         # don't use resized_img to show, use frame
    #         if(len(results2)!=0):
    #             if(len(results2[0])!=0):
    #                 # print(results2[0][0])
    #                 pass

    #         for name,conf,box in results2:
    #             resized_img2=cv2.putText(resized_img2, "{}".format(name), (int(box[0]), int(box[1])-3),
    #                                     cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                                     (255, 0, 255), 2)
    #             resized_img2 = cv2.rectangle(resized_img2, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,0,255), 1)
                
    #         if(len(resized_img2)>0 and x1!=0 and y1!=0 and x2!=0 and y2!=0):
    #             # pass
    #             resized_img[y1:y2, x1:x2] = cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1])
    #             # cv2.imwrite('frames/{:04d}.jpg'.format(count), resized_img[y1:y2, x1:x2])
    #             # print(cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1]).shape)
    #             # cv2.imshow('frame', cv2.resize(crop_rgb_with_argwhere(resized_img2),cropped_image.shape[:2][::-1]))

    #     cv2.imshow('frame', resized_img)
    #     # save frame in frames folder 
    #     # save frame as jpg file
    #     # cv2.imwrite('frames/{:04d}.jpg'.format(count), resized_img)
    #     count += 1
    #     # cv2.imshow('frame', resized_img2)
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()

