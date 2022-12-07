yolo_path = "yolov5"
from distutils.log import error
import sys
from tkinter import W, Image
from turtle import window_height
import cv2
import os
import numpy as np
sys.path.append(yolo_path)
from utils.augmentations import letterbox
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device, time_sync
from models.common import DetectMultiBackend
import torch
import torch.backends.cudnn as cudnn
from pynvml.smi import nvidia_smi
nvsmi = nvidia_smi.getInstance()
import timeit
from PIL import Image

def maxx(box):
    conf_list = []
    if len(box) > 1:
        for b in box:
            conf_list.append(b[5])
        index = np.argmax(conf_list)
        return [box[index]]
    else:
        return box

def convert_box(box, img_width, img_height, cls):
    x0 = int((box[0] - ((box[2]) / 2)*0.9) * img_width)
    y0 = int((box[1] - ((box[3]) / 2)*0.9) * img_height)
    x1 = int((box[0] + ((box[2]) / 2)*0.9) * img_width)
    y1 = int((box[1] + ((box[3]) / 2)*0.9) * img_height)
    if x0<0:
        x0 = 0
    if y0<0:
        y0 = 0
    return [x0, y0, x1, y1, cls]
def convert_box_no(box, img_width, img_height, cls, conf):
    x0 = int(box[0] * img_width)
    y1 = int(box[1] * img_height)
    w = int(box[2] * img_width)
    h = int(box[3] * img_height)
    conf = conf.cpu().data.numpy()
    return [x0, y1, w, h, cls, float(conf)]

@torch.no_grad()
def load_model(weights="",  # model.pt path(s)
        data='data/data.yaml',  # dataset.yaml path
        imgsz=[640, 640],  # inference size (height, width)
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    model.warmup(imgsz=(1 , 3, *imgsz))  # warmup
    # print("device",device)
    return model,device
@torch.no_grad()
def detect_box(model,
        device,
        source,  # file/dir/URL/glob, 0 for webcam
        imgsz=[736,736],  # inference size (height, width)
        conf_thres=0.3,  # confidence threshold
        iou_thres=0.7,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        ):
    
    # Load model
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    # dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    im0s = source
    img = letterbox(im0s, imgsz, stride=stride, auto=pt)[0]

    # Convert
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(img)

    im = torch.from_numpy(im).to(device)
    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im)
    im0s = source
    # NMS
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
    result=[]
    # Process predictions
    for i, det in enumerate(pred):  # per image
        im0= im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        box_image=[]
        box_image_no = []
        # print(det[:, :4])
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            box_image=[]
            for *xyxy, conf, cls in reversed(det):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh)  # label format
                line=(('%g ' * len(line)).rstrip() % line)
                line=line.split(" ")

                line= [float(value) if i!=0 else int(value) for i,value in enumerate(line)]
                cls=line[0]
                box=convert_box(line[1:],im0.shape[1],im0.shape[0], cls)
                box_no = convert_box_no(line[1:],im0.shape[1],im0.shape[0], cls, conf)
                # if box[0] > int(im0.shape[1]*0.02):
                    # if int(im0.shape[1]*0.1) < box[2] < int(im0.shape[1]*0.9):
                
                box_image.append(box)
                box_image_no.append(box_no)
        if len(box_image_no) > 4:
            box0 = []
            box1 = []
            box2 = []
            box3 = []
            for box in box_image_no:
                
                if box[4] == 0:
                    box0.append(box)
                if box[4] == 1:
                    box1.append(box)
                if box[4] == 2:
                    box2.append(box)
                if box[4] == 3:
                    box3.append(box)
            box0 = maxx(box0)
            box1 = maxx(box1)
            box2 = maxx(box2)
            box3 = maxx(box3)
            try:
                box_image_no = [box0[0], box1[0], box2[0], box3[0]]
            except:
                print(box0, box1, box2, box3)

    return box_image, box_image_no

def crop_box(img_ori, box_img,img_output, check_crop):
    img = img_ori
    img_orii = img_ori.copy()
    if len(box_img)!= 0:
        crop_list = []
        for i in range(len(box_img)):
            croped = img_ori[box_img[i][1]:box_img[i][3], box_img[i][0]: box_img[i][2]]
            crop_list.append(croped)
            img = cv2.rectangle(img_orii, (box_img[i][0],box_img[i][1]), (box_img[i][2],box_img[i][3]), (0,0,255), 2)
            cv2.putText(img, str(box_img[i][4]) + str(box_img[i]), (box_img[i][0],box_img[i][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 1)
            if check_crop == True:
                cv2.imwrite(img_output , croped)
        # if check_crop == False:
        #     cv2.imwrite(img_output, img)
    if check_crop == True:
        return crop_list
    else:
        return img

def getMemoryUsage():
    usage = nvsmi.DeviceQuery("memory.used")["gpu"][0]["fb_memory_usage"]
    return "%d %s" % (usage["used"], usage["unit"])


def mer_box(boxs, w, h):
    box0 = []
    box1 = []
    box2=[]
    box3=[]

    for box in boxs:
        box0.append(box[0])
        box1.append(box[1])
        box2.append(box[2])
        box3.append(box[3])
    x0 = min(box0)
    y0 = min(box1)
    x1 = max(box2)
    y1 = max(box3)
    
    # w1 = x1 -x0
    # h1 = y1 - y0
    # if w1 > h1 :
    #   est = w1 -h1 
    #   y0 = max(0, y0 - est//2)
    #   y1 = min(h, y1 + est//2)
    # else:
    #   est = h1 -w1 
    #   x0 = max(0, x0 - est//2)
    #   x1 = min(w, x1 + est//2)
    bounding_box = (int(x0) , int (y0) , int(x1), int(y1))
    return bounding_box


if __name__ == "__main__":
    folder_img = "/home/anlab/Desktop/Songuyen/PIl_detection/data_bicycle/images/train_new_2811/"
    folder_output = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/CVNet/rerank_yolo/data20221128/train/"
    weight = "/media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/detect_bycicle/CP/best.pt"

    torch.cuda.set_per_process_memory_fraction(0.2, 0)

    model, device = load_model(weights=weight)
    print("GPU Memory_____: %s" % getMemoryUsage())
    count = 0
    tt = 0
    err = 0
    with open('/home/anlab/Desktop/Songuyen/PIl_detection/data_bicycle/images/train_new_2811/paths.txt','r') as f:
        IMAGE_PATH_DB = [line.strip('\n') for line in f.readlines()]


    # for pa in os.listdir(folder_img):

    #     mk_path = folder_output + pa
    #     isExit = os.path.exists(mk_path)
    #     if not isExit:
    #         os.mkdir(mk_path)


    for path in IMAGE_PATH_DB:

        img_ori = cv2.imread(folder_img + path)
        # center = img_ori.shapes
        tt+=1
        start = timeit.default_timer()
        box_img, box_image_no = detect_box(model, device, img_ori,imgsz=[768,768],conf_thres=0.2, iou_thres = 0.3)
        # print("box_image_no", box_image_no)
        img_output = folder_output + path
        stop = timeit.default_timer()
        print('Time: ', stop - start)  

        img_out = crop_box(img_ori, box_img, img_output, check_crop = False)  #check == True --> croped_list, check==False ---> img_rectangle

        if len(box_image_no) == 4:
            # print(box_image_no)
            
            for box in box_image_no:
                if box[4] == 0:
                    box0 = box
                if box[4] == 1:
                    box1 = box
                if box[4] == 2:
                    box2 = box
                if box[4] == 3:
                    box3 = box

            if box1[0] < box0[0]:
                img = cv2.flip(img_ori, 1)
            else:
                img = img_ori
            box0 = []
            box1 = []
            box2=[]
            box3=[]
            for box in box_image_no:
                box0.append(box[0])
                box1.append(box[1])
                box2.append(box[2])
                box3.append(box[3])
            x0 = min(box0)
            y0 = min(box1)
            x1 = max(box0)
            y1 = max(box1)
            padx = int((x1 - x0)*0.03)
            pady = int((y1-y0)*0.3)
            if pady > y0*5:
                pady = y0 * 5
            # print(path)
            # print(img.shape)
            if padx > x0:
                padx = x0
            img = img[int(y0 - int(pady/5)):int(y1 + pady), int(x0 -padx):int(x1 + padx)]
            # print(img.shape)
            # print(int(y0 - int(pady/5)), int(y1 + pady))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img = Image.fromarray(img)
            img = img.convert('RGB')
            # cv2.imwrite(img_output, img)
            img.save(img_output)

        if len(box_image_no) != 4:
            center = img_ori.shape
            w =  center[1] * 0.7
            h =  center[0] * 0.7
            x = center[1]/2 - w/2
            y = center[0]/2 - h/2
            img = img_ori[int(y):int(y+h), int(x):int(x+w)]
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # cv2.imwrite(img_output, img)
            img = Image.fromarray(img)
            img = img.convert('RGB')
            # cv2.imwrite(img_output, img)
            img.save(img_output)

        #     print("box_image_no", box_image_no)

        #     print(len(box_img))
        #     err +=1
# print(err, tt)
