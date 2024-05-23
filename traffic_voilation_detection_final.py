import numpy as np
import os
# import systemcheck
import cv2
import torch
from mailsend import sendmail
from mailsend import sendmail1

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

import torchvision.ops.boxes as bops
from datetime import datetime 

import detect_plate_and_number


def get_model(weights,  # model.pt path(s)
        imgsz=640,  # inference size (pixels) 
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        ):

    
    ###################### Load Model for Detection  #########################################  
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # print("Names:", names)

    # Half
    half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
    if pt:
        model.model.half() if half else model.model.float()

    if pt and device.type != 'cpu':
        model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup

    ###################### Model ready for Detection  #########################################  
    model_yolo = (device,model,stride, names, pt, jit, imgsz, half )
    print("Model ready for Detection")
    return model_yolo


def detect(model_yolo, 
        source, 
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        conf_thres=0.30,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=100,  # maximum detections per image
        ):

    device,model,stride, names, pt, jit, imgsz, half = model_yolo
     # Dataloader
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)

    for path, im, raw, vid_cap, s in dataset:
    
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        # Inference
        pred = model(im)
       
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, max_det=max_det)

        coords = list()
        class_list = list()
        # Process predictions

        selected_list = ['vehicle_person', 'helmet', 'person', 'motorcycle']
        
        
        for i, det in enumerate(pred):  # per image
            
            save_path = os.getcwd()+"/"+source  # im.jpg
            annotator = Annotator(raw, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to raw size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], raw.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class

                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    if names[c] in selected_list:
                        annotator.box_label(xyxy, label, color=colors(c, True))
                        xyxy.append(conf)
                        coords.append(xyxy)
                        class_list.append(names[c])
          
                  
            annotated_image = annotator.result()
        
    return annotated_image, coords, class_list


                
video = 0
ls = []
if __name__ == '__main__':

    video_name ="video2.mp4" #"nine.mp4"
    rotate = [0,90,180, 270][0]
    
    if ".mp4" in video_name:
        cap = cv2.VideoCapture(video_name)

        if (cap.isOpened()== False): 
            print("Error opening video stream or file")
            
        else:
            folder_name = "temp"
            video = 1
            print("Processing Video")
            count = 0
            while(cap.isOpened()):
         
                count += 1
                ret, frame = cap.read()
                if ret == True:
                    if count % 10 == 0:
                        if rotate == 270:
                            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                        if rotate == 180:
                            frame = cv2.rotate(frame, cv2.ROTATE_180)
                        if rotate == 90:
                            frame = cv2.rotate(frame, cv2.ROTATE_90)
                        ls.append(frame)
                else: 
                    break

            cap.release()
            cv2.destroyAllWindows()

        print("Processing Video Done..")
    else:
        folder_name = "testimages2"

        ls = os.listdir(folder_name)
        ls.sort()

    details = list()
    all_ious = list()
    images_processed = 0

    model_custom = get_model(weights= "bestv5_helmet.pt")
    model_person = get_model(weights= "bestv5_person.pt")
    
    print("Total Images:", len(ls), "|" )

    for img in ls[:]:
        if video:
            cv2.imwrite("temp/live.jpg", img)
            img = "live.jpg"
        if ".jpg" in img or ".jpeg" in img or ".png" in img: 
            print("Processing:", img)
            images_processed += 1

            image, custom_coords, custom_classes = detect(model_custom, source = f"{folder_name}/{img}")
            image2, person_coords, person_classes = detect(model_person, source = f"{folder_name}/{img}")

            raw_image = cv2.imread(f"{folder_name}/{img}")

            final_class = [] # All the items detected by both the classes
            final_coords = [] # Coordinates of those items

            for cc in custom_classes:
                final_class.append(cc)
            for pc in person_classes:
                final_class.append(pc)

            for ctc in custom_coords:
                final_coords.append(ctc)
            for psc in person_coords:
                final_coords.append(psc)


            for i in range(len(final_class)):
                if final_class[i] in ["vehicle_person", "motorcycle"]:
                    # print("CLASS MAIN:", final_class[i])
                    repeated = 0
                    if final_class[i] == "motorcycle":
                        calculated_iou = 0
                        for k in range(len(final_class)):
                            # print("CLASS CHECK:", final_class[k])
                            if final_class[k] == "vehicle_person":
                                box1 = torch.tensor([[final_coords[i][0], final_coords[i][1], final_coords[i][2], final_coords[i][3]]], dtype=torch.float)
                                box2 = torch.tensor([[final_coords[k][0], final_coords[k][1], final_coords[k][2], final_coords[k][3]]], dtype=torch.float)
                                iou = float(bops.box_iou(box2, box1)[0][0])
                                if iou > calculated_iou:
                                    calculated_iou = iou

                                # print("Bike IOU", calculated_iou)
                                if calculated_iou > 0.45:
                                    repeated = 1
                                    break
                    if repeated:  
                        # print("Repeated... COntinuining")          
                        continue

                    # print("Detected: ", final_class[i])
                    # print(final_coords[i])

                    width = int(final_coords[i][2] - final_coords[i][0])
                    height = int(final_coords[i][3] - final_coords[i][1])
                    print("Bike Dimensions: ", height, width, height/width)

                    if (height/width) > 2.35:
                        print("Partial Bike Detected ")
                        continue


                    center_coord = ((int(final_coords[i][0]) + int(width//2)) , (int(final_coords[i][1]) + int(height//2)))
                    
                        
                    area = height*width
                    # print("Center Coord:", center_coord)
                    # print("Area:", area)

                    if final_class[i] == "motorcycle":
                        axesLength = (int(width/1.5), int(height))
                        final_coords[i][1] = final_coords[i][1] - (height//2)
                        if final_coords[i][1] < 0:
                            final_coords[i][1] = 0
                    else:
                        axesLength = (int(width/1.5), int(height/1.5))
                    

                    box1 = torch.tensor([[final_coords[i][0], final_coords[i][1], final_coords[i][2], final_coords[i][3]]], dtype=torch.float)
                    # print("Confidence:", final_coords[i][4])

                    voilation = 0

                    
                  
                    calculated_iou = 0
                    calculated_person = 0
                    for k in range(len(final_class)):
                        calculated_iou = 0
                        # print("CLASS:", final_class[k])
                        if final_class[k] == "person":
                            box2 = torch.tensor([[final_coords[k][0], final_coords[k][1], final_coords[k][2], final_coords[k][3]]], dtype=torch.float)

                            iou = float(bops.box_iou(box2, box1)[0][0])

                            widthp = int(final_coords[k][2] - final_coords[k][0])
                            heightp = int(final_coords[k][3] - final_coords[k][1])
                            areap = widthp * heightp

                            iou = (area/areap) * iou

                            if iou > calculated_iou:
                                calculated_iou = iou

                            # print("Person IOU", calculated_iou)


                        if final_class[i] == "motorcycle":
                            iou_thres = 0.35
                        else:
                            iou_thres = 0.6

                        if calculated_iou > iou_thres:
                            calculated_person += 1
                            
                    # print("Persons Detected:", calculated_person)



                    if area < 6000:
                        print("Too Far to check for Helmet")
                        
                    else: 
                        if calculated_person > 0:
                            calculated_iou = 0
                            for j in range(len(final_class)):
                                if final_class[j] == "helmet":
                                    print("Helmet Detected")
                                    box2 = torch.tensor([[final_coords[j][0], final_coords[j][1], final_coords[j][2], final_coords[j][3]]], dtype=torch.float)
                                    iou = float(bops.box_iou(box1, box2)[0][0])
                                    widthh = int(final_coords[j][2] - final_coords[j][0])
                                    heighth = int(final_coords[j][3] - final_coords[j][1])
                                    areah = widthh * heighth

                                    iou = (area/areah) * iou
                                    if iou > calculated_iou:
                                        calculated_iou = iou

                                    # print("HELMET IOU:", calculated_iou)

                            if calculated_iou < 0.2:
                                print("!!! VOILATION !!! NO HELMET FOUND FOR INDEX:", i)
                                voilation = 1

                                color = (0, 0, 200)

                                image = cv2.ellipse(image, center_coord, axesLength, 0, 0, 360, color, 3)
                                font = cv2.FONT_HERSHEY_SIMPLEX
                                image = cv2.putText(image, 'HELMET VOILATION',(10,20), font, 0.6, color, 2, cv2.LINE_AA)


                                fact = 1
                                c1 = int(int(final_coords[i][1]))
                                c2 = int(int(final_coords[i][3]))

                                c3 = int(int(final_coords[i][0])*fact)
                                c4 = int(int(final_coords[i][2])//fact)
                                cropped_image = raw_image[c1:c2, c3:c4]

                                # cv2.imshow("Cropped Image", cropped_image)
                                # cv2.waitKey(1000)
                                cv2.imwrite("./temp/voilation_vehicle.jpg", cropped_image)
                                numplate_list = detect_plate_and_number.detect_plate("./temp/voilation_vehicle.jpg")
                                if len(numplate_list) > 0:
                                    print("Detected Numplates:", numplate_list)
                                else:
                                    print("No Number Detected")
                                    numplate_list.append("-")
                                
                                dt = str(datetime.now())[:-7].replace(":","-")
                                cv2.imwrite(f"./voilations/voilation_vehicle{dt}.jpg", cropped_image)

                                #Send Voilation Image through Mail
                                # sendmail(f"vemem30905@agaseo.com")
                                if len(numplate_list) > 0:
                                    sendmail(f"./voilations/voilation_vehicle{dt}.jpg",numplate_list)
                                else:
                                    sendmail1(f"./voilations/voilation_vehicle{dt}.jpg")
                                # sendmail(f"./voilations/voilation_vehicle{dt}.jpg")

                                
                                if "voilations.csv" in os.listdir():
                                    f1 = open("voilations.csv", 'a')
                                    print("File opened in Append Mode")
                                else:
                                    f1 = open("voilations.csv", 'w')
                                    print("File opened in Write Mode")
                                    f1.write("Date Time, Vehicle Type, Voilation Type, Vehicle Number, Vehicle Image, Fine\n")
                                
                                numplate = numplate_list[0].replace("\n","").replace(",","")
                                f1.write(f"{dt},Bike,No Helmet, {numplate},/voilations/voilation_vehicle{dt}.jpg,500 \n")
                                f1.close()
     
                        else:
                            print("")
                            # print("No Person Detected on Bike")
                

            for p in range(len(final_class)):
                if final_class[p] == "person":
                    color = (255, 0, 0) #BLUE
                    start_point = (int(final_coords[p][0]), int(final_coords[p][1]))
                    end_point = (int(final_coords[p][2]), int(final_coords[p][3]))
                    image = cv2.rectangle(image, start_point, end_point, color, 2)

            
            cv2.imshow("Custom YOLO Output", cv2.resize(image,(image.shape[1]//2, image.shape[0]//2)))
            # cv2.imshow("YOLO Output", image2)
            print("################################")
            cv2.waitKey(3000)
