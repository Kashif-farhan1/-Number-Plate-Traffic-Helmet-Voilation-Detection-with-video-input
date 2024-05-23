
#pip3 install torchvision pandas pyyaml tqdm seaborn


import numpy as np
import os
# import systemcheck
import cv2
import torch

from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.plots import Annotator, colors
from utils.torch_utils import select_device

from PIL import Image
import pytesseract
import os
import re

from tqdm import tqdm

import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import matplotlib.pyplot as plt
import keras_ocr
pipeline = keras_ocr.pipeline.Pipeline()



# construct the argument parse and parse the arguments

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result



TRAINED_YOLO = "best_number_plate.pt" #Pretraine Model Filename



model_image_size = 640 #Image Size with which Model was Trained
model,imgsz,device = None, None, None

def yolo_detect(source, conf_thres=0.30, iou_thres=0.45, lite = False):
    global model,imgsz,device, model_image_size

    if model is None:# If model is not Loaded
        ###################### Load Model for Detection  ############################# 
        device = select_device(" ")
        model = DetectMultiBackend(TRAINED_YOLO, device=device)
        imgsz = check_img_size(model_image_size, s=model.stride)  # check image size

        # print("Names:", model.names,)
        print("Device:", device.type)

        if model.pt and device.type != 'cpu' :
            print("Activated Half Precision for GPU")
            model.model.half()# half precision only supported by PyTorch on CUDA
            model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))# warmup     
        else:
            model.model.float()

    loaded_image = LoadImages(source, img_size=imgsz, stride=model.stride)

    for path, im, raw, vid_cap, _ in loaded_image:
        im = torch.from_numpy(im).to(device)
        if device.type != 'cpu':
            im = im.half()
        else:
            im = im.float()  # uint8 to float
        
        im /= 255  # 0.0 - 255.0 to 0.0 - 1.0 #Normalise Image
        if len(im.shape) == 3:
            im = im[None] # [R][G][B] -> [[R][G][B]]
       
        model_preds = model(im) # Predictions from Model
        final_preds = non_max_suppression(model_preds, conf_thres, iou_thres,max_det=100)

        coords = list()
        class_list = list()

        # Process predictions
        for detection in final_preds:  # per image
            annotator = Annotator(raw, line_width=3, example=str(model.names))
            if len(detection):
                detection[:, :4] = scale_coords(im.shape[2:], detection[:, :4], raw.shape).round()# Rescale boxesto raw size
              
                for *xyxy, conf, cls in reversed(detection):
                    c = int(cls)  # integer class
                    label = f'{model.names[c]} {conf:.2f}' #f'{names[c]}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
                    xyxy.append(conf)
                    coords.append(xyxy)
                    class_list.append(model.names[c])
            annotated_image = annotator.result()

    if lite: #Only return Annotated Output
        return annotated_image
    
    #Return Annoated with Prediction Details
    return annotated_image, coords, class_list


def detect_plate(imgpath):
    print("Processing Number Plate:", imgpath)

    numplate_list = []

    image,cd,cl = yolo_detect(source = imgpath) # Check and detect coord of number plate
    # image = yolo_detect(source = imgpath, lite = True)

    cv2.imshow("YOLO Output NumPlate", image)
    # print(image.shape)

    for coords in cd:
        # Crop only number plate from complete image
        c1 = int(coords[1])
        c2 = int(coords[3])

        c3 = int(coords[0])#+120
        c4 = int(coords[2])

        cropped_image = image[c1:c2, c3:c4]
        #Cropped Image of Number Plate

        cropped_image_area = cropped_image.shape[0] * cropped_image.shape[1]
        cropped_image_ratio = cropped_image.shape[0] / cropped_image.shape[1]
        # print("cropped_image_ratio: ", cropped_image_ratio)

        square = 0
        if cropped_image_ratio > 0.5:
            print("Square Number Plate")
            square = 1
            # print(cropped_image.shape)
            # cropped_image = cropped_image[140:,:]
            # print(cropped_image.shape)


        # factor = (50000/cropped_image_area) + 1;
        # if factor < 1:
        #     factor = 1

        # # print(cropped_image_area, factor)
        # cropped_image = cv2.resize(cropped_image, (int(cropped_image.shape[1]*factor), int(cropped_image.shape[0]*factor)))

        # print("CI:", cropped_image.shape)

        # cv2.waitKey(3000)
        
        # print(cd)
        # cv2.imshow("Cropped Output", cropped_image)
        # cv2.waitKey(7000)
        
        
        impath = "./temp/temp.jpg"
        cv2.imwrite(impath, cropped_image)

        preprocess = "thresh"

        # load the example image and convert it to grayscale
        tempimg = cv2.imread(impath)
        gray = cv2.cvtColor(tempimg, cv2.COLOR_BGR2GRAY)

        
        
        # gray = cv2.medianBlur(gray, 3)
        gray = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        cv2.imshow("Gray", gray)
        # cv2.waitKey(5000)
        

        contours, hier = cv2.findContours(gray,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        area_list = []
        ind_list = []
        for cnt in contours:
            area_list.append(cv2.contourArea(cnt))

        last_max = 1
        for i in range(5):
            maxval = max(area_list)

            max_ratio = last_max/maxval
            # print("Max Ratio:", max_ratio)
            if max_ratio > 3:
                # print("B1")
                break
            total_ratio = (gray.shape[0] * gray.shape[1])/maxval
            # print("Total ratio:", total_ratio)
            if total_ratio > 3:
                # print("B2")
                break

            ind = area_list.index(maxval)
            area_list[ind] = 0
            ind_list.append(ind)
            last_max = maxval

        # print("IND list:", ind_list)
        if len(ind_list) == 0:
            print("No Contour Found.. Skipping")
            continue
        final_contours = []
        for i in range(len(ind_list)):
            final_contours.append(contours[ind_list[i]])
        
        cnt = final_contours[-1]
        # print("Area", cv2.contourArea(cnt))
        # print("Drawing COntours")
        stencil = np.zeros(tempimg.shape).astype(tempimg.dtype)
        # stencil *= 255
        color = [255,255,255]
        cv2.fillPoly(stencil, [cnt], color)
        # tempimg = cv2.bitwise_and(tempimg, stencil)
        cv2.drawContours(tempimg,[cnt],0,(255,255,255),20)

        for row in tqdm(range(stencil.shape[0])):
            for col in range(stencil.shape[1]):
            
                if (int(stencil[row][col][0]) + int(stencil[row][col][1]) + int(stencil[row][col][2])) < 10:
                    gray[row][col] = 255


        # cv2.imshow("Contours", tempimg)
        # cv2.imshow("Stencil", stencil)
        # cv2.imshow("Gray", gray)
        # cv2.waitKey(5000)

        # write the grayscale image to disk as a temporary file so we can
        # apply OCR to it
        done = 0
        for i in range(5):
            for j in range(2):
                graytemp = gray.copy()
                angle = i*5
                if j == 1:
                    if angle == 0:
                        continue
                    angle = angle * -1


                # print("Angle:", angle)
                graytemp = rotate_image(graytemp, angle)
                
                cv2.imshow("Output", graytemp)
                cv2.waitKey(10)

                filename = "./temp/temp_binary.jpg"
                cv2.imwrite(filename, graytemp)
                
                if square:
                    images = [keras_ocr.tools.read(filename),]
                    # generate text predictions from the images
                    prediction_groups = pipeline.recognize(images)

                    predicted_image = prediction_groups[0]
                    text = ""
                    for txt, box in predicted_image:
                        text += txt
                    text.replace("\n", "")
                    text = re.sub(r'\W+', '', text)
                    # print("OP:", text)
                    

                else:
                    text = pytesseract.image_to_string(Image.open(filename))
                    text = re.sub(r'\W+', '', text)
                    # print("Text:", text)

                if square and (len(text) < 3):
                    print("Trying with Tesseract")
                    text = pytesseract.image_to_string(Image.open(filename))
                    text = re.sub(r'\W+', '', text)
                
                # if (not square) and (len(text) < 3):
                #     print("Trying with Keras")
                #     images = [keras_ocr.tools.read(filename),]
                #     # generate text predictions from the images
                #     prediction_groups = pipeline.recognize(images)

                #     predicted_image = prediction_groups[0]
                #     text = ""
                #     for txt, box in predicted_image:
                #         text += txt
                #     text.replace("\n", "")
                #     text = re.sub(r'\W+', '', text)


                if len(text) > 3:
                    print("OP:", text)
                    print(len(text))
                    numplate_list.append(text)
                    # print("OCR Done 1")
                    done = 1
                    break
                else:
                    print("Unable to recognise Text...")
                
            if done:
                print("OCR Done")
                return numplate_list
                break

    cv2.waitKey(2000)
    return numplate_list


testimage_folder = "testimages2"                

if __name__ == '__main__':

    ls = os.listdir(testimage_folder)
    ls.sort()

    

    for img in ls[:]:
        if ".jpg" in img or ".jpeg" in img or ".png" in img: 
            # print("Processing:", img)
            detect_plate(f"{testimage_folder}/{img}")
