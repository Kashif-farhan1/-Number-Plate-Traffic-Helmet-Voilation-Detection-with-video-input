import cv2
import os

video_name = 'nine.mp4'
cap = cv2.VideoCapture(video_name)

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

folder_name = video_name.split(".")[0]
try:
    os.mkdir(folder_name)
except FileExistsError:
    print("Folder Already Exist")

# Read until video is completed

print("Processing Video")
count = 0
while(cap.isOpened()):
  # Capture frame-by-frame
  count += 1
  ret, frame = cap.read()
  if ret == True:
 
    # cv2.imshow('Frame',frame)
    if count % 2 ==0:
        cv2.imwrite(f"{folder_name}/frame{count}.jpg", frame)
 
  # Break the loop
  else: 
    break
 
# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()

print("Processing Video Done..")