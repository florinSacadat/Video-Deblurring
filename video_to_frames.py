import cv2
import os

print(cv2.__version__)
# for i in range(57,63):
#   vidcap = cv2.VideoCapture('/media/student/2.0 TB Hard Disk/Florin-Dataset/Video/240fps-HD/r_'+i.__str__()+'.MP4')
#   success,image = vidcap.read()
#   count = 0
#   success = True
#   while success:
#     cv2.imwrite("/media/student/2.0 TB Hard Disk/Florin-Dataset/Frames/Original/"+i.__str__()+"/%d.png" % count, image)     # save frame as JPEG file
#     success,image = vidcap.read()
#
#     count += 1
vidcap = cv2.VideoCapture('/home/student/Documents/VideoToFrames/r_o2.MP4')
success,image = vidcap.read()
count = 0
success = True
while success:
  cv2.imwrite("/home/student/Documents/VideoToFrames/O2/%s.png" % count.__str__().zfill(4), image)     # save frame as JPEG file
  success,image = vidcap.read()

  count += 1