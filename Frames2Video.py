import cv2
import numpy as np
import os
from os.path import isfile, join

def Frames2Video(pathIn,pathOut):

    fps = 30
    frame_array = []
    files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]  # for sorting the file names properly
    files.sort()
    for i in range(len(files)):
        filename = pathIn + files[i]
        # reading each files
        img = cv2.imread(filename)
        height=720
        width=1280
        size = (width, height)

        # inserting the frames into an image array
        frame_array.append(img)
        out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()