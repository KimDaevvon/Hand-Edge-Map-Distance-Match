import cv2
import numpy as np
import matplotlib.pyplot as plt

import files
import preproc

def GetHist(bgr_list):
    hists = []
    for row_idx, row in enumerate(bgr_list):
        for col_idx, bgr_img in enumerate(row):

            blur_img = cv2.GaussianBlur(bgr_img, (7,7), sigmaX = 1.0)
            ycrcb_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2YCrCb)
            Y, _, _ = cv2.split(ycrcb_img)
            
            t, mask = cv2.threshold(Y, 140, 255, cv2.THRESH_BINARY) 
            hist = cv2.calcHist(
                [ycrcb_img],
                channels=[1, 2],
                mask=mask,
                histSize=[256, 256],
                ranges=[0, 256, 0, 256]
            )
            cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            hists.append(hist)
    
    hist = np.maximum.reduce(hists)
    hist = cv2.pow(hist.astype(np.float32), 0.4)
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return hist


def FormData():
    
    files.RenameHands()
    print("Template images sorted")
    print("Loading template images...")
    bgr_list = files.PosesLoad()
    print("Template images loaded")
    
    print("Generating Histogram...")
    hist = GetHist(bgr_list)
    files.SaveHist(hist)
    print("Histogram generated")
    
    print("Generating Edge Distance Maps...")
    edge_list = [[] for _ in range(files.NUMBER_OF_POSE)]
    for row_idx, row in enumerate(bgr_list):
        for col_idx, img in enumerate(row):
            edge_img = preproc.GetSortedEdgeImg(img, hist)
            rotated_list = preproc.RotateEdgeImg(edge_img)
            plt.imshow(edge_img)
            plt.title(f"pose{row_idx+1}: {col_idx+1}st img")
            plt.show()
            edge_list[row_idx].extend(rotated_list)
    print("Edge Distance Maps generated")
    files.SaveDistanceMap(edge_list)
    return



