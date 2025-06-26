import cv2
import numpy as np


BGR_BG_COLOR = [0, 0, 0]
EDGE_IMG_SIZE = 500
OBJECT_ORIGIN_X = 200
OBJECT_ORIGIN_Y = 450

def Resize(img):
    h, w = img.shape[:2]
    scale = 300.0 / max(h, w)

    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

def GetBackProjImg(bgr_img, hist):
    blur_img = cv2.GaussianBlur(bgr_img, (9,9), sigmaX = 1.4)
    ycrcb_img = cv2.cvtColor(blur_img, cv2.COLOR_BGR2YCrCb)
    
    Y = ycrcb_img[:,:, 0]
    t, mask = cv2.threshold(Y, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) #remove black
    
    backproj = cv2.calcBackProject(
        [ycrcb_img], channels=[1,2],
        hist=hist,
        ranges=[0,256,0,256],
        scale=1
    )
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    # fg_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    # fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,   kernel, iterations=1)
    cv2.bitwise_and(backproj, backproj, dst = backproj, mask=mask)
    return backproj


def RemoveBG(bgr_img, backproj_img):   
    _, bin_img = cv2.threshold(backproj_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    max_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(bin_img)
    cv2.drawContours(mask, [max_contour], -1, 255, thickness=-1)
    retimg = cv2.bitwise_and(bgr_img, bgr_img, mask=mask)
    
    return retimg
    
def RemoveSmallComp(edge_img, min_size=120):

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(edge_img, connectivity=8)

    cleaned_img = np.zeros_like(edge_img)
    
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_size:
            cleaned_img[labels == label] = 255

    return cleaned_img

def CutNRotate(bgrimg):
    ########## 캔버스 확장 (500×500 중앙 정렬) ##########
    h0, w0 = bgrimg.shape[:2]
    canvas = np.full((EDGE_IMG_SIZE, EDGE_IMG_SIZE, 3), BGR_BG_COLOR, dtype=bgrimg.dtype)
    dx0 = (EDGE_IMG_SIZE - w0) // 2
    dy0 = (EDGE_IMG_SIZE - h0) // 2
    canvas[dy0:dy0+h0, dx0:dx0+w0] = bgrimg
    bgrimg = canvas

    ##########외각 Contour 검출##########
    img = cv2.cvtColor(bgrimg, cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(largest_contour, True)
    largest_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    ##########RECT 검출 후 기울기를 이용해 이미지 및 컨투어 회전#############
    rect = cv2.minAreaRect(largest_contour.reshape(-1,2))
    h, w = bgrimg.shape[:2]               # 이제 h, w 모두 500
    rotate_degree = -(90 - rect[2]) if rect[1][0] > rect[1][1] else rect[2]
    center = (w//2, h//2)                 # 캔버스 중심 (250,250)
    rotation_matrix = cv2.getRotationMatrix2D(center, rotate_degree, 1)
    rotated_img = cv2.warpAffine(
        bgrimg, rotation_matrix, (EDGE_IMG_SIZE, EDGE_IMG_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BGR_BG_COLOR
    )
    rotated_contour = cv2.transform(largest_contour, rotation_matrix) 
    
    ##########회전한 컨투어로 손목 검출#############
    # 1) convexHull 으로 인덱스만 뽑기
    hull = cv2.convexHull(rotated_contour, returnPoints=False)  # shape=(M,1)
    hull_pts = cv2.convexHull(rotated_contour, returnPoints=True)
    # 2) 1차원 배열로 풀어서 오름차순 정렬
    hull_idxs = np.sort(hull.flatten())
    # 3) (M,1) 형태로 다시 변경
    hull_idxs = hull_idxs.reshape(-1,1).astype(np.int32)
    # 4) convexityDefects 호출
    defects = cv2.convexityDefects(rotated_contour, hull_idxs)
    
    fp_idx = max(defects[:,0,2], key=lambda i: rotated_contour[i][0][1])
    deep_point = tuple(rotated_contour[fp_idx][0])
    rotated_img[deep_point[1]:, : ,:] = BGR_BG_COLOR
    
    
    ###########RECT도 회전 시킨다음 객체를 비율에 맞게 객체의 왼쪽 하단 지점을 지정하여 정렬###########
    rect_points = cv2.boxPoints(rect)
    rotated_rect = cv2.transform(rect_points[:,None,:], rotation_matrix).reshape(-1,2)
    vertex = sorted(rotated_rect[:], key=lambda p: p[1])  # 윗부분 꼭지점, 아래부분 꼭지점으로 분리
    vertex[2][1] = deep_point[1]
    vertex[3][1] = deep_point[1]
    
    obj_origin = np.where((rotated_img[deep_point[1] - 1] != 0).any(axis=1))[0][0]  ####객체의 왼쪽 하단 지점
    dx1 = - obj_origin
    dy1 = - vertex[2][1]
    dx2 = OBJECT_ORIGIN_X
    dy2 = OBJECT_ORIGIN_Y
    scale = 300/(vertex[2][1] - vertex[0][1])
    dx = scale * dx1 + dx2
    dy = scale * dy1 + dy2
    
    M = np.array([[scale, 0, dx],
                  [0, scale, dy]], dtype=np.float32)

    # 최종 이동·스케일링도 캔버스 크기 고정
    shifted = cv2.warpAffine(
        rotated_img, M,
        (EDGE_IMG_SIZE, EDGE_IMG_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=BGR_BG_COLOR
    )

    return shifted

def RotateEdgeImg(edge_img, dseta = 3, seta_range = 10):
    edge_list = []
    h, w = edge_img.shape[:2]
    
    # -30° ~ 30° 회전
    for r in range(seta_range*2):
        angle = dseta * (r - seta_range)
        M = cv2.getRotationMatrix2D((OBJECT_ORIGIN_X, OBJECT_ORIGIN_Y), angle, 1.0)

        rotated = cv2.warpAffine(edge_img, M, (w, h),
                                 flags=cv2.INTER_NEAREST,
                                 borderValue=0)
        edge_list.append(rotated)
    return edge_list

def GetSortedEdgeImg(bgr_img, hist):  

        bgr_img = Resize(bgr_img)
        backproj = GetBackProjImg(bgr_img, hist)
        rm_bg_img= RemoveBG(bgr_img, backproj)
        if rm_bg_img is None:
            return None
        preproc_img = CutNRotate(rm_bg_img)

        canny_img = RemoveSmallComp(cv2.Canny(preproc_img, 100, 120))
        canny_img[448:,:] = 0
        return canny_img

