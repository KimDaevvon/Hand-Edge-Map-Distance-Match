import numpy as np
import cv2
import torch
from pathlib import Path
import re


NUMBER_OF_POSE = 7
EDM_DIR = "./data/"
EDM_FILE_NAME = "Edge_Distance_Map.pt"

def RenameHands():
    #각 파일 내에서 hand1.jpg, hand2.jpg, ... 로 변경
    for pose_idx in range(1, NUMBER_OF_POSE + 1):
        folder = Path(f'./pose/pose{pose_idx}')
        if not folder.is_dir():
            continue

        # 처리할 이미지 파일 목록(.jpg/.jpeg/.png)
        img_files = sorted([
            fp for fp in folder.iterdir()
            if fp.is_file() and fp.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])

        # 임시 이름으로 모두 변경(중복 방지)
        tmp_files = []
        for i, fp in enumerate(img_files):
            tmp = folder / f"__tmp_{i}{fp.suffix}"
            fp.rename(tmp)
            tmp_files.append(tmp)

        # 임시 파일들을 순서대로 hand{i}.jpg 로 변경
        for idx, tmp in enumerate(tmp_files, start=1):
            target = folder / f"hand{idx}.jpg"
            tmp.rename(target)



def natural_key(path: Path):
    m = re.search(r'(\d+)$', path.stem)
    return int(m.group(1)) if m else path.stem

def PosesLoad():
    img_list = [[] for _ in range(NUMBER_OF_POSE)]
    for folder in range(1, NUMBER_OF_POSE+1):
        p = Path(f'./pose/pose{folder}')
        if not p.is_dir(): 
            continue

        jpg_files = sorted(
            [fp for fp in p.iterdir() if fp.is_file() and fp.suffix.lower()=='.jpg'],
            key=natural_key
        )

        for fp in jpg_files:
            img = cv2.imread(str(fp), cv2.IMREAD_COLOR)  
            img_list[folder-1].append(img)

    return img_list


def SaveDistanceMap(edge_list):
    Path(EDM_DIR).mkdir(parents=True, exist_ok=True)

    all_maps = []
    for row in edge_list:
        map_list = []
        for img in row:
            dist_map = cv2.distanceTransform(255 - img, cv2.DIST_L2, 3)
            dist_map_norm = cv2.normalize(
                dist_map, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
            map_list.append(dist_map_norm)
        all_maps.append(map_list)

    P = len(all_maps)
    # 포즈별 최대 맵 개수
    max_M = max(len(m) for m in all_maps)
    H, W  = all_maps[0][0].shape

    # (P, max_M, H, W) 배열 생성
    np_tensor = np.full((P, max_M, H, W), 100, dtype=np.uint8)

    # 없는 슬롯은 100으로 패딩
    for i, maps in enumerate(all_maps):
        for j, m in enumerate(maps):
            np_tensor[i, j] = m

    t_tensor = torch.from_numpy(np_tensor)  # dtype=torch.uint8

    torch.save(t_tensor, Path(EDM_DIR) / EDM_FILE_NAME)
    print(f"Saved tensor shape={t_tensor.shape}")
        
def SaveHist(hist):
    np.save(EDM_DIR + "histogram.npy", hist)
    
def LoadHist():
    hist_path = Path(EDM_DIR + "histogram.npy")
    if not hist_path.exists():
        return None
    return np.load(hist_path)

def LoadEDM(device):
    tensor_path = Path(EDM_DIR + EDM_FILE_NAME) 
    if not tensor_path.exists():
        return None, None, None
    tensor = torch.load(tensor_path,map_location=device)
    tensor = tensor.float()
    P, M, H, W = tensor.shape
    flat = tensor.reshape(P * M, H * W) 
    
    return flat, P, M

