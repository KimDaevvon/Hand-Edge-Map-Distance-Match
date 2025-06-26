import numpy as np
import cv2
import torch


NEGATIVE_PIX_VAL = 100
EXCEPT_THRESHOLD = 10

#음의 가중치를 가지는 픽셀들을 빈 공간에 뿌려 단순히 에지가 많은 레퍼런스에 대한 거짓 매칭 방지
def ScatterNegWeightPix(edge_img,   
                        x_range=(100, 400),
                        y_range=(100, 450),
                        grid_size=(5, 5),
                        num_regions=6,
                        neg_pix_val=NEGATIVE_PIX_VAL): 

    xmin, xmax = x_range
    ymin, ymax = y_range

    # ROI에서 0/1 맵 생성 후 적분 영상 계산
    roi = (edge_img[ymin:ymax, xmin:xmax] > 0).astype(np.uint8)
    ii = cv2.integral(roi)  # (h+1, w+1)

    rows, cols = grid_size
    H = ymax - ymin
    W = xmax - xmin
    cell_h = H // rows
    cell_w = W // cols

    # 각 셀별 에지 픽셀 개수 계산
    cells = []
    for i in range(rows):
        for j in range(cols):
            y0 = i * cell_h; y1 = y0 + cell_h
            x0 = j * cell_w; x1 = x0 + cell_w
            s = ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]
            cells.append((s, i, j))

    # 에지 개수 기준 오름차순 정렬
    cells.sort(key=lambda x: x[0])

    # 최소 에지 개수 값과 동일한 셀 목록
    min_s = cells[0][0]
    same = [(i, j) for s, i, j in cells if s == min_s]

    if len(same) >= num_regions:
        mid = len(same) // 2
        half = num_regions // 2
        # 중간을 중심으로 num_regions개를 슬라이스
        start = max(0, mid - half)
        selected = same[start:start + num_regions]
    else:
        # 동일 개수 영역이 부족하면 기존 로직대로 앞에서부터
        selected = [(i, j) for _, i, j in cells[:num_regions]]

    # 선택된 영역에 중앙 사각형 테두리로 NEGATIVE_PIX_VAL 설정
    for i, j in selected:
        x0 = xmin + j * cell_w
        y0 = ymin + i * cell_h
        sq_w = cell_w // 2
        sq_h = cell_h // 2
        rx0 = x0 + (cell_w - sq_w) // 2
        ry0 = y0 + (cell_h - sq_h) // 2
        rx1 = rx0 + sq_w
        ry1 = ry0 + sq_h
        cv2.rectangle(edge_img,
                      (rx0, ry0), (rx1-1, ry1-1),
                      color=neg_pix_val,
                      thickness=1)

    return edge_img


def GetBestEdgeDist(edge_tensor, edm_tensor,number_pose, num_map_per_pose, device):

    best_score = torch.tensor(10000, device=device)
    best_label = None

    H, W = edge_tensor.shape
    
    # edge/neg 인덱스 
    edge_pts = torch.nonzero(edge_tensor == 255, as_tuple=False)
    neg_pts  = torch.nonzero(edge_tensor == NEGATIVE_PIX_VAL, as_tuple=False)
    edge_idx = edge_pts[:,0] * W + edge_pts[:,1]           # (E,)
    neg_idx  = neg_pts[:,0]  * W + neg_pts[:,1]            # (N,)


    # 병렬 score 계산 (T 맵을 한꺼번에 처리)
    vals     = edm_tensor[:, edge_idx]                           # (T, E)
    neg_vals = edm_tensor[:, neg_idx]                            # (T, N)

    mean_vals = vals.mean(dim=1)                           # (T,)
    max_vals  = vals.max(dim=1)[0].unsqueeze(1)            # (T,1)
    max_neg   = neg_vals.max(dim=1)[0].clamp_min(1e-6).unsqueeze(1)  # (T,1)

    norm_neg  = (1 - neg_vals / max_neg) * max_vals        # (T, N)
    neg_score = norm_neg.mean(dim=1) * 0.3           # (T,)

    total     = mean_vals + neg_score                      # (T,)

    # 최솟값 위치에서 score, label 뽑기
    best_score, idx_min = total.min(dim=0)
    best_label = int((idx_min // num_map_per_pose).item()) + 1

    # 결과 출력 및 threshold 판정
    best_score_val = best_score.item()
    return best_label if best_score_val < EXCEPT_THRESHOLD else None
