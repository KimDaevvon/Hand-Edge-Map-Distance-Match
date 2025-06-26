import os
import glob
import cv2
import torch
import preproc, matching, files
import matplotlib.pyplot as plt
import time

def test():
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    hist = files.LoadHist() 
    flat_edm, num_pose, num_map_per_pose = files.LoadEDM(device)

    image_paths = sorted(glob.glob('./test/*.jpg'))
    durations = []  # 처리 시간 저장용 리스트

    for img_path in image_paths:
        bgr = cv2.imread(img_path)
        if bgr is None:
            print(f"[WARN] 이미지 로드 실패: {img_path}")
            continue

        start = time.perf_counter()

        edge = preproc.GetSortedEdgeImg(bgr, hist)
        if edge is not None:
            edge = matching.ScatterNegWeightPix(edge)
            edge_tensor = torch.from_numpy(edge).to(device, non_blocking=True)
            label = matching.GetBestEdgeDist(
                edge_tensor,
                flat_edm, num_pose, num_map_per_pose,
                device
            )
        else:
            label = None

        end = time.perf_counter()
        durations.append(end - start)  # 리스트에 추가

        # 결과 출력
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6,6))
        plt.imshow(rgb)
        plt.title(f"{os.path.basename(img_path)} → Predicted: {label}")
        plt.axis('off')
        plt.show()

    # 평균 계산 및 출력
    if durations:
        avg_time = sum(durations) / len(durations)
        print(f"\n총 {len(durations)}장 처리 시간 평균: {avg_time:.4f}초")

if __name__ == "__main__":
    test()
