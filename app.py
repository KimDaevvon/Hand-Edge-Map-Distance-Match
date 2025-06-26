import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps, ImageSequence
import cv2
import numpy as np
import torch
import time
import gc

import formdata
import matching
import preproc
import files

REF_PATH = "./pose/"

class APP(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Vision UI")
        self.geometry("800x600")

        # 장치 및 모델/데이터 초기화
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hist = files.LoadHist()
        flat, P, M = files.LoadEDM(self.device)
        self.flat_edm_tensor = flat
        self.num_pose = P
        self.num_map_per_pose = M

        # 버튼 프레임
        btn_frame = tk.Frame(self)
        btn_frame.pack(side=tk.TOP, pady=10)
        tk.Button(btn_frame, text="Generate Data", command=self.generate_data).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Predict Image", command=self.predict_image).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Track GIF", command=self.track_gif).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Exit",                  command=self.exit_app).pack(side=tk.LEFT, padx=5)
        
        # 이미지 비교용 프레임
        self.img_frame = tk.Frame(self)
        self.img_frame.pack(expand=True, fill=tk.BOTH)

        # 입력 및 참조 라벨 생성만, 초기에는 숨김
        self.input_label = tk.Label(self.img_frame, compound=tk.TOP)
        self.input_label.pack(side=tk.LEFT, padx=10, pady=10)

        self.ref_label = tk.Label(self.img_frame, compound=tk.TOP)
        self.ref_label.pack(side=tk.LEFT, padx=10, pady=10)
        self.ref_label.pack_forget()

        # 내부 상태
        self.current_path = None
        self.current_ref = None
        self.label_timer = 0.0

    def _show_labels(self):
        if not self.input_label.winfo_ismapped():
            self.input_label.pack(side=tk.LEFT, padx=10, pady=10)
        if not self.ref_label.winfo_ismapped():
            self.ref_label.pack(side=tk.LEFT, padx=10, pady=10)

    def generate_data(self):
        formdata.FormData()
        print("Generate Data.")

    def predict_image(self):
        self._show_labels()
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if not path:
            return

        try:
            pil_img = Image.open(path)
            pil_img = ImageOps.exif_transpose(pil_img).convert('RGB')
        except Exception as e:
            messagebox.showerror("Error", f"이미지 로드 실패: {e}")
            return

        arr = np.array(pil_img)
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        edge = preproc.GetSortedEdgeImg(bgr, self.hist)
        edge = matching.ScatterNegWeightPix(edge)
        edge_tensor = torch.from_numpy(edge).to(self.device, non_blocking=True)
        label = matching.GetBestEdgeDist(
            edge_tensor, self.flat_edm_tensor,
            self.num_pose, self.num_map_per_pose, self.device
        )

        in_img = pil_img.copy()
        in_img.thumbnail((600, 500), Image.LANCZOS)
        self.in_photo = ImageTk.PhotoImage(in_img)
        self.input_label.config(
            image=self.in_photo,
            text=f"Input: {path.split('/')[-1]}\nPredicted Label: {label}"
        )
        self.input_label.image = self.in_photo

        if label is not None:
            ref_path = REF_PATH + f"pose{label}/hand1.jpg"
            try:
                ref_img = Image.open(ref_path)
                ref_img = ImageOps.exif_transpose(ref_img).convert('RGB')
                ref_img.thumbnail((600, 500), Image.LANCZOS)
                self.ref_photo = ImageTk.PhotoImage(ref_img)
                self.ref_label.config(image=self.ref_photo, text=f"Reference\n{ref_path}")
                self.ref_label.image = self.ref_photo
            except Exception as e:
                self.ref_label.config(text=f"Ref load error:\n{e}", image='')
        else:
            self.ref_label.pack_forget()

    def track_gif(self):
        self._show_labels()
        path = filedialog.askopenfilename(filetypes=[("GIF files", "*.gif")])
        if not path:
            return

        self.current_path = path
        self.current_ref = None
        self.gif = Image.open(path)
        self.gif_iter = iter(ImageSequence.Iterator(self.gif))
        self.label_timer = time.time()
        self._play_next_gif_frame()
        self.label_buffer = []  

    def _play_next_gif_frame(self):
        
        try:
            frame = next(self.gif_iter)
        except StopIteration:
            return

        # 이미지 디스플레이
        pil_fr = ImageOps.exif_transpose(frame).convert('RGB')
        disp = pil_fr.copy()
        disp.thumbnail((600, 500), Image.LANCZOS)
        photo = ImageTk.PhotoImage(disp)
        self.input_label.config(image=photo)
        self.input_label.image = photo

        # 20fps 간격으로만 예측 수행
        now = time.time()
        if now - self.label_timer >= 1/20:
            self.label_timer = now
            
            # 1) 새 프레임 예측
            arr    = np.array(pil_fr)
            bgr_fr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            edge   = preproc.GetSortedEdgeImg(bgr_fr, self.hist)
            edge   = matching.ScatterNegWeightPix(edge)
            new_label = matching.GetBestEdgeDist(
                torch.from_numpy(edge).to(self.device),
                self.flat_edm_tensor,
                self.num_pose, self.num_map_per_pose,
                self.device
            )
            
            # 2) 버퍼에 추가 (최대 길이 3 유지)
            self.label_buffer.append(new_label)
            if len(self.label_buffer) > 3:
                self.label_buffer.pop(0)

            # 3) 버퍼가 3개 차 있고, 마지막 3개가 모두 같으면 안정 레이블로 갱신
            if len(self.label_buffer) == 3 and \
               self.label_buffer[0] == self.label_buffer[1] == self.label_buffer[2]:
                stable = self.label_buffer[-1]
                if stable != self.current_ref:
                    self.current_ref = stable
                    # 참조 이미지 갱신
                    if stable is not None:
                        try:
                            ref = ImageOps.exif_transpose(
                                Image.open(REF_PATH + f"pose{stable}/hand1.jpg")
                            ).convert('RGB')
                            ref.thumbnail((600, 500), Image.LANCZOS)
                            rphoto = ImageTk.PhotoImage(ref)
                            self.ref_label.config(image=rphoto,
                                                  text=f"Reference\n{stable}")
                            self.ref_label.image = rphoto
                            if not self.ref_label.winfo_ismapped():
                                self.ref_label.pack(
                                    side=tk.LEFT, padx=10, pady=10)
                        except Exception:
                            self.ref_label.pack_forget()
                    else:
                        self.ref_label.pack_forget()

        # 4) 화면에는 항상 self.current_ref 만 표시
        disp_label = self.current_ref
        label_str = (f"Input: {self.current_path.split('/')[-1]}"
                     f"\nPredicted: {disp_label}")
        self.input_label.config(text=label_str)

        # 다음 프레임 예약
        dur = self.gif.info.get('duration', 50)
        self.after(dur, self._play_next_gif_frame)
        

    def track_realtime(self):
        messagebox.showinfo("Info", "RealTime Video tracking not implemented yet.")

    def exit_app(self):
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        try:
            self.input_label.destroy()
            self.ref_label.destroy()
        except:
            pass

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
        for attr in ("hist", "flat_edm_tensor", "current_path", "current_ref"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        self.destroy()

if __name__ == "__main__":
    app = APP()
    app.mainloop()
