"""학습된 모델로 영상에서 객체 감지 + 추적, 앞 5초만 mp4로 저장
궤적(trajectory)은 trajectory.csv로도 저장
"""
import sys
import csv
from collections import defaultdict

import cv2
import numpy as np
from ultralytics import YOLO

weights, src = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else "result_5s.mp4"
SECONDS = 5

model = YOLO(weights)
cap = cv2.VideoCapture(src)
fps = cap.get(cv2.CAP_PROP_FPS) or 30
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
max_frames = int(fps * SECONDS)

history = defaultdict(list)   # id -> 최근 중심점 목록
rows = []                     # frame, id, class, x, y, w, h
frame_idx = 0

while cap.isOpened() and frame_idx < max_frames:
    ok, frame = cap.read()
    if not ok:
        break
    r = model.track(frame, persist=True, tracker="bytetrack.yaml",
                    conf=0.3, verbose=False)[0]
    annotated = r.plot()

    if r.boxes is not None and r.boxes.id is not None:
        xywh = r.boxes.xywh.cpu().numpy()
        ids = r.boxes.id.int().cpu().tolist()
        clss = r.boxes.cls.int().cpu().tolist()
        for (x, y, bw, bh), tid, c in zip(xywh, ids, clss):
            history[tid].append((float(x), float(y)))
            history[tid] = history[tid][-60:]
            rows.append([frame_idx, tid, r.names[c], x, y, bw, bh])
        for tid in ids:                        # 현재 프레임에 있는 객체의 궤적만 그리기
            pts = history[tid]
            if len(pts) > 1:
                poly = np.array(pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(annotated, [poly], False, (0, 255, 255), 2)

    writer.write(annotated)
    frame_idx += 1

cap.release()
writer.release()

with open("trajectory.csv", "w", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["frame", "track_id", "class", "x", "y", "w", "h"])
    wr.writerows(rows)

print(f"저장 완료: {out} ({frame_idx}프레임, 추적 ID {len(history)}개)")
