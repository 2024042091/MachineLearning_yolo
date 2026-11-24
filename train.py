"""YOLOv8 커스텀 학습 (3개 부류). 사용: python train.py dataset/data.yaml"""
import sys
from ultralytics import YOLO

data = sys.argv[1] if len(sys.argv) > 1 else "dataset/data.yaml"

model = YOLO("yolov8n.pt")  # 사전학습 가중치 (전이학습)
model.train(data=data, epochs=60, imgsz=512, batch=8, project="runs", name="custom")

m = model.val()
print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}  "
      f"Precision={m.box.mp:.3f}  Recall={m.box.mr:.3f}")
print("가중치: runs/custom/weights/best.pt")
