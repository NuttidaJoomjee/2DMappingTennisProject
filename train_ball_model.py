from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data=r"D:\TennisProject\ball_dataset\data.yaml",
    epochs=30,
    fraction=0.15,        # ~1000 of ~6650 train images/epoch (incl. the
                          # hard-negative crops+variants added for the
                          # stadium-light false positive) -- full-dataset
                          # epochs measured at ~26min/epoch on this CPU;
                          # this is a bounded middle ground (~1-1.5hr total)
    imgsz=320,
    batch=16,
    workers=4,
    device="cpu",
    val=False,            # skip per-epoch validation to save time; spot-check after
    patience=1000,
    project=r"D:\TennisProject\ball_training_runs",
    name="tennis_ball_yolov8n_v2",
    exist_ok=True,
)
