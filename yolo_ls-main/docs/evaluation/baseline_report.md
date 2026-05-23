# Baseline Evaluation Report

- Generated at: 2026-02-28 15:35:48
- Repo: https://github.com/Zhye26/yolo_ls.git

## 1) Detection Training Metrics
- Source: `/home/ubuntu/yolo_ls/runs/detect/experiments/yolo12n_20260126_144349/results.csv`
- Final epoch: 100
- Final Precision: 0.9192
- Final Recall: 0.8903
- Final mAP@50: 0.9360
- Final mAP@50-95: 0.6395
- Best mAP@50: 0.9371 (epoch 88)
- Best mAP@50-95: 0.6395 (epoch 100)

## 2) OCR Sample Evaluation
- Model: `/home/ubuntu/yolo_ls/models/plate_ocr.pt`
- Validation file: `/home/ubuntu/yolo_ls/datasets/cblprd/val.txt`
- Sample size: 150
- Valid images: 150
- Exact-match accuracy: 0.7933
- Prediction coverage: 0.8867
- Average confidence (predicted): 0.9902
- Eval time: 1.13s

## 3) Runtime Benchmark
- Model: `/home/ubuntu/yolo_ls/models/yolo12n_vehicle.pt`
- Device: cpu
- Frames: 100
- Avg latency: 72.10 ms
- P95 latency: 78.13 ms
- Throughput: 13.87 FPS
- Avg detections/frame: 6.59

## 4) Test Status
- Return code: 0
- Passed: 9
- Warnings: 10
- Duration: 58.68s
- Summary: 9 passed, 10 warnings in 53.00s

## 5) Conclusion
- The project has a runnable end-to-end baseline suitable for thesis demonstrations.
- Next step for stronger academic results: add controlled ablation and scenario-based error analysis.
