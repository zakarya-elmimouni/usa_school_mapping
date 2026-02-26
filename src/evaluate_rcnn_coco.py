import os
import json
import glob
from pathlib import Path
from PIL import Image
import torch
import numpy as np

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# =========================
# CONFIG
# =========================
DATA_ROOT = "dataset/usa/golden_data"
IMG_DIR = f"{DATA_ROOT}/images/test"
LBL_DIR = f"{DATA_ROOT}/labels/test"

MODEL_PATH = "results/finetuning_rcnn_best_params/best_model_global_1.pt"

OUTPUT_DIR = "results/finetuning_rcnn_best_params/official_coco_eval"
GT_JSON = f"{OUTPUT_DIR}/gt_coco.json"
PRED_JSON = f"{OUTPUT_DIR}/pred_coco.json"

NUM_CLASSES = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# STEP 1  YOLO then COCO GT
# =========================
def yolo_to_coco():

    images = []
    annotations = []
    ann_id = 1
    img_id = 1

    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))

    for img_path in img_paths:

        img = Image.open(img_path)
        width, height = img.size

        images.append({
            "id": img_id,
            "file_name": Path(img_path).name,
            "width": width,
            "height": height
        })

        lbl_path = os.path.join(LBL_DIR, Path(img_path).stem + ".txt")

        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                lines = f.readlines()

            for line in lines:
                cls, cx, cy, w, h = map(float, line.strip().split())

                x = (cx - w/2) * width
                y = (cy - h/2) * height
                box_w = w * width
                box_h = h * height

                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [x, y, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0
                })

                ann_id += 1

        img_id += 1

    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "school"}
        ]
    }

    with open(GT_JSON, "w") as f:
        json.dump(coco_format, f)

    print("GT COCO JSON generated.")


from torchvision.ops import box_iou

def compute_f1_recall50():

    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(PRED_JSON)

    TP = 0
    FP = 0
    FN = 0

    for img_id in coco_gt.getImgIds():

        gt_ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
        dt_ann_ids = coco_dt.getAnnIds(imgIds=[img_id])

        gt_anns = coco_gt.loadAnns(gt_ann_ids)
        dt_anns = coco_dt.loadAnns(dt_ann_ids)

        gt_boxes = []
        for ann in gt_anns:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x+w, y+h])

        dt_boxes = []
        dt_scores = []
        for ann in dt_anns:
            x, y, w, h = ann["bbox"]
            dt_boxes.append([x, y, x+w, y+h])
            dt_scores.append(ann["score"])

        if len(gt_boxes) == 0 and len(dt_boxes) == 0:
            continue

        if len(gt_boxes) == 0:
            FP += len(dt_boxes)
            continue

        if len(dt_boxes) == 0:
            FN += len(gt_boxes)
            continue

        gt_boxes = torch.tensor(gt_boxes)
        dt_boxes = torch.tensor(dt_boxes)
        dt_scores = torch.tensor(dt_scores)

        # Sort predictions by score (COCO style)
        sorted_idx = torch.argsort(dt_scores, descending=True)
        dt_boxes = dt_boxes[sorted_idx]

        ious = box_iou(dt_boxes, gt_boxes)

        matched_gt = set()

        for i in range(len(dt_boxes)):
            max_iou, idx = torch.max(ious[i], dim=0)

            if max_iou >= 0.5 and idx.item() not in matched_gt:
                TP += 1
                matched_gt.add(idx.item())
            else:
                FP += 1

        FN += len(gt_boxes) - len(matched_gt)

    precision50 = TP / (TP + FP + 1e-6)
    recall50 = TP / (TP + FN + 1e-6)
    f1_50 = 2 * precision50 * recall50 / (precision50 + recall50 + 1e-6)

    return precision50, recall50, f1_50

# =========================
# STEP 2  Model  COCO Predictions
# =========================
def load_model():

    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features, NUM_CLASSES + 1
    )

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    return model


def generate_predictions():

    model = load_model()
    predictions = []
    img_paths = sorted(glob.glob(os.path.join(IMG_DIR, "*.*")))

    img_id = 1

    with torch.no_grad():
        for img_path in img_paths:

            img = Image.open(img_path).convert("RGB")
            width, height = img.size

            img_tensor = torch.tensor(np.array(img)/255.).permute(2,0,1).float()
            img_tensor = img_tensor.to(DEVICE)

            output = model([img_tensor])[0]

            boxes = output["boxes"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):

                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1

                predictions.append({
                    "image_id": img_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "score": float(score)
                })

            img_id += 1

    with open(PRED_JSON, "w") as f:
        json.dump(predictions, f)

    print("Prediction COCO JSON generated.")


# =========================
# STEP 3  OFFICIAL COCO EVALUATION
# =========================
def coco_evaluation():

    coco_gt = COCO(GT_JSON)
    coco_dt = coco_gt.loadRes(PRED_JSON)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = coco_eval.stats

    # Compute recall50 & F1@0.5
    precision50, recall50, f1_50 = compute_f1_recall50()

    results_txt_path = os.path.join(OUTPUT_DIR, "coco_metrics.txt")

    with open(results_txt_path, "w") as f:

        f.write("===== OFFICIAL COCO EVALUATION RESULTS =====\n\n")

        f.write(f"mAP@[0.5:0.95]: {metrics[0]:.6f}\n")
        f.write(f"mAP@0.50: {metrics[1]:.6f}\n")
        f.write(f"mAP@0.75: {metrics[2]:.6f}\n\n")

        f.write(f"Precision@0.50: {precision50:.6f}\n")
        f.write(f"Recall@0.50: {recall50:.6f}\n")
        f.write(f"F1@0.50: {f1_50:.6f}\n\n")

        f.write(f"AP_small: {metrics[3]:.6f}\n")
        f.write(f"AP_medium: {metrics[4]:.6f}\n")
        f.write(f"AP_large: {metrics[5]:.6f}\n\n")

        f.write(f"AR@1: {metrics[6]:.6f}\n")
        f.write(f"AR@10: {metrics[7]:.6f}\n")
        f.write(f"AR@100: {metrics[8]:.6f}\n")

    print(f"\nCOCO metrics saved in {results_txt_path}")


# =========================
# RUN PIPELINE
# =========================
if __name__ == "__main__":

    print("Step 1: Converting GT to COCO...")
    yolo_to_coco()

    print("Step 2: Generating predictions...")
    generate_predictions()

    print("Step 3: Running official COCO evaluation...")
    coco_evaluation()