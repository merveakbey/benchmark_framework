from collections import defaultdict

from benchmark.evaluators.base_evaluator import BaseEvaluator


class SimpleDetectionEvaluator(BaseEvaluator):
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        self.total_tp = 0
        self.total_fp = 0
        self.total_fn = 0

        self.class_stats = defaultdict(lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_predictions": 0,
            "num_ground_truths": 0,
        })

    def add_sample(self, predictions, ground_truths) -> None:
        preds_by_class = defaultdict(list)
        gts_by_class = defaultdict(list)

        for pred in predictions:
            preds_by_class[pred.class_id].append(pred)

        for gt in ground_truths:
            # ignore=1 olanları şimdilik GT sayımına dahil etmeyelim
            if getattr(gt, "ignore", 0) == 1:
                continue
            gts_by_class[gt.class_id].append(gt)

        all_class_ids = set(preds_by_class.keys()) | set(gts_by_class.keys())

        for class_id in all_class_ids:
            preds = preds_by_class[class_id]
            gts = gts_by_class[class_id]

            preds = sorted(preds, key=lambda x: x.score, reverse=True)

            matched_gt_indices = set()

            class_name = None
            if preds:
                class_name = preds[0].class_name
            elif gts:
                class_name = gts[0].class_name
            else:
                class_name = str(class_id)

            self.class_stats[class_name]["num_predictions"] += len(preds)
            self.class_stats[class_name]["num_ground_truths"] += len(gts)

            for pred in preds:
                best_iou = 0.0
                best_gt_idx = -1

                for gt_idx, gt in enumerate(gts):
                    if gt_idx in matched_gt_indices:
                        continue

                    iou = self._compute_iou(pred.bbox_xyxy, gt.bbox_xyxy)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= self.iou_threshold and best_gt_idx >= 0:
                    matched_gt_indices.add(best_gt_idx)
                    self.total_tp += 1
                    self.class_stats[class_name]["tp"] += 1
                else:
                    self.total_fp += 1
                    self.class_stats[class_name]["fp"] += 1

            fn_count = len(gts) - len(matched_gt_indices)
            self.total_fn += fn_count
            self.class_stats[class_name]["fn"] += fn_count

    def evaluate(self) -> dict:
        precision = self.total_tp / (self.total_tp + self.total_fp) if (self.total_tp + self.total_fp) > 0 else 0.0
        recall = self.total_tp / (self.total_tp + self.total_fn) if (self.total_tp + self.total_fn) > 0 else 0.0

        classwise = {}
        for class_name, stats in self.class_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            class_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            class_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            classwise[class_name] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": round(class_precision, 6),
                "recall": round(class_recall, 6),
                "num_predictions": stats["num_predictions"],
                "num_ground_truths": stats["num_ground_truths"],
            }

        return {
            "iou_threshold": self.iou_threshold,
            "tp": self.total_tp,
            "fp": self.total_fp,
            "fn": self.total_fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "classwise_metrics": classwise,
        }

    def _compute_iou(self, box_a, box_b) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

        union_area = area_a + area_b - inter_area
        if union_area <= 0:
            return 0.0

        return inter_area / union_area