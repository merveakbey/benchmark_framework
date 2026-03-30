from collections import defaultdict
from typing import Dict, List

from benchmark.evaluators.base_evaluator import BaseEvaluator


class COCODetectionEvaluator(BaseEvaluator):
    def __init__(self, iou_thresholds=None):
        if iou_thresholds is None:
            # COCO: 0.50:0.05:0.95
            iou_thresholds = [round(0.50 + i * 0.05, 2) for i in range(10)]

        self.iou_thresholds = iou_thresholds
        self.ground_truths = []
        self.predictions = []

    def add_sample(self, predictions, ground_truths) -> None:
        valid_gts = [gt for gt in ground_truths if getattr(gt, "ignore", 0) == 0]
        self.ground_truths.append(valid_gts)
        self.predictions.append(predictions)

    def evaluate(self) -> dict:
        class_ids = self._collect_class_ids()

        per_threshold_ap: Dict[float, Dict[int, float]] = {}
        classwise_ap_50 = {}
        classwise_ap_50_95 = {}

        for iou_thr in self.iou_thresholds:
            per_class_ap = {}
            for class_id in class_ids:
                ap = self._evaluate_class_at_iou(class_id, iou_thr)
                per_class_ap[class_id] = ap
            per_threshold_ap[iou_thr] = per_class_ap

        map_50 = 0.0
        if 0.5 in per_threshold_ap and class_ids:
            map_50 = sum(per_threshold_ap[0.5].values()) / len(class_ids)

        map_50_95 = 0.0
        if class_ids:
            all_ap_values = []
            for iou_thr in self.iou_thresholds:
                all_ap_values.extend(per_threshold_ap[iou_thr].values())
            if all_ap_values:
                map_50_95 = sum(all_ap_values) / len(all_ap_values)

        for class_id in class_ids:
            ap50 = per_threshold_ap.get(0.5, {}).get(class_id, 0.0)
            ap5095 = sum(per_threshold_ap[iou_thr].get(class_id, 0.0) for iou_thr in self.iou_thresholds) / len(self.iou_thresholds)

            class_name = self._resolve_class_name(class_id)
            classwise_ap_50[class_name] = round(ap50, 6)
            classwise_ap_50_95[class_name] = round(ap5095, 6)

        simple_stats = self._compute_simple_stats_at_iou(0.5)

        classwise_metrics = {}
        for class_id in class_ids:
            class_name = self._resolve_class_name(class_id)
            classwise_metrics[class_name] = {
                "ap50": classwise_ap_50[class_name],
                "ap50_95": classwise_ap_50_95[class_name],
                "tp": simple_stats["classwise"].get(class_name, {}).get("tp", 0),
                "fp": simple_stats["classwise"].get(class_name, {}).get("fp", 0),
                "fn": simple_stats["classwise"].get(class_name, {}).get("fn", 0),
                "precision": simple_stats["classwise"].get(class_name, {}).get("precision", 0.0),
                "recall": simple_stats["classwise"].get(class_name, {}).get("recall", 0.0),
                "num_predictions": simple_stats["classwise"].get(class_name, {}).get("num_predictions", 0),
                "num_ground_truths": simple_stats["classwise"].get(class_name, {}).get("num_ground_truths", 0),
            }

        return {
            "map_50": round(map_50, 6),
            "map_50_95": round(map_50_95, 6),
            "precision": round(simple_stats["precision"], 6),
            "recall": round(simple_stats["recall"], 6),
            "tp": simple_stats["tp"],
            "fp": simple_stats["fp"],
            "fn": simple_stats["fn"],
            "classwise_metrics": classwise_metrics,
            "iou_thresholds": self.iou_thresholds,
        }

    def _collect_class_ids(self):
        class_ids = set()

        for sample_preds in self.predictions:
            for pred in sample_preds:
                class_ids.add(pred.class_id)

        for sample_gts in self.ground_truths:
            for gt in sample_gts:
                class_ids.add(gt.class_id)

        return sorted(class_ids)

    def _resolve_class_name(self, class_id: int) -> str:
        for sample_preds in self.predictions:
            for pred in sample_preds:
                if pred.class_id == class_id:
                    return pred.class_name

        for sample_gts in self.ground_truths:
            for gt in sample_gts:
                if gt.class_id == class_id:
                    return gt.class_name

        return str(class_id)

    def _evaluate_class_at_iou(self, class_id: int, iou_threshold: float) -> float:
        predictions = []
        gt_count = 0

        for image_idx, (sample_preds, sample_gts) in enumerate(zip(self.predictions, self.ground_truths)):
            class_preds = [p for p in sample_preds if p.class_id == class_id]
            class_gts = [g for g in sample_gts if g.class_id == class_id]

            gt_count += len(class_gts)

            for pred in class_preds:
                predictions.append({
                    "image_idx": image_idx,
                    "score": float(pred.score),
                    "bbox": pred.bbox_xyxy,
                })

        if gt_count == 0:
            return 0.0

        predictions.sort(key=lambda x: x["score"], reverse=True)

        gt_by_image = {}
        for image_idx, sample_gts in enumerate(self.ground_truths):
            gt_by_image[image_idx] = [g for g in sample_gts if g.class_id == class_id]

        matched = {
            image_idx: [False] * len(gt_by_image[image_idx])
            for image_idx in gt_by_image
        }

        tp = []
        fp = []

        for pred in predictions:
            image_idx = pred["image_idx"]
            pred_box = pred["bbox"]
            gts = gt_by_image[image_idx]

            best_iou = 0.0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if matched[image_idx][gt_idx]:
                    continue
                iou = self._compute_iou(pred_box, gt.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                matched[image_idx][best_gt_idx] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if not tp:
            return 0.0

        tp_cum = []
        fp_cum = []
        running_tp = 0
        running_fp = 0

        for t, f in zip(tp, fp):
            running_tp += t
            running_fp += f
            tp_cum.append(running_tp)
            fp_cum.append(running_fp)

        recalls = []
        precisions = []

        for tpc, fpc in zip(tp_cum, fp_cum):
            recall = tpc / gt_count if gt_count > 0 else 0.0
            precision = tpc / (tpc + fpc) if (tpc + fpc) > 0 else 0.0
            recalls.append(recall)
            precisions.append(precision)

        ap = self._compute_ap_101_point(recalls, precisions)
        return ap

    def _compute_ap_101_point(self, recalls: List[float], precisions: List[float]) -> float:
        if not recalls or not precisions:
            return 0.0

        ap = 0.0
        for r in [i / 100 for i in range(101)]:
            p = 0.0
            for recall, precision in zip(recalls, precisions):
                if recall >= r:
                    p = max(p, precision)
            ap += p

        return ap / 101.0

    def _compute_simple_stats_at_iou(self, iou_threshold: float) -> dict:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        class_stats = defaultdict(lambda: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "num_predictions": 0,
            "num_ground_truths": 0,
            "precision": 0.0,
            "recall": 0.0,
        })

        for sample_preds, sample_gts in zip(self.predictions, self.ground_truths):
            preds_by_class = defaultdict(list)
            gts_by_class = defaultdict(list)

            for pred in sample_preds:
                preds_by_class[pred.class_id].append(pred)
            for gt in sample_gts:
                gts_by_class[gt.class_id].append(gt)

            all_class_ids = set(preds_by_class.keys()) | set(gts_by_class.keys())

            for class_id in all_class_ids:
                preds = sorted(preds_by_class[class_id], key=lambda x: x.score, reverse=True)
                gts = gts_by_class[class_id]

                class_name = self._resolve_class_name(class_id)
                class_stats[class_name]["num_predictions"] += len(preds)
                class_stats[class_name]["num_ground_truths"] += len(gts)

                matched_gt_indices = set()

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

                    if best_iou >= iou_threshold and best_gt_idx >= 0:
                        matched_gt_indices.add(best_gt_idx)
                        total_tp += 1
                        class_stats[class_name]["tp"] += 1
                    else:
                        total_fp += 1
                        class_stats[class_name]["fp"] += 1

                fn_count = len(gts) - len(matched_gt_indices)
                total_fn += fn_count
                class_stats[class_name]["fn"] += fn_count

        total_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        total_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0

        for class_name, stats in class_stats.items():
            tp = stats["tp"]
            fp = stats["fp"]
            fn = stats["fn"]

            stats["precision"] = round(tp / (tp + fp), 6) if (tp + fp) > 0 else 0.0
            stats["recall"] = round(tp / (tp + fn), 6) if (tp + fn) > 0 else 0.0

        return {
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": total_precision,
            "recall": total_recall,
            "classwise": class_stats,
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