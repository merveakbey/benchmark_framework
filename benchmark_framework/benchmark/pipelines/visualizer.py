import cv2


class DetectionVisualizer:
    def __init__(self, config: dict):
        self.config = config
        self.window_name = config.get("visualization", {}).get("window_name", "benchmark")

    def render(self, dataset_item, predictions):
        image = dataset_item.image.copy()

        for pred in predictions:
            x1 = int(pred.bbox[0])
            y1 = int(pred.bbox[1])
            x2 = int(pred.bbox[2])
            y2 = int(pred.bbox[3])

            class_name = getattr(pred, "class_name", str(pred.class_id))
            score = getattr(pred, "confidence", 0.0)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name} {score:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )

        return image

    def imshow(self, image):
        cv2.imshow(self.window_name, image)

    def wait_key(self, delay_ms: int):
        cv2.waitKey(delay_ms)

    def close(self):
        cv2.destroyAllWindows()