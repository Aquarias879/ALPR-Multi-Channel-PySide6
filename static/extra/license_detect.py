import time
import cv2
import numpy as np
import onnxruntime as model
import warnings

# Ignore specific UserWarning related to provider names
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names")

class ObjectDetector:
    def __init__(self, onnx_model_path, class_names, conf_thres, iou_thres, use_gpu=True):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = class_names
        self.colors = np.random.default_rng(3).uniform(0, 255, size=(len(class_names), 3))
        self.use_gpu = use_gpu

        self.initialize_model(onnx_model_path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, onnx_model_path):
        try:
            providers = ['DmlExecutionProvider'] if self.use_gpu else ['CPUExecutionProvider']
            self.session = model.InferenceSession(onnx_model_path, providers=providers)
            self.get_input_output_details()
        except Exception as e:
            raise RuntimeError(f"Error initializing the model: {e}")

    def get_input_output_details(self):
        model_inputs = self.session.get_inputs()
        model_outputs = self.session.get_outputs()

        self.input_names = [input_tensor.name for input_tensor in model_inputs]
        self.input_shape = model_inputs[0].shape
        self.input_height, self.input_width = self.input_shape[2], self.input_shape[3]

        self.output_names = [output_tensor.name for output_tensor in model_outputs]

    def detect_objects(self, image):
        input_tensor = self.prepare_input(image)
        outputs = self.inference(input_tensor)
        self.boxes, self.scores, self.class_ids = self.process_output(outputs)
        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        return input_tensor

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})
        return outputs

    def process_output(self, output):
        predictions = np.squeeze(output[0]).T
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)
        boxes = self.extract_boxes(predictions)
        indices = self.multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        # Apply additional filtering for overlapping boxes
        filtered_boxes, filtered_scores, filtered_class_ids = self.filter_overlapping_boxes(
            boxes[indices], scores[indices], class_ids[indices])

        return filtered_boxes, filtered_scores, filtered_class_ids

    def extract_boxes(self, predictions):
        boxes = predictions[:, :4]
        boxes = self.rescale_boxes(boxes)
        boxes = self.xywh2xyxy(boxes)
        return boxes

    def rescale_boxes(self, boxes):
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = boxes / input_shape.astype(np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def multiclass_nms(self, boxes, scores, class_ids, iou_threshold):
        unique_class_ids = np.unique(class_ids)
        keep_boxes = []

        for class_id in unique_class_ids:
            class_indices = np.where(class_ids == class_id)[0]
            class_boxes = boxes[class_indices, :]
            class_scores = scores[class_indices]
            class_keep_boxes = self.nms(class_boxes, class_scores, iou_threshold)
            keep_boxes.extend(class_indices[class_keep_boxes])

        return keep_boxes

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]
        keep_boxes = []

        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
            keep_indices = np.where(ious < iou_threshold)[0]
            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        iou = intersection_area / union_area
        return iou

    def filter_overlapping_boxes(self, boxes, scores, class_ids, iou_threshold=0.5):
        """Remove overlapping boxes with high IoU."""
        keep = []
        for i in range(len(boxes)):
            overlap_found = False
            for j in range(i):
                iou = self.compute_iou(boxes[i], np.array([boxes[j]]))
                if iou > iou_threshold:
                    overlap_found = True
                    break
            if not overlap_found:
                keep.append(i)

        return boxes[keep], scores[keep], class_ids[keep]

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):
        det_img = image.copy()

        img_height, img_width = image.shape[:2]
        font_size = min([img_height, img_width]) * 0.0006
        text_thickness = int(min([img_height, img_width]) * 0.001)

        det_img = self.draw_masks(det_img, self.boxes, self.class_ids, mask_alpha)

        for class_id, box, score in zip(self.class_ids, self.boxes, self.scores):
            color = self.colors[class_id]
            self.draw_box(det_img, box, color)
            label = self.class_names[class_id]
            caption = f'{label} {int(score * 100)}%'
            self.draw_text(det_img, caption, box, color, font_size, text_thickness)

        return det_img

    @staticmethod
    def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
                 thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    @staticmethod
    def draw_text(image: np.ndarray, text: str, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255),
                  font_size: float = 0.001, text_thickness: int = 2) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale=font_size, thickness=text_thickness)
        th = int(th * 1.2)

        cv2.rectangle(image, (x1, y1),
                      (x1 + tw, y1 - th), color, -1)

        return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255),
                           text_thickness, cv2.LINE_AA)

    def draw_masks(self, image: np.ndarray, boxes: np.ndarray, classes: np.ndarray,
                   mask_alpha: float = 0.3) -> np.ndarray:
        mask_img = image.copy()

        for box, class_id in zip(boxes, classes):
            color = self.colors[class_id]
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

        return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)
