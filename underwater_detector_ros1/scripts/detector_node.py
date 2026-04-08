#!/usr/bin/env python3

import os
import json
import numpy as np
import cv2
import onnxruntime as ort

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

CLASS_NAMES = [
    'Unknown Instance', 'Scissors', 'Plastic Cup', 'Metal Rod', 'Fork',
    'Bottle', 'Soda Can', 'Case', 'Plastic Bag', 'Cup', 'Goggles',
    'Flipper', 'LoCo', 'Aqua', 'Pipe', 'Snorkel', 'Spoon', 'Lure',
    'Screwdriver', 'Car', 'Tripod', 'ROV', 'Knife', 'Dive Weight'
]

DEFAULT_MODEL = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'models', 'best.onnx')


def nms(boxes, scores, iou_threshold):
    """Simple NMS. boxes: (N,4) xyxy, scores: (N,). Returns kept indices."""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= iou_threshold]
    return keep


class UnderwaterDetector(object):
    def __init__(self):
        rospy.init_node('underwater_detector')

        weights       = rospy.get_param('~weights',    DEFAULT_MODEL)
        imgsz         = rospy.get_param('~imgsz',      640)
        self.conf_thr = rospy.get_param('~conf_thres', 0.25)
        self.iou_thr  = rospy.get_param('~iou_thres',  0.45)
        device        = rospy.get_param('~device',     'cpu')

        self.imgsz  = (imgsz, imgsz)
        self.bridge = CvBridge()

        providers = (['CUDAExecutionProvider', 'CPUExecutionProvider']
                     if device == 'cuda' else ['CPUExecutionProvider'])
        self.session = ort.InferenceSession(weights, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        rospy.loginfo("Model loaded | providers=%s", self.session.get_providers())

        self.pub_img  = rospy.Publisher('/detections/image', Image, queue_size=10)
        self.pub_json = rospy.Publisher('/detections/json',  String, queue_size=10)

        rospy.Subscriber('/zedm/zed_node/left_raw/image_raw_color',
                         Image, self.image_callback, queue_size=1,
                         buff_size=2**24)
        rospy.loginfo("Waiting for images on /zedm/zed_node/left_raw/image_raw_color ...")

    def preprocess(self, img_bgr):
        img = cv2.resize(img_bgr, self.imgsz)
        img = img[:, :, ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
        return img[np.newaxis]                              # (1, 3, H, W)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr("cv_bridge error: %s", e)
            return

        orig_h, orig_w = frame.shape[:2]
        tensor = self.preprocess(frame)

        outputs = self.session.run(None, {self.input_name: tensor})
        # output shape: (1, 28, 8400) -> 28 = 4 (cx,cy,w,h) + 24 classes
        pred = outputs[0][0].T   # (8400, 28)

        cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        class_scores = pred[:, 4:]                         # (8400, 24)
        class_ids    = class_scores.argmax(axis=1)
        confidences  = class_scores[np.arange(len(class_ids)), class_ids]

        mask = confidences >= self.conf_thr
        cx, cy, w, h   = cx[mask], cy[mask], w[mask], h[mask]
        confidences    = confidences[mask]
        class_ids      = class_ids[mask]

        # Scale from model input size to original frame size
        sx = orig_w / self.imgsz[0]
        sy = orig_h / self.imgsz[1]
        x1 = ((cx - w / 2) * sx).astype(int)
        y1 = ((cy - h / 2) * sy).astype(int)
        x2 = ((cx + w / 2) * sx).astype(int)
        y2 = ((cy + h / 2) * sy).astype(int)

        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(float)
        keep  = nms(boxes, confidences, self.iou_thr) if len(boxes) else []

        detections = []
        for i in keep:
            label = CLASS_NAMES[class_ids[i]]
            score = float(confidences[i])
            bx1, by1, bx2, by2 = int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])
            detections.append({
                'label':      label,
                'confidence': round(score, 3),
                'bbox':       [bx1, by1, bx2, by2]
            })
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 100), 2)
            cv2.putText(frame, "%s %.2f" % (label, score),
                        (bx1, by1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 100), 2)

        self.pub_img.publish(self.bridge.cv2_to_imgmsg(frame, 'bgr8'))
        out = String()
        out.data = json.dumps(detections)
        self.pub_json.publish(out)

        if detections:
            rospy.loginfo("Detected %d: %s", len(detections),
                          [d['label'] for d in detections])


def main():
    node = UnderwaterDetector()
    rospy.spin()


if __name__ == '__main__':
    main()
