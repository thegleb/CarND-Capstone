import numpy as np
import tensorflow as tf
import cv2
from PIL import ImageDraw
from PIL import Image as PIL_Image

from styx_msgs.msg import TrafficLight

DEBUG = False

NN_GRAPH_PREFIX = './light_classification/'

MODEL_TO_SIM = {
    '1': {'ros_label': TrafficLight.GREEN, 'bbox_color': 'green'},
    '2': {'ros_label': TrafficLight.RED, 'bbox_color': 'red'},
    '3': {'ros_label': TrafficLight.YELLOW, 'bbox_color': 'yellow'},
}

MODEL_TO_SITE = {
    '1': {'ros_label': TrafficLight.GREEN, 'bbox_color': 'green'},
    '2': {'ros_label': TrafficLight.RED, 'bbox_color': 'red'},
    '3': {'ros_label': TrafficLight.YELLOW, 'bbox_color': 'yellow'},
}

class TLClassifier(object):
    def __init__(self, is_site):
        self.is_site = is_site
        if is_site:
            graph_path = 'site/ssd_mobilenet_v1_coco_20000_gamma/frozen_inference_graph.pb'
        else:
            graph_path = 'sim/ssd_mobilenet_v1_coco_20000steps/frozen_inference_graph.pb'
        self.graph = self.load_graph(NN_GRAPH_PREFIX + graph_path)
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.sess = tf.Session(graph=self.graph)

    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph

    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes

    def draw_boxes(self, image, boxes, classes, lookup_dict, thickness=3):
        """Draw bounding boxes on the image"""
        draw = ImageDraw.Draw(image)
        for i in range(len(boxes)):
            bot, left, top, right = boxes[i, ...]
            class_id = int(classes[i])
            color = lookup_dict[str(class_id)]['bbox_color']
            draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

    def to_image_coords(self, boxes, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(boxes)
        box_coords[:, 0] = boxes[:, 0] * height
        box_coords[:, 1] = boxes[:, 1] * width
        box_coords[:, 2] = boxes[:, 2] * height
        box_coords[:, 3] = boxes[:, 3] * width

        return box_coords

    def adjust_gamma(self, img, gamma=1.0):
        inv_gamma = 1.0 / gamma
        table = np.array([
          ((i / 255.0) ** inv_gamma) * 255
          for i in np.arange(0, 256)])
        return cv2.LUT(img.astype(np.uint8), table.astype(np.uint8))

    def adjust_contrast(self, img):
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(2,2))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        img[:,:,0] = clahe.apply(img[:,:,0])
        img = cv2.cvtColor(img, cv2.COLOR_LAB2RGB)
        return img

    def preprocess(self, img, gamma=0.4):
        img = self.adjust_contrast(img)
        img = self.adjust_gamma(img, gamma)
        return img

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        sess = self.sess

        # height, width, shape = image.shape
        # rescale image
        # image = cv2.resize(image, (int(width / 2), int(height / 2)))
        # crop image
        # image = image[int(0.2 * height):int(0.6 * height), int(width * 0.3):int(width * 0.6)]
        # convert to RGB for detection
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.preprocess(image, 0.4)
        image_np = np.expand_dims(image, 0)
        # Actual detection.
        (boxes, scores, classes) = sess.run(
            [
                self.detection_boxes,
                self.detection_scores,
                self.detection_classes
            ],
            feed_dict={self.image_tensor: image_np}
        )

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.5
        # Filter boxes with a confidence score less than `confidence_cutoff`
        lookup_dict = MODEL_TO_SITE if self.is_site else MODEL_TO_SIM
        if DEBUG is False:
            annotated_image = image
        else:
            height, width, channels = image.shape
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            adjusted_boxes = self.to_image_coords(boxes, height, width)
            print(classes)
            print(scores)

            pil_image = PIL_Image.fromarray(image)
            self.draw_boxes(pil_image, adjusted_boxes, classes, lookup_dict)
            annotated_image = np.asarray(pil_image)

        light_status = TrafficLight.UNKNOWN
        if len(scores) == 0 or scores[0] < confidence_cutoff:
            return light_status, annotated_image

        likely_color = int(classes[0])

        if str(likely_color) in lookup_dict:
            light_status = lookup_dict[str(likely_color)]['ros_label']

        if DEBUG:
            print('likely_color [' + str(likely_color) + ']')
            print('light status is ' + str(light_status))
        return light_status, annotated_image
