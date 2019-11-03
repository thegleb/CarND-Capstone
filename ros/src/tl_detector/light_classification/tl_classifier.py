import numpy as np
import tensorflow as tf
import cv2

from styx_msgs.msg import TrafficLight

nn_graph_prefix = './light_classification/'

class TLClassifier(object):
    def __init__(self):
        # self.graph = self.load_graph(nn_graph_prefix + 'ssd_inception_v2_10000steps/frozen_inference_graph.pb')
        self.graph = self.load_graph(nn_graph_prefix + 'ssd_mobilenet_v1_coco_10000steps/frozen_inference_graph.pb')
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

        # image = preprocess(image)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
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
        # boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
        # print(classes)
        # print(scores)
        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        # width, height = image.size
        # adjusted_boxes = self.to_image_coords(boxes, height, width)
        light_status = TrafficLight.UNKNOWN
        if len(scores) == 0 or scores[0] < confidence_cutoff:
            return light_status

        # likely_color = int(classes[0])
        likely_color = int(classes[0])
        print('likely_color [' + str(likely_color) + ']')
        if likely_color == 1:
            light_status = TrafficLight.GREEN
        elif likely_color == 2:
            light_status = TrafficLight.RED
        elif likely_color == 3:
            light_status = TrafficLight.YELLOW
        return light_status
