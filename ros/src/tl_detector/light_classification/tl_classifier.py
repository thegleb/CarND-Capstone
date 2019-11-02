import numpy as np
import tensorflow as tf

from styx_msgs.msg import TrafficLight

nn_graph_prefix = './light_classification/'

class TLClassifier(object):
    def __init__(self):
        # self.graph = self.load_graph(nn_graph_prefix + 'ssd_inception_v2_1000steps/frozen_inference_graph.pb')
        self.graph = self.load_graph(nn_graph_prefix + 'ssd_inception_v2_10000steps/frozen_inference_graph.pb')
        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.detection_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.graph.get_tensor_by_name('detection_classes:0')

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
        #TODO implement light color prediction


        # image = preprocess(image)
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        with tf.Session(graph=self.graph) as sess:
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

            confidence_cutoff = 0.4
            # Filter boxes with a confidence score less than `confidence_cutoff`
            boxes, scores, classes = self.filter_boxes(confidence_cutoff, boxes, scores, classes)
            print(classes)
            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            # width, height = image.size
            # adjusted_boxes = self.to_image_coords(boxes, height, width)
            if len(scores) == 0:
                return TrafficLight.UNKNOWN

            value = max(scores)
            idx = np.where(scores == value)
            print('max [' + str(value) + '] for class [' + str(classes[value]) + ']')
            if int(classes[idx]) == 1:
                return TrafficLight.GREEN
            elif int(classes[idx]) == 2:
                return TrafficLight.RED
            elif int(classes[idx]) == 3:
                return TrafficLight.YELLOW
            else:
                return TrafficLight.UNKNOWN
