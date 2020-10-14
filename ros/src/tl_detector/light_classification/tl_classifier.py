import rospy
from styx_msgs.msg import TrafficLight

PATH_TO_GRAPH = ''
THRESHOLD = .5


class TLClassifier(object):
    def __init__(self):
        """
        Load the frozen graph
        """
        self.graph = tf.Graph()
        with self.graph.as_default():
            graph_df = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_GRAPH, 'rb') as fid:
                graph_df.ParseFromString(fid.read())
                tf.import_graph_def(graph_df, name='')

            self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
            self.classes = self.graph.get_tensor_by_name('detection_classes:0')
            self.boxes = self.graph.get_tensor_by_name('detection_boxes:0')
            self.scores = self.graph.get_tensor_by_name('detection_scores:0')
            self.num_detections = self.graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=self.graph)

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        with self.graph.as_default():
            img_expand = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: img_expand})

        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        prediction = TrafficLight.UNKNOWN
        if scores[0] > THRESHOLD:
            if classes[0] == 1:
                prediction = TrafficLight.GREEN
                rospy.logdebug('Green')
            elif classes[0] == 2:
                prediction = TrafficLight.RED
                rospy.logdebug('RED')
            elif classes[0] == 3:
                prediction = TrafficLight.YELLOW
                rospy.logdebug('YELLOW')
            else:
                rospy.debuglog('UNKNOWN')
        else:
            rospy.debuglog('UNKNOWN -- below threshold')

        return prediction
