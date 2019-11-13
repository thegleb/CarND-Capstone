#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose, TwistStamped
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
from scipy.spatial import KDTree
import tf
import math
import yaml
import time

DEBUG = False

LARGE_NUMBER = 100000
NUM_CONFIRMATIONS = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.is_site = self.config['is_site']
        if DEBUG:
            print('is_site [' + str(self.is_site) + ']')
        self.upcoming_stop_line_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.image_saver_pub = rospy.Publisher('/image_annotated', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.curr_vel = None

        # self.image_save_counter = 0

        self.waypoints = None
        self.waypoints_tree = None
        self.has_image = False
        self.last_processed = time.time()
        self.is_processing = False
        self.upcoming_stop_line = None

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        rospy.Subscriber('/image_color', Image, self.image_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.velocity_cb)

        # rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.upcoming_stop_line_pub.publish(self.upcoming_stop_line)
            # print(time.time() - self.last_processed)
            # process the first image and then every ~2 seconds
            # if self.has_image == False or time.time() - self.last_processed > 2.0:
            if self.pose and \
                    self.camera_image and \
                    (self.has_image == False or self.is_processing == False) and \
                    self.waypoints_tree:
                self.is_processing = True

                self.last_processed = time.time()
                self.initialized = True

                self.has_image = True
                start = time.time()
                light_wp, state = self.process_traffic_lights()
                if DEBUG:
                    print('detection took ' + str(time.time() - start))
                    print('detected state ' + str(state))
                # rospy.loginfo("light_wp [%i] state [%i]", light_wp, state)

                '''
                Publish upcoming red lights at detection frequency.
                Each predicted state has to occur `NUM_CONFIRMATIONS` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''

                # check distance from light
                if light_wp > -1:
                    line_x = self.waypoints.waypoints[light_wp].pose.pose.position.x
                    line_y = self.waypoints.waypoints[light_wp].pose.pose.position.y
                    car_x = self.pose.pose.position.x
                    car_y = self.pose.pose.position.y
                    dist_to_stopline = math.sqrt(pow(line_x - car_x, 2) + pow(line_y - car_y, 2))
                else:
                    dist_to_stopline = LARGE_NUMBER

                if DEBUG:
                    print('distance from the next light ' + str(dist_to_stopline))

                if self.state != state:
                    self.state_count = 1
                    self.state = state
                elif self.state_count >= NUM_CONFIRMATIONS:
                    self.last_state = self.state
                    if state == TrafficLight.GREEN:
                        light_wp = -1
                    elif state == TrafficLight.YELLOW:
                        # if we are close to the light and it's yellow, then keep going; else stop.
                        # if it takes worst case 2 seconds to notice the light change from green to yellow,
                        # it is likely to turn red in about ~1 more second.
                        # at current velocity we expect to cross the line in 0.8 seconds before it turns red.
                        light_wp = -1 if dist_to_stopline < self.curr_vel * 0.8 else light_wp
                    self.last_wp = light_wp
                    self.upcoming_stop_line = Int32(light_wp)
                else:
                    self.upcoming_stop_line = Int32(self.last_wp)
                self.state_count += 1

                self.is_processing = False

            rate.sleep()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints
        if self.waypoints_tree is None:
            self.waypoints_tree = KDTree([[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in
                                 waypoints.waypoints])

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def velocity_cb(self, msg):
        self.curr_vel = msg.twist.linear.x

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.camera_image = msg

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        return self.waypoints_tree.query([x, y], 1)[1]

    def get_light_state(self):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if not self.has_image:
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        light_state, annotated_image = self.light_classifier.get_classification(cv_image)

        if DEBUG:
            print('publishing camera image')
            self.image_saver_pub.publish(self.bridge.cv2_to_imgmsg(annotated_image, "rgb8"))

        return light_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        diff = LARGE_NUMBER
        if self.pose:
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line_x, line_y = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line_x, line_y)
                d = temp_wp_idx - car_wp_idx
                if 0 <= d < diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

        # ~220 waypoints is around 110 meters. At a velocity of 40 km/hr this offers us ~10 seconds to react.
        # Detection around 0.5-0.6sec/iteration and 2 iterations to confirm detection means we
        # have around ~9 seconds / ~100 meters to stop.
        if closest_light and diff < 220:
            state = self.get_light_state()
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
