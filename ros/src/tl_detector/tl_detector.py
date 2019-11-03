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

NUM_SEEN_BEFORE_STATE_CHANGE = 2
# anything closer than this and we risk running a red light or stopping too quickly
MIN_STOPPING_DISTANCE = 50
DEBUG = False

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_stop_line_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        self.image_saver_pub = rospy.Publisher('/image_annotated', Image, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
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

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)
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
            if self.camera_image and (self.has_image == False or self.is_processing == False):
                self.is_processing = True

                self.last_processed = time.time()
                self.initialized = True

                self.has_image = True
                start = time.time()
                light_wp, state = self.process_traffic_lights()
                if DEBUG is True:
                    print('detection took ' + str(time.time() - start))
                    print('detected state ' + str(state))
                # rospy.loginfo("light_wp [%i] state [%i]", light_wp, state)

                '''
                Publish upcoming red lights at detection frequency.
                Each predicted state has to occur `NUM_SEEN_BEFORE_STATE_CHANGE` number
                of times till we start using it. Otherwise the previous stable state is
                used.
                '''

                # check distance from light
                line_x = self.waypoints.waypoints[light_wp].pose.pose.position.x
                line_y = self.waypoints.waypoints[light_wp].pose.pose.position.y
                car_x = self.pose.pose.position.x
                car_y = self.pose.pose.position.y
                dist_to_stopline = math.sqrt(pow(line_x - car_x, 2) + pow(line_y - car_y, 2))

                if DEBUG is True:
                    print('distance from the next light ' + str(dist_to_stopline))

                if self.state != state:
                    self.state_count = 1
                    self.state = state
                elif self.state_count >= NUM_SEEN_BEFORE_STATE_CHANGE:
                    self.last_state = self.state
                    if state == TrafficLight.GREEN:
                        light_wp = -1
                    elif state == TrafficLight.YELLOW:
                        # if we are close to the light and it's yellow, then keep going; else stop
                        # based on guesstimate of 10 meters at top sim velocity (23 mph = ~10 m/sec)
                        # if (as of time of writing) it takes worst case 2 seconds to notice the light switch
                        # from green to yellow, it is likely to change in about ~1 more second
                        # so at current velocity we expect to cross the line before the light turns red
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

        if DEBUG is True:
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

        if closest_light:
            state = self.get_light_state()
            return line_wp_idx, state

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
