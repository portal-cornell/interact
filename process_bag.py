import rospy
# from quaternion import from_euler_angles, as_float_array
from visualization_msgs.msg import Marker, MarkerArray

from std_msgs.msg import Float32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Point
import math
import numpy as np

def euler_from_quaternion(x, y, z, w):
    """
    Convert a quaternion into euler angles (roll, pitch, yaw)
    roll is rotation around x in radians (counterclockwise)
    pitch is rotation around y in radians (counterclockwise)
    yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4) 
    
    return roll_x, pitch_y, yaw_z # in degrees

class BagReaderHR():
    def __init__(self):
        rospy.init_node('realworld', anonymous=True)
        ee_sub = rospy.Subscriber('/ee_curr_pose', PoseStamped, self.ee_callback, queue_size=1)
        human_forecast_subscriber = rospy.Subscriber('/human_forecast', MarkerArray, self.human_forecast_callback, queue_size=1)
        self.atiksh_pose = {}
        self.kushal_pose = {}
        self.ee_pose = None

    def human_forecast_callback(self, marker_array):
        for marker in marker_array.markers:
            # print(marker.ns)
            if 'Kushal' in marker.ns:
                left_edge, right_edge = marker.ns.split('-')[2].split('_')
                p1, p2 = marker.points
                self.kushal_pose[left_edge] = np.array([p1.x, p1.y, p1.z])
                self.kushal_pose[right_edge] = np.array([p2.x, p2.y, p2.z])
            if 'Atiksh' in marker.ns:
                left_edge, right_edge = marker.ns.split('-')[2].split('_')
                p1, p2 = marker.points
                self.atiksh_pose[left_edge] = np.array([p1.x, p1.y, p1.z])
                self.atiksh_pose[right_edge] = np.array([p2.x, p2.y, p2.z])

    def get_marker_array(self, alpha=1, red=1, green=0, blue=0):
        pose = self.ee_pose
        SCALE = 0.015
        marker = Marker()
        marker.header.frame_id = "panda_link0"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker.LINE_LIST
        marker.id = 0
        marker.scale.x = SCALE
        marker.action = marker.ADD 
        marker.ns = f'robot-ee'
        marker.color.r = red
        marker.color.g = green
        marker.color.b = blue
        marker.color.a = alpha
        p1m = Marker()
        p1m.header.frame_id = "panda_link0"
        p1m.header.stamp = rospy.Time.now()
        p1m.type = marker.SPHERE_LIST
        p1m.id = 1
        p1m.scale.x = SCALE*2
        p1m.scale.y = SCALE*2
        p1m.scale.z = SCALE*2
        p1m.action = p1m.ADD
        p1m.color.r = red
        p1m.color.g = green
        p1m.color.b = blue
        p1m.color.a = alpha/2
        pos1, pos2 = pose[0], pose[1]
        p1, p2 = Point(), Point()
        x, y, z = pos1
        p1.x, p1.y, p1.z = x, y, z
        x, y, z = pos2
        p2.x, p2.y, p2.z = x, y, z

        p1m.points = [p1]
        marker.points = [p1, p2]

        marker_array = MarkerArray()
        marker_array.markers = [marker, p1m]
        return marker_array
    
    def get_atiksh_rpy(self):
        wrist_pos = self.atiksh_pose['RWristOut']
        hand_pos = self.atiksh_pose['RHandOut']
        diff = hand_pos-wrist_pos
        dist = np.linalg.norm(diff)
        return np.arccos(diff[0]/dist), np.arccos(diff[1]/dist), np.arccos(diff[2]/dist)

    def ee_callback(self, pose, ee_length = 0.0):
        pos = pose.pose.position
        x1, y1, z1 = pos.x, pos.y, pos.z
        ori = pose.pose.orientation 
        q_x, q_y, q_z, q_w = ori.w, ori.x , ori.y, ori.z
        roll, pitch, yaw = euler_from_quaternion(q_x, q_y, q_z, q_w)
        x2, y2, z2 = x1+np.cos(roll)*ee_length, y1+np.cos(pitch)*ee_length, z1+np.cos(yaw)*ee_length
        print("ROBOT RPY = ", roll, pitch, yaw)

        roll, pitch, yaw = self.get_atiksh_rpy()
        print("Human RPY = ", roll, pitch, yaw)
        self.ee_pose = [[x1, y1, z1], [x2, y2, z2]]

ee_pub = rospy.Publisher('/ee_pub', MarkerArray)
br = BagReaderHR()
rate = rospy.Rate(15)

ee_poses = []
atiksh_poses = []
kushal_poses = []
for timestep in range(1000):
    print(f'Time = {timestep/15}')
    if br.ee_pose is not None:
        robot_ee = br.get_marker_array()
        ee_pub.publish(robot_ee)
    rate.sleep()
    
