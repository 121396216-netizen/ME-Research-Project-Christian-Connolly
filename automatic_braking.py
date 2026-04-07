import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import geometry_msgs.msg
import math


class automatic_emergency_braking(Node):

    def __init__(self):
        super().__init__('automatic_emergency_braking')
        self.brake_publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 10)
        self.brake_bool_publisher = self.create_publisher(Bool, '/brake_bool', 10)
        #self.brake_publisher = self.create_publisher(geometry_msgs.msg.Twist, 'cmd_vel', 10)

        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_subscription = self.create_subscription(Odometry, '/odom', self.odom_callback,10) #'ego_racecar/odom'
        
        # Lidar configuration
        # self.car_width = 0.2
        # self.lidar_to_front = 0.05
        # self.lidar_to_side = 0.1
        # self.lidar_to_back = 0.27
        # self.lidar_mounting_angle = math.radians(0)  # Lidar mounting angle in radians
        # self.lidar_to_edge = self.calculate_lidar_to_edge()
        self.lidar_to_edge = 0.1
        self.speed = 0
        
    def odom_callback(self, odom_msg):
        self.speed = -1 * odom_msg.twist.twist.linear.x

    def scan_callback(self, scan_msg: LaserScan) -> None:
        #for i, angle in enumerate(scan_msg.angle_min + i * scan_msg.angle_increment for i in range(len(scan_msg.ranges))):
        #    adjusted_range = scan_msg.ranges[i] - self.lidar_to_edge
        #    ttc = adjusted_range / self.speed if self.speed != 0.0 else float('inf')
            
        adjusted_range = [range - self.lidar_to_edge for range in scan_msg.ranges]
        min_range = min(adjusted_range)     
        ttc = min_range / self.speed if self.speed != 0.0 else float('inf')
        threshold_high = 0.3
        threshold_low = 0.2
        print(ttc)
        #print(self.speed)
        if abs(ttc) < threshold_high:
            # Perform emergency braking action
            brake_msg = AckermannDriveStamped()
            #twist = geometry_msgs.msg.Twist()
            #twist.linear.x = 0.0
            #twist.linear.y = 0.0
            #twist.linear.z = 0.0
            #twist.angular.x = 0.0
            #twist.angular.y = 0.0
            #twist.angular.z = 0.0
            
            #self.brake_publisher.publish(twist)
            brake_msg.drive.speed = 0.0
            self.brake_publisher.publish(brake_msg)
            
            brake_bool_msg = Bool()
            brake_bool_msg.data = True if ttc < threshold_low else False
            self.brake_bool_publisher.publish(brake_bool_msg)
            self.get_logger().warn("Emergency braking activated!")

    # def calculate_lidar_to_edge(self):
    #     # Calculate the distance from the edge of the car to the wall based on the lidar's mounting angle
    #     return self.lidar_to_back + self.car_width / 2 * math.tan(self.lidar_mounting_angle)

def main(args=None):
    rclpy.init(args=args)
    aeb = automatic_emergency_braking()
    rclpy.spin(aeb)
    rclpy.shutdown()
    
    
if __name__ == '__main__':
    main()