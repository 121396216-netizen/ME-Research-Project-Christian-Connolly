# sim_ws/src/wall_following/wall_following/wall_following_node.py
import rclpy
import math
from rclpy.node import Node
#import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
class WallFollow(Node):
    """ 
    Implement Wall Following on the car
    """
    def __init__(self):
        super().__init__('wall_follow_node')
        # Create subscribers and publishers
        self.scan_subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 25)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/ackermann_cmd', 25)
        # Set PID gains
        self.kp = 5
        self.kd = 0.9
        self.ki = 0
        # Speed PID gains
        self.kp_speed = 1.00 # Proportional gain for speed
        self.ki_speed = 0.01 # Integral gain for speed
        self.kd_speed = 0.005 # Derivative gain for speed
        # Store history
        self.integral_steering = 0.0
        self.prev_error_steering = 0.0
        self.error_steering = 0.0
        self.integral_speed = 0.0
        self.prev_error_speed = 0.0
        # Store any necessary values
        self.safe_speed = 0.0
        self.desired_distance = 0.5
        self.latest_steering_angle = 0.0 #[-100,100] percentage of steering angle
        self.desired_velocity = 0.5 #[-100,100] percentage of power
        self.zero = 90 # left of the car 90 degrees

        self.theta = 50 #scan angle of lidar
        self.dTheta = 90 - self.theta
        self.deltaT = 0.01 # change to 10ms when controlled
        #self.target_speed = 0.0 #current speed of the car, should be close to desired_velocity
        
    def get_ranges_at_angles(self, msg):
        """
        Get lidar range measurements at specific angles.
        Args:
                msg: Incoming LaserScan message
                angle_0: First angle in degrees (default: 0)
                angle_theta: Second angle in degrees (default: 50)
        Returns:
                tuple containing range measurements at angle_0 and angle_theta
        """
        # Convert angles to radians
        angle_0_rad = math.radians(self.zero)
        angle_theta_rad = math.radians(self.theta)
        # Calculate the corresponding indices for the specified angles
        index_1 = int((angle_0_rad - msg.angle_min) / msg.angle_increment)
        index_2 = int((angle_theta_rad - msg.angle_min) / msg.angle_increment)
        # Retrieve range measurements at the specified indices
        range_at_0 = msg.ranges[index_1] if 0 <= index_1 < len(msg.ranges) and not math.isnan(msg.ranges[index_1]) and not math.isinf(msg.ranges[index_1]) else 0.0
        range_at_theta = msg.ranges[index_2] if 0 <= index_2 < len(msg.ranges) and not math.isnan(msg.ranges[index_2]) and not math.isinf(msg.ranges[index_2]) else 0.0
        #print ("distance: ",range_at_0, range_at_theta)
        return range_at_0, range_at_theta

    def get_error(self,range_at_0,range_at_theta):
        """
        Calculates the error to the wall. Follow the wall to the left (going 
        counter clockwise in the Levine loop).
        You potentially will need to use get_range()
        Args:
                range_data: single range array from the LiDAR
                dist: desired distance to the wall
        Returns:
                error: calculated error
        """

        alpha = math.atan(((range_at_theta*math.cos(self.dTheta))-
        range_at_0)/(range_at_theta*math.sin(self.dTheta)))
        currentDistance = range_at_0 * math.cos(alpha) # y distance to the wall
        L = self.safe_speed * self.deltaT #length of AC every 10ms
        print("alpha: ", alpha)
        print("CD: ",currentDistance)
        futureDistance = (currentDistance + (L * math.sin(alpha))) # CD = y + L * sin(alpha)
        error_steering = -(self.desired_distance - futureDistance) # error
        print(error_steering) 
        return error_steering
    
    def compute_speed(self, steering_angle):
        # Convert steering angle from radians to degrees for comparison
        angle = abs(math.degrees(steering_angle)) 
        # Determine speed based on steering angle
        if angle > 20:
            speed = 1.0
        elif angle > 10:
            speed = 2.0
        else:
            speed = 3.0
        return speed

    def pid_control(self, error_steering, desired_velocity):
        """
        Based on the calculated error, publish vehicle control
        Args:
                error: calculated error
                velocity: desired velocity
        Returns:
                None
        """
        self.integral_steering += (error_steering * self.deltaT)
        derivative = error_steering - self.prev_error_steering
        # PID control equation
        steering_angle_correction = (self.kp * error_steering) + (self.ki * 
        self.integral_steering) + (self.kd * derivative)
        steering_angle = steering_angle_correction
        max_steering_angle = math.radians(30) # steering max is 30 degrees
        print("steering corr",steering_angle_correction)
        steering_angle = max(-max_steering_angle, min(max_steering_angle, 
        steering_angle))
        self.safe_speed = self.compute_speed(steering_angle)
        target_speed = self.safe_speed
        # Calculate speed error (difference between desired and actual velocity)
        #speed_error = self.desired_velocity - target_speed # You'll need to measure or estimate actual_velocity
        # Update integral and derivative for speed
        #self.integral_speed += speed_error * self.deltaT
        #derivative_speed = (speed_error - self.prev_error_speed) / self.deltaT
        # PID for speed
        #speed_correction = self.kp_speed * speed_error + self.ki_speed * self.integral_speed + self.kd_speed * derivative_speed
        #target_speed = self.desired_velocity + speed_correction # Adjust target speed based on correction
        # Ensure target_speed is within acceptable bounds
        #target_speed = max(0, min(max_speed, target_speed)) # max_speed is the maximum speed your vehicle can handle
        # Actuate the car with PID
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = target_speed
        self.drive_publisher.publish(drive_msg)
        # Update previous error
        self.prev_error_steering = error_steering
        self.latest_steering_angle = steering_angle
        #self.prev_error_speed = speed_error
    
    def scan_callback(self, msg):
        """
        Callback function for LaserScan messages. Calculate the error and 
        publish the drive message in this function.
        Args:
                msg: Incoming LaserScan message
        Returns:
                None
        """
        #print("LiDAR Ranges:", msg.ranges)
        # Replace with error calculated by get_error()
        range_at_0, range_at_theta = self.get_ranges_at_angles(msg)
        #self.get_ranges_at_angles(self, msg)
        error_steering = self.get_error(range_at_0, range_at_theta)
        print("Error:", error_steering)
        # Actuate the car with PID
        self.pid_control(error_steering, self.desired_velocity)
        print("Steering Angle:", self.latest_steering_angle)
        
def main(args=None):
    rclpy.init(args=args)
    print("WallFollow Initialized")
    wall_follow_node = WallFollow()
    rclpy.spin(wall_follow_node)
        
    wall_follow_node.destroy_node()
    rclpy.shutdown()
        
if __name__ == '__main__':
    main()