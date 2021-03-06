#!/usr/bin/env python
#################################################################################
# Copyright 2018 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################

# Authors: Gilbert #

import rospy
import numpy as np
import math
import time
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
# from simulation_stage_1_respawnGoal import Respawn
# from respawnGoal import Respawn
# from simulation_stage_3_respawnGoal import Respawn
from simulation_stage_2_respawnGoal import Respawn
# from simulation_stage_oneway_respawnGoal_changing import Respawn
from gazebo_msgs.msg import ModelStates, ModelState
from geometry_msgs.msg import Pose

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.modelstates = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.goals=0
        self.crashes=0
        self.k=0
        self.Odom_x = []
        self.Odom_y = []
        self.Modelstates_x = []
        self.Modelstates_y = []
        self.Modelstates_x_avg = []
        self.Modelstates_y_avg = []
        self.difference_x = 0
        self.difference_y = 0
        self.speed_back = []
        self.action_back = []


    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w]
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    def getState(self, scan):
        scan_range = []
        heading = self.heading
        min_range = 0.13
        done = False

        for i in range(len(scan.ranges)):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        obstacle_min_range = round(min(scan_range), 2)
        obstacle_angle = np.argmin(scan_range)
        if min_range > min(scan_range) > 0:
            done = True

        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        if current_distance < 0.3:
            self.get_goalbox = True

        return scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done

        # Ausrechnen der Differenz zwischen ODOM und dem realen Messort
        # obstacle = ModelState()
        # model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        # for i in range(len(model.name)):
        #     if model.name[i] == 'turtlebot3_burger':
        #         obstacle.pose = model.pose[i]
        # self.Odom_x.append(self.position.x)
        # self.Odom_y.append(self.position.y)
        # self.Modelstates_x.append(obstacle.pose.position.x)
        # self.Modelstates_y.append(obstacle.pose.position.y)
        # self.k += 1
        # if self.k==500:
        #     for i in range(len(self.Odom_x)):
        #         self.difference_x += math.fabs(self.Odom_x[i] - self.Modelstates_x[i])
        #         self.difference_y += math.fabs(self.Odom_y[i] - self.Modelstates_y[i])
        #     difference_x_avg = self.difference_x / len(self.Odom_x)
        #     difference_y_avg = self.difference_y / len(self.Odom_x)
        #     self.k = 0
        #     self.Modelstates_x_avg.append(difference_x_avg)
        #     self.Modelstates_y_avg.append(difference_y_avg)
        #     difference_x_avg=0
        #     difference_y_avg=0
        #     self.Odom_x = []
        #     self.Odom_y= []
        #     self.Modelstates_x= []
        #     self.Modelstates_y= []
        #
        #     for i in range(len(self.Modelstates_x_avg)):
        #         print i+1, ". difference x:", self.Modelstates_x_avg[i]
        #     for i in range(len(self.Modelstates_y_avg)):
        #         print i+1, ". difference y:", self.Modelstates_y_avg[i]


    def setReward(self, state, done, action):
        yaw_reward = []
        obstacle_min_range = state[-2]
        current_distance = state[-3]
        heading = state[-4]
        self.speed_back.append(action)

        for i in range(5):
            angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)

        distance_rate = 2 ** (current_distance / self.goal_distance)

        if obstacle_min_range < 0.5:
            ob_reward = -3
        else:
            ob_reward = 0

        reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward - 3


        if done:
            rospy.loginfo("Collision!!")
            reward = -200
            self.pub_cmd_vel.publish(Twist())
            self.crashes += 1
            print "Reached goals:", self.goals, "Crashes:", self.crashes, "Total tries:", self.crashes

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 500
            self.pub_cmd_vel.publish(Twist())
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
            self.goals+=1
            print "Reached goals:", self.goals, "Crashes:", self.crashes, "Total tries:", self.crashes

        return reward, self.crashes, self.goals


    def speed(self):
        speed = 0.15
        if self.goal_distance < 0.5:
            speed = 0.15 + 0.1 * ((self.goal_distance-0.5)/0.5)
        return speed

    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
        vel_cmd = Twist()
        # vel_cmd.linear.x = 0.15
        vel_cmd.linear.x = self.speed()
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        state, done = self.getState(data)
        reward, crashes, goals = self.setReward(state, done, action)


        return np.asarray(state), reward, done
#, crashes, goals

    def reset(self):
        rospy.wait_for_service('gazebo/reset_simulation')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_simulation service call failed")

        # driving backwards when crashing but has some bugs
        # obstacle = ModelState()
        # crash_distance = 0.35
        # model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        # for i in range(len(model.name)):
        #     if model.name[i] == 'turtlebot3_burger':
        #         obstacle.pose = model.pose[i]
        # a = obstacle.pose.position.x
        # b = obstacle.pose.position.y
        # crash_position = True
        # while crash_position:
        #     vel_cmd = Twist()
        #     vel_cmd.linear.x = -0.2
        #     self.pub_cmd_vel.publish(vel_cmd)
        #     model = rospy.wait_for_message('gazebo/model_states', ModelStates)
        #     for i in range(len(model.name)):
        #         if model.name[i] == 'turtlebot3_burger':
        #             obstacle.pose = model.pose[i]
        #             c = obstacle.pose.position.x
        #             d = obstacle.pose.position.y
        #     if math.fabs(math.sqrt((a-c)**2+(b-d)**2)) > crash_distance:
        #         vel_cmd.linear.x = 0
        #         self.pub_cmd_vel.publish(vel_cmd)
        #         crash_position = False

        crash_position = True

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)
