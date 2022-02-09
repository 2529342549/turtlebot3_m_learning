#!/usr/bin/env python

import numpy as np
import os
import time
import subprocess
import tf
import math
from math import radians, copysign, sqrt, pow, pi, atan2
import time
import Tkinter as tk
from PIL import ImageTk, Image

np.random.seed(1)
PhotoImage = ImageTk.PhotoImage
UNIT = 30  # pixels
HEIGHT = 21  # grid height
WIDTH = 21  # grid width


class Env(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.action_space = ['u', 'd', 'l', 'r', 'l_u', 'r_d', 'l_d', 'r_u']
        self.n_actions = len(self.action_space)
        self.title('LSTM_RL')
        self.geometry('{0}x{1}'.format(HEIGHT * UNIT, HEIGHT * UNIT))
        self.shapes = self.load_images()
        self.canvas = self._build_canvas()
        self.texts = []
        self.act = 0
        self.total_x = 0
        self.total_y = 0

    def _build_canvas(self):
        canvas = tk.Canvas(self, bg='aliceblue',  # '#F0F8FF'
                           height=HEIGHT * UNIT,
                           width=WIDTH * UNIT)
        # create grids
        for c in range(0, WIDTH * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = c, 0, c, HEIGHT * UNIT
            canvas.create_line(x0, y0, x1, y1)
        for r in range(0, HEIGHT * UNIT, UNIT):  # 0~400 by 80
            x0, y0, x1, y1 = 0, r, HEIGHT * UNIT, r
            canvas.create_line(x0, y0, x1, y1)

        # add img to canvas
        self.rectangle = canvas.create_image(315, 345, image=self.shapes[0])
        self.triangle1 = canvas.create_image(195, 195, image=self.shapes[1])
        self.triangle2 = canvas.create_image(195, 435, image=self.shapes[1])
        self.triangle3 = canvas.create_image(435, 195, image=self.shapes[1])
        self.triangle4 = canvas.create_image(435, 435, image=self.shapes[1])
        self.circle = canvas.create_image(525, 105, image=self.shapes[2])
        self.yellow_rectangle1 = canvas.create_image(255, 315, image=self.shapes[3])
        self.yellow_rectangle2 = canvas.create_image(285, 315, image=self.shapes[3])
        self.yellow_rectangle3 = canvas.create_image(345, 225, image=self.shapes[3])
        self.yellow_rectangle4 = canvas.create_image(345, 255, image=self.shapes[3])

        # pack all
        canvas.pack()

        return canvas

    def load_images(self):
        rectangle = PhotoImage(
            Image.open("/home/wangqiang/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/img/rectangle.png").resize((20, 20)))
        triangle = PhotoImage(
            Image.open("/home/wangqiang/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/img/triangle.png").resize((20, 20)))
        circle = PhotoImage(
            Image.open("/home/wangqiang/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/img/circle.png").resize((20, 20)))
        yellow_rectangle = PhotoImage(
            Image.open("/home/wangqiang/catkin_ws/src/turtlebot3_machine_learning/turtlebot3_dqn/img/Yellow_Rectangle.png").resize(
                (20, 20)))

        return rectangle, triangle, circle, yellow_rectangle

    def coords_to_state(self, coords):
        x = int((coords[0] - 15) / 30)
        y = int((coords[1] - 15) / 30)
        return [x, y]

    def state_to_coords(self, state):
        x = int(state[0] * 30 + 15)
        y = int(state[1] * 30 + 15)
        return [x, y]

    def reset(self):
        self.update()
        time.sleep(0.1)
        x, y = self.canvas.coords(self.rectangle)
        self.canvas.move(self.rectangle, UNIT / 2 - x + 300, UNIT / 2 - y + 330)
        self.render()
        return self.coords_to_state(self.canvas.coords(self.rectangle))

    def step(self, action):
        state = self.canvas.coords(self.rectangle)
        base_action = np.array([0, 0])
        self.render()
        if action == 0:  # up
            if state[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if state[1] < (HEIGHT - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # left
            if state[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:  # right
            if state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 4:  # left_up
            if state[1] > UNIT and state[0] > UNIT:
                base_action[0] -= UNIT
                base_action[1] -= UNIT
        elif action == 5:  # right_down
            if state[1] < (HEIGHT - 1) * UNIT and state[0] < (WIDTH - 1) * UNIT:
                base_action[0] += UNIT
                base_action[1] += UNIT
        elif action == 6:  # left_down
            if state[0] > UNIT and state[0] < (WIDTH - 1) * UNIT:
                base_action[0] -= UNIT
                base_action[1] += UNIT
        elif action == 7:  # right_up
            if state[0] < (WIDTH - 1) * UNIT and state[1] > UNIT:
                base_action[0] += UNIT
                base_action[1] -= UNIT

        self.canvas.move(self.rectangle, base_action[0], base_action[1])
        self.canvas.tag_raise(self.rectangle)
        next_state = self.canvas.coords(self.rectangle)

        if next_state == self.canvas.coords(self.circle):
            reward = 100
            done = True
            rospy.loginfo("Success!!!")
        elif next_state in [self.canvas.coords(self.triangle1),
                            self.canvas.coords(self.triangle2),
                            self.canvas.coords(self.triangle3),
                            self.canvas.coords(self.triangle4)]:
            reward = -100
            done = True
            rospy.loginfo("Collision!!!")
            self.pub_cmd_vel.publish(Twist())
            self.act = 0
        else:
            reward = 0
            done = False

        next_state = self.coords_to_state(next_state)
        return next_state, reward, done

    def render(self):
        time.sleep(0.03)
        self.update()
