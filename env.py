"""
fuck
"""

import os
import sys

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import gridspec

from numpy import array, zeros
from math import *
from model import *
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

MAX_CONTROL = 10
MIN_CONTROL = -10
MAX_FORCE = 10

MAX_ROTATION = 2*pi
MIN_ROTATION = -2*pi
MAX_DROTATION = 10
MIN_DROTATION = -10

MAX_TRANSLATION = 5
MIN_TRANSLATION = -5
MAX_DTRANSLATION = 3
MIN_DTRANSLATION = -3

LIMITS = np.asarray([
    [MAX_TRANSLATION, MAX_DTRANSLATION, MAX_ROTATION, MAX_DROTATION],
    [MIN_TRANSLATION, MIN_DTRANSLATION, MIN_ROTATION, MIN_DROTATION]
])

scaler = MinMaxScaler(feature_range=(-1.0, 1.0)).fit(
    LIMITS
)


class InvertedPendulum:
    # def __init__(self, rotational=array([pi + 0.05, 0]), translational=array([0.0, 0.0])):
    def __init__(self, rotational=array([0.00, 0]), translational=array([0.0, 0.0])):
        self.time = 0.0
        self.control = 0.0
        self.rotation = rotational
        self.translation = translational
        self.state_list = []

        for i in range(10):
            self.state_list.append(self.get_current_state())

        self.LENGTH = 0.5
        self.GRAVITY = 9.81
        self.MASS = 0.4
        self.DT = 0.01

    def system_equation(self, y):

        theta = y[0]
        omega = y[1]

        d_omega = 1 / self.LENGTH * \
                 (\
                    -self.GRAVITY * sin(theta) \
                    -(self.control / self.MASS) * cos(theta) - 0.10 * omega \
                 )
        return array([omega, d_omega])

    def runge_kutta(self, y):

        k1 = self.system_equation(y)
        k2 = self.system_equation(y + self.DT / 2.0 * k1)
        k3 = self.system_equation(y + self.DT / 2.0 * k2)
        k4 = self.system_equation(y + self.DT * k3)

        return y + (self.DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def update_state(self):
        self.state_list.append(
            [
                self.translation[0],
                self.translation[1],
                self.rotation[0],
                self.rotation[1]
            ]
        )
        self.time += self.DT
        self.control = clip(-MAX_CONTROL, self.control, MAX_CONTROL)

        self.translation[0] = clip(-1.0e300, self.translation[0], 1.0e300)
        self.translation[1] = clip(-1.0e300, self.translation[1], 1.0e300)

        self.translation[1] += (self.control / self.MASS) * self.DT
        self.translation[0] += self.translation[1] * self.DT

        self.rotation = self.runge_kutta(self.rotation)

        def __wrap(value, max, min=0):
            value -= min
            max -= min
            if max == 0:
                return min

            value = value % max
            value += min
            while value < min:
                value += max

            return value

        self.rotation[0] = __wrap(self.rotation[0], max=2*pi, min=-2*pi)

    def simulate_step(self):
        self.update_state()

    def set_control(self, control):
        self.control = control

    def get_rotation(self):
        return self.rotation[0]

    def get_rotation_velocity(self):
        return self.rotation[1]

    def get_translation(self):
        return self.translation[0]

    def get_linear_velocity(self):
        return self.translation[1]

    def get_current_state(self):
        return np.asarray([self.translation[0], self.translation[1], self.rotation[0], self.rotation[1]])

    def get_state(self, i):
        if i > len(self.state_list):
            raise ValueError("index out of bounds")

        return np.asarray(self.state_list[len(self.state_list)-1-i])

    def get_states(self, i):
        return np.asarray(self.state_list[len(self.state_list)-1-i:])

    def simulate_steps_control(self, steps, controller):

        x = []
        f = []

        for i in range(steps):
            # control_input = self.get_current_state().reshape((1, -1))
            control_input = np.append(self.get_current_state().reshape((1, -1)), self.get_states(4), axis=0)
            control_input = scaler.transform(control_input).reshape((1, -1))

            control_output = controller.predict(control_input)
            self.control = control_output
            self.update_state()

            f.append(self.control)
            x.append((self.rotation[0], self.translation[0]))

        print("min force: ",min(f)," max force: ",max(f))

        return array(x)

    def render_sim(self, x):
        pmax = max(max(p[1:]) for p in x)
        pmin = min(min(p[1:]) for p in x)

        amax = max(max(p[0:]) for p in x)
        amin = min(min(p[0:]) for p in x)
        """
        Animates the pendulum
        ---------------------
        t : ndarray, shape(m)
            time array
        states : ndarray, shape(m,p)
            state vector
        """
        # set up figure, axis and plot element
        fig = plt.figure(figsize=(10,10))

        gs  = gridspec.GridSpec(3, 1, height_ratios=[4, 1 ,1])

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])
        ax3 = fig.add_subplot(gs[2])

        #ax1_ax = plt.subplot(311)
        ax1.set_xlim([pmin - (self.LENGTH + 1), pmax + (self.LENGTH + 1)])
        ax1.set_ylim([-3.05, 3.05])
        ax1.set_aspect('equal')

        #ax2_ax = plt.subplot(312)
        ax2.set_xlim([0, len(x) * self.DT])
        ax2.set_ylim([pmin, pmax])
        ax2.set_xlabel("time")
        ax2.set_ylabel("position")
        ax2.set_title("Postion / Time")

        #ax3_ax = plt.subplot(313)
        ax3.set_xlim([0, len(x) * self.DT])
        ax3.set_ylim([-amax, amax])
        ax3.set_xlabel("time")
        ax3.set_ylabel("angle")
        ax3.set_title("Angle / Time")

        #fig.tight_layout()

        line, = ax1.plot([],[],lw=2,marker='o',markersize=5)
        position, = ax2.plot([],[],'-b')
        angle, = ax3.plot([],[],'-b')

        def init():
            line.set_data([],[])
            position.set_data([],[])
            angle.set_data([],[])

            return line, position, angle,

        def animate(i):

            px = zeros(3)
            py = zeros(3)
            dx = zeros(i)
            da = zeros(i)
            dt = zeros(i)

            for j in range(i):
                dx = np.append(dx, x[j][1])
                dt = np.append(dt, j * self.DT)
                da = np.append(da, x[j][0])

            dx1 = x[i][1] + self.LENGTH * sin(x[i][0])
            dy1 = 0 - self.LENGTH * cos(x[i][0])

            px[0] = x[i][1] + 0.5
            px[1] = x[i][1] - 0.5
            py[0] = 0
            py[1] = 0

            thisx = [px[0], px[1], x[i][1], dx1]
            thisy = [py[0], py[1], 0, dy1]

            line.set_data(thisx, thisy)
            position.set_data(dt, dx)
            angle.set_data(dt, da)

            return line, position, angle,

        anim = animation.FuncAnimation(fig, animate, frames=len(x), init_func=init,
                interval=30, blit=False, repeat=True)
        #anim.save('pendulum.mp4', fps=18, dpi=75)
        plt.show(fig)


def clip(lo, x, hi):
    return lo if x <= lo else hi if x >= hi else x


def normalize(value, min, max):
    return ((1 - (-1))/(max - min)) * (value - max) + 1


def main(argv):

    if len(argv) == 0:
        usage()
        exit(0)

    n = argv[0]
    m_path = "models/" + n + "_model.h5"
    w_path = "models/" + n + "_weights.h5"

    model = load_model(m_path)
    model.load_weights(w_path)

    pendulum = InvertedPendulum()
    states = pendulum.simulate_steps_control(500, model)
    pendulum.render_sim(states)


def usage():
    print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])