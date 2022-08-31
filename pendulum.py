import sys
import getopt
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import animation
from matplotlib import gridspec

from numpy import array, zeros, cos, sin, arange
from math import *
from mlp import *

MAX_ROTATION    = pi*6
MIN_ROTATION    = pi/2.0
MAX_DROTATION   = 20
MIN_DROTATION   = -20

MAX_TRANSLATION = 15
MIN_TRANSLATION = -15
MAX_DTRANSLATION = 13
MIN_DTRANSLATION = -13

MAX_CONTROL     = 10

class InvertedPendulum:
    def __init__(self, rotational=array([pi/2 + 0.2, 0]), translational=array([0,0])):
        self.time = 0.0
        self.ctrl = 0.0
        self.rot = rotational
        self.trans = array([0.0, 0.0])

        self.LENGTH = 0.5
        self.GRAVITY = 9.81
        self.MASS = 0.4
        self.DT = 0.01

    def sysEq(self, y):
        theta = y[0]
        omega = y[1]
        dOmega = 1 / self.LENGTH * \
                 ( \
                    -self.GRAVITY * cos(theta) \
                    -(self.ctrl / self.MASS) * sin(theta) - 0.01 * omega \
                 )
        return array([omega, dOmega])

    def rk4(self, y):
        k1 = self.sysEq(y)
        k2 = self.sysEq(y + self.DT / 2.0 * k1)
        k3 = self.sysEq(y + self.DT / 2.0 * k2)
        k4 = self.sysEq(y + self.DT * k3)

        return y + (self.DT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def update(self, ctrlInp):
        self.time += self.DT
        self.ctrl = clip(-20.0, ctrlInp, 20.0)

        self.trans[0] = clip(-1.0e300, self.trans[0], 1.0e300)
        self.trans[1] = clip(-1.0e300, self.trans[1], 1.0e300)

        self.trans[1] += (self.ctrl / self.MASS) * self.DT
        self.trans[0] += self.trans[1] * self.DT

        self.rot = self.rk4(self.rot)

    def simulateStep(self):
        self.update(self.ctrl)

    def simulateSteps(self, steps):
        x = []
        direction = 0
        time = zeros(steps)
        time_step = 2.0 / steps

        x = []

        for i in range(steps):
            self.update(random.uniform(-10.0, 10.0))
            x.append((self.rot[0], self.trans[0]))

        return array(x)

    def setControl(self, ctrl):
        self.ctrl = ctrl

    def getScaledState(self):
        return [
            normalize(self.trans[1], MIN_TRANSLATION, MAX_TRANSLATION),
            normalize(self.trans[0], MIN_DTRANSLATION, MAX_DTRANSLATION),
            normalize(self.rot[0], MIN_ROTATION, MAX_ROTATION),
            normalize(self.rot[1], MIN_DROTATION, MAX_DROTATION)
        ]

    def simulateStepsControl(self, steps, mlp):
        x = []
        direction = 0
        time = zeros(steps)
        time_step = 2.0 / steps

        forces = []

        pctrl = 0.0
        history = []

        for i in range(steps):

            if i > 4:
                pinp = zeros(24)
                pinp[0] = normalize(self.rot[0], MIN_ROTATION, MAX_ROTATION)
                pinp[1] = normalize(self.rot[1], MIN_DROTATION, MAX_DROTATION)
                pinp[2] = normalize(self.trans[0], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[3] = normalize(self.trans[1], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[4] = normalize(history[i-1][0], MIN_ROTATION, MAX_ROTATION)
                pinp[5] = normalize(history[i-1][1], MIN_DROTATION, MAX_DROTATION)
                pinp[6] = normalize(history[i-1][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[7] = normalize(history[i-1][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[8] = normalize(history[i-2][0], MIN_ROTATION, MAX_ROTATION)
                pinp[9] = normalize(history[i-2][1], MIN_DROTATION, MAX_DROTATION)
                pinp[10] = normalize(history[i-2][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[11] = normalize(history[i-2][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[12] = normalize(history[i-3][0], MIN_ROTATION, MAX_ROTATION)
                pinp[13] = normalize(history[i-3][1], MIN_DROTATION, MAX_DROTATION)
                pinp[14] = normalize(history[i-3][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[15] = normalize(history[i-3][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[16] = normalize(history[i-4][0], MIN_ROTATION, MAX_ROTATION)
                pinp[17] = normalize(history[i-4][1], MIN_DROTATION, MAX_DROTATION)
                pinp[18] = normalize(history[i-4][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[19] = normalize(history[i-4][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[20] = normalize(history[i-5][0], MIN_ROTATION, MAX_ROTATION)
                pinp[21] = normalize(history[i-5][1], MIN_DROTATION, MAX_DROTATION)
                pinp[22] = normalize(history[i-5][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[23] = normalize(history[i-5][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                mlp.aups(pinp.reshape(24,1))
                pctrl = mlp.output()[0,0]
                # pctrl *= MAX_CONTROL

            self.update(pctrl)
            x.append((self.rot[0], self.trans[0]))

            forces.append(pctrl)

            history.append([
                self.rot[0],
                self.rot[1],
                self.trans[0],
                self.trans[1]
            ])

        print "max force: ",max(forces), " min: ",min(forces)

        return array(x)

    def renderSim(self, x):
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

            dx1 = x[i][1] + self.LENGTH * cos(x[i][0])
            dy1 = self.LENGTH * sin(x[i][0])

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
    #invp = InvertedPendulum()
    #states = invp.simulateSteps(500)
    #invp.renderSim(states)

    wf = pickle.loads(open("output/weights_" + argv[0] + ".txt",'r').read())
    bf = pickle.loads(open("output/bias_" + argv[0] + ".txt",'r').read())

    mlp = multilayer_perceptron(24, [24, 12, 12, 1], ["tansig","tansig","tansig","purelin"])
    mlp.genWeightBias(2000)
    mlp.weights = wf
    mlp.bias = bf

    invp = InvertedPendulum()
    states = invp.simulateStepsControl(500, mlp)
    invp.renderSim(states)

def usage():
    print __doc__

if __name__ == "__main__":
    main(sys.argv[1:])