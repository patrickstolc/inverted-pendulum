import random
import sys
import os
import traceback
import multiprocessing
import pickle

from mlp import *
from _pendulum import *
from numpy import *

POPULATION_SIZE = 100
ELITISM         = 10
MUTATION_RATE   = 0.50

NEURON_COUNT    = [24, 12, 1]
ACTIVE_FUNCS    = ["tansig","tansig","purelin"]
ITERATIONS      = 500
WEIGHT_RANGE    = 4
EPOCHS          = 1000
POOLS           = multiprocessing.cpu_count()

MAX_ROTATION    = pi*1.5
MIN_ROTATION    = pi/2.0
MAX_DROTATION   = 10
MIN_DROTATION   = -10

MAX_TRANSLATION = 5
MIN_TRANSLATION = -5
MAX_DTRANSLATION = 3
MIN_DTRANSLATION = -3

MAX_CONTROL     = 20

NUM_INPUTS		= 24

def crossover(perceptronA, perceptronB):
    weightA = perceptronA.weights
    biasA = perceptronA.bias

    weightB = perceptronB.weights
    biasB = perceptronB.bias

    newWeight = []
    newBias = []

    for i in range(len(weightA)):
        if random.random() < 0.5:
            newWeight.append(weightA[i])
            newBias.append(biasA[i])
        else:
            newWeight.append(weightB[i])
            newBias.append(biasB[i])

        child = multilayer_perceptron(NUM_INPUTS, NEURON_COUNT, ACTIVE_FUNCS)
        child.genWeightBias(WEIGHT_RANGE)
        child.weights = newWeight
        child.bias = newBias

    return child

def mutate(perceptron):
    if random.random() < MUTATION_RATE:
        if random.random() < 0.5:
            perceptron.weights[random.randint(0, (len(perceptron.weights)-1))] += random.uniform(-0.1, 0.1)
        else:
            perceptron.bias[random.randint(0, (len(perceptron.bias)-1))] += random.uniform(-0.1, 0.1)

    return perceptron

def selection(L):
    newGeneration = [L[0]]

    for i in range(ELITISM):
        newGeneration.append(L[i])

    for i in range(POPULATION_SIZE - ELITISM - 1):
        p1 = L[random.randint(0, len(L)-1)]
        p2 = L[random.randint(0, len(L)-1)]

        child = crossover(p1,p2)
        child = mutate(child)

        newGeneration.append(child)

    return newGeneration

def sortByFitness(Lpair):
    return sorted(Lpair, cmp = lambda x, y : 1 if x[0] > y[0] else -1)

def testIndividual((perceptron, pendulum, steps)):
    errors = []

    try:

        history = []

        for x in range(steps):

            pctrl = 0.0
            pinp = zeros(24)

            if x > 4:

                pinp[0] = normalize(pendulum.rot[0], MIN_ROTATION, MAX_ROTATION)
                pinp[1] = normalize(pendulum.rot[1], MIN_DROTATION, MAX_DROTATION)
                pinp[2] = normalize(pendulum.trans[0], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[3] = normalize(pendulum.trans[1], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[4] = normalize(history[x-1][0], MIN_ROTATION, MAX_ROTATION)
                pinp[5] = normalize(history[x-1][1], MIN_DROTATION, MAX_DROTATION)
                pinp[6] = normalize(history[x-1][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[7] = normalize(history[x-1][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[8] = normalize(history[x-2][0], MIN_ROTATION, MAX_ROTATION)
                pinp[9] = normalize(history[x-2][1], MIN_DROTATION, MAX_DROTATION)
                pinp[10] = normalize(history[x-2][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[11] = normalize(history[x-2][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[12] = normalize(history[x-3][0], MIN_ROTATION, MAX_ROTATION)
                pinp[13] = normalize(history[x-3][1], MIN_DROTATION, MAX_DROTATION)
                pinp[14] = normalize(history[x-3][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[15] = normalize(history[x-3][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[16] = normalize(history[x-4][0], MIN_ROTATION, MAX_ROTATION)
                pinp[17] = normalize(history[x-4][1], MIN_DROTATION, MAX_DROTATION)
                pinp[18] = normalize(history[x-4][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[19] = normalize(history[x-4][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                pinp[20] = normalize(history[x-5][0], MIN_ROTATION, MAX_ROTATION)
                pinp[21] = normalize(history[x-5][1], MIN_DROTATION, MAX_DROTATION)
                pinp[22] = normalize(history[x-5][2], MIN_TRANSLATION, MAX_TRANSLATION)
                pinp[23] = normalize(history[x-5][3], MIN_DTRANSLATION, MAX_DTRANSLATION)

                perceptron.aups(pinp.reshape(24,1))
                pctrl = perceptron.output()[0,0]

                pendulum.update(pctrl)

            rotError = (pi - pendulum.rot[0])**2 + (pendulum.rot[1])**2
            transError = pendulum.trans[0]**2 + (pendulum.trans[1])**2

            if abs(pendulum.rot[0] - pi) > pi/2.0:
                rotError = 1e9

            if abs(pendulum.trans[0]) > 2:
                transError =  1e4

            error = rotError + transError
            errors.append(error)

            history.append([pendulum.rot[0], pendulum.rot[1], pendulum.trans[0], pendulum.trans[1]])
    except:
        print('caught exception in wt (x = %d):' % x)
        traceback.print_exc()
        print()
        raise e

    return [(sum(errors)/ITERATIONS), perceptron, pendulum]

def testPopulation(population, pendulumLeft, pendulumRight):
    p = multiprocessing.Pool(POOLS)
    argsLeft = []
    argsRight = []
    pairs = []

    initialRotation = pendulumLeft[0].rot

    for x in range(POPULATION_SIZE):
        argsLeft.append((population[x], pendulumLeft[x], ITERATIONS))

    for x in range(POPULATION_SIZE):
        argsRight.append((population[x], pendulumRight[x], ITERATIONS))

    Lpairs = p.map(testIndividual, argsLeft)
    Rpairs = p.map(testIndividual, argsRight)

    for x in range(POPULATION_SIZE):
        pairs.append( [ Lpairs[x][0] + Rpairs[x][0], Lpairs[x][1], Lpairs[x][2], Rpairs[x][2] ] )

    sortedPairs = sortByFitness(pairs)
    p.close();

    print("best fitness: " + "{:.3f}".format(sortedPairs[0][0]))

    return map(lambda x : x[1], sortedPairs)

def populate():
    generation = []
    for x in range(POPULATION_SIZE):
        org = multilayer_perceptron(NUM_INPUTS, NEURON_COUNT, ACTIVE_FUNCS)
        org.genWeightBias(WEIGHT_RANGE)
        generation.append(org)

    return generation

def generatePendulums(rot, trans):
    pendulums = []
    for x in range(POPULATION_SIZE):
        pendulums.append(InvertedPendulum(rot, trans))

    return pendulums

def writeOut(best, n):

    weightsFile = open("output/weights_" + n + ".txt",'w')
    sw = pickle.dumps(best.weights, protocol=0)
    weightsFile.write(sw);
    weightsFile.close()

    biasFile = open("output/bias_" + n + ".txt", 'w')
    sb = pickle.dumps(best.bias, protocol=0)
    biasFile.write(sb)
    biasFile.close()

def trainSystemNetwork():
    population = populate()
    best = None

    if not os.path.exists("output"):
        os.makedirs("output")

    for x in range(EPOCHS):

        thetaChoices = [random.uniform(-0.1,-0.2), random.uniform(0.1, 0.2)]

        initRL = array([(pi) + random.uniform(-0.1, 0.2), random.uniform(-0.1, 0.1)])
        initRR = array([(pi) + random.uniform(-0.1, 0.2), random.uniform(-0.1, 0.1)])
        initT  = array([random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)])

        pendulumLeft = generatePendulums(initRL, initT)
        pendulumRight = generatePendulums(initRR, initT)

        sortedPopulation = testPopulation(population, pendulumLeft, pendulumRight)

        if x % 1 == 0:
            writeOut(sortedPopulation[0], str(x))

        if x == EPOCHS - 1:
            best = sortedPopulation[0]

        population = selection(sortedPopulation)
        print("progress: " + "{:.3f}".format(100.0 * float(x)/float(EPOCHS)) + "%")

    writeOut(best, "")

def normalize(value, min, max):
    return ((1 - (-1))/(max - min)) * (value - max) + 1

if __name__ == "__main__":
    trainSystemNetwork()