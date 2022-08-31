from env import *
from model import *
from numpy import *

from sklearn.preprocessing import MinMaxScaler

# simulation parameters
ITERATIONS = 500
MAX_FORCE = 10

# genetic algorithm parameters
POPULATION_SIZE = 50
EPOCHS = 1000
ELITISM = 10
SELECTION_RATE = math.floor(POPULATION_SIZE*0.75)
MUTATION_RATE = 0.05
MIN_WEIGHT = -1.00
MAX_WEIGHT = 1.00

# neural network parameters
INPUT_LAYER = {
    "input_size": 24,
    "output_size": 24
}
HIDDEN_LAYER = [
    12,
    12
]
INPUT_ACTIVATION = 'tanh'
HIDDEN_ACTIVATION = 'tanh'

MAX_ROTATION = 2.0*pi
MIN_ROTATION = -2.0*pi
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


def crossover(a, b):
    child = Model(
        input_layer=INPUT_LAYER,
        hidden_layer=HIDDEN_LAYER,
        input_activation=INPUT_ACTIVATION,
        hidden_activation=HIDDEN_ACTIVATION
    )

    w0 = child.get_weights()
    w1 = a.get_weights()
    w2 = b.get_weights()

    for i in range(len(w1)):
        for j in range(len(w1[i])):
            for k in range(len(w1[i][j])):
                if type(w1[i][j][k]) == np.ndarray:
                    for v in range(len(w1[i][j][k])):
                        if random.random() < 0.5:
                            w0[i][j][k][v] = w1[i][j][k][v]
                        else:
                            w0[i][j][k][v] = w2[i][j][k][v]
                else:
                    if random.random() < 0.5:
                        w0[i][j][k] = w1[i][j][k]
                    else:
                        w0[i][j][k] = w2[i][j][k]

    child.set_weights(w0)
    return child


def mutate(a):
    w = a.get_weights()
    for v in range(len(w)):
        i_w = v
        for i in range(len(w[i_w])):
            for j in range(len(w[i_w][i])):

                if type(w[i_w][i][j]) == np.ndarray:
                    for n in range(len(w[i_w][i][j])):
                        if random.random() < MUTATION_RATE:
                            w[i_w][i][j][n] += random.uniform(-0.1, 0.1)
                else:
                    if random.random() < MUTATION_RATE:
                        w[i_w][i][j] += random.uniform(-0.1, 0.1)

        a.set_weights(w)

    return a


def selection(population):
    new_generation = []

    for i in range(ELITISM):
        new_generation.append(population[i])

    for i in range(POPULATION_SIZE - ELITISM - 1):
        p1 = random.randint(0, SELECTION_RATE)
        p2 = random.randint(0, SELECTION_RATE)

        child = crossover(population[p1], population[p2])
        child = mutate(child)

        new_generation.append(child)

    return new_generation


def sort_by_fitness(pair):
    return sorted(pair, key=lambda x: x[0])


def test_individual(controller, pendulum, steps):

    errors = []
    for i in range(steps):
        # control_input = pendulum.get_current_state().reshape((1, -1))
        if i > 4:
            control_input = np.append(pendulum.get_current_state().reshape((1, -1)), pendulum.get_states(4).reshape((5, -1)), axis=0)
            control_input = scaler.transform(control_input).reshape((1, -1))

            control_output = controller.predict(control_input)[0][0]
            control_output = clip(-MAX_FORCE, control_output, MAX_FORCE)

            pendulum.control = control_output
            pendulum.update_state()

        rot_error = (pi - pendulum.rotation[0])**2
        trans_error = pendulum.translation[0]**2

        if abs(pendulum.rotation[0] - pi) > pi/2.0:
            errors = (sum(errors) / steps) + 1e6
            break

        if abs(pendulum.translation[0]) > 5:
            errors = (sum(errors) / steps) + 1e6
            break

        error = rot_error + trans_error
        errors.append(error)
        if i == (steps - 1):
            errors = sum(errors) / ITERATIONS
            break

    return [errors, controller, pendulum]


def test_population(population, pendulums):
    scores = []
    for i in range(len(population)):
        scores.append(test_individual(population[i], pendulums[i], ITERATIONS))

    sorted_population = sort_by_fitness(scores)
    return sorted_population


def populate():
    generation = []
    for x in range(POPULATION_SIZE):
        individual = Model(
            input_layer=INPUT_LAYER,
            hidden_layer=HIDDEN_LAYER,
            input_activation=INPUT_ACTIVATION,
            hidden_activation=HIDDEN_ACTIVATION
        )
        generation.append(individual)

    return generation


def generate_pendulum(rotation, translation, n_pendulum=1):
    pendulums = []
    for x in range(n_pendulum):
        pendulums.append(InvertedPendulum(rotation, translation))

    return pendulums


def save_model(model, epoch):
    model.save("models/" + str(epoch))


def train_system(save_best=True):
    population = populate()

    if not os.path.exists("models"):
        os.makedirs("models")

    for x in range(EPOCHS):
        init_rotation = array(
            [
                # pi + random.uniform(-0.1, 0.1),
                pi + (0.1 * random.randn() + 0.0),
                0.00 #random.uniform(0.01, 0.01)
            ]
        )
        init_translation = array(
            [
                0.00,  # random.uniform(0.0, 0.0),
                0.00 # random.uniform(0.00, 0.00)
            ]
        )

        pendulums = generate_pendulum(init_rotation, init_translation, POPULATION_SIZE)
        sorted_population = test_population(population, pendulums)

        if save_best:
            save_model(sorted_population[0][1], x)

        population = selection(list(map(lambda a: a[1], sorted_population)))
        fitnesses = list(map(lambda a: a[0], sorted_population))
        print("---"*20)
        print("best fitness: ", sorted_population[0][0])
        print("mean fitness: ", mean(fitnesses))
        print("progress: " + "{:.3f}".format(100.0 * float(x)/float(EPOCHS)) + "%")


if __name__ == "__main__":
    train_system()
