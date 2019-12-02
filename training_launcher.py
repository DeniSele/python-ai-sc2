import sys
import subprocess
import numpy
import evo_algorithm
import neural_network
import pickle


# When you start for the first time or want to start all over again:
#   - Set USE_SAVED_WEIGHTS to False
#   - Set LOAD_GENERATION_NUMBER to 0
#
# To use existing data:
#   - Set USE_SAVED_WEIGHTS to True
#   - Set LOAD_GENERATION_NUMBER to N
#   N - number of generation files to be used
USE_SAVED_WEIGHTS = True
LOAD_GENERATION_NUMBER = 1

# Population size
sol_per_pop = 4

# Half the population to break it up into pieces
half_part_of_sol = 2

# Mating pool size (number of parents)
num_parents_mating = 2

# Number of generations
num_generations = 2

# Mutation percent
mutation_percent = 10

# The number of neurons in the input layer
input_size = 4


initial_pop_weights = []
if not USE_SAVED_WEIGHTS:
    # Creating the initial population.

    print("Create a new population")
    for curr_sol in numpy.arange(0, sol_per_pop):
        HL1_neurons = 6
        input_HL1_weights = numpy.random.uniform(
            low=-0.1, high=0.1, size=(input_size, HL1_neurons))
        HL2_neurons = 5
        HL1_HL2_weights = numpy.random.uniform(
            low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))
        output_neurons = 3
        HL2_output_weights = numpy.random.uniform(
            low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))

        initial_pop_weights.append(
            numpy.array(
                [input_HL1_weights, HL1_HL2_weights, HL2_output_weights]))
else:
    # Get the latest saved version of the data, based on LOAD_GENERATION_NUMBER
    for curr_sol in numpy.arange(0, sol_per_pop):
        f = open("weights\\generation_" + str(LOAD_GENERATION_NUMBER) +
                 "_weights_" + str(curr_sol) + ".pkl", "rb")
        new_weight = pickle.load(f)
        initial_pop_weights.append(new_weight)
        f.close()

pop_weights_mat = numpy.array(initial_pop_weights)
pop_weights_vector = evo_algorithm.mat_to_vector(pop_weights_mat)

for generation in range(LOAD_GENERATION_NUMBER,
                        num_generations + LOAD_GENERATION_NUMBER):
    print("Generation: ", generation)

    # Converting the solutions from being vectors to matrices.
    pop_weights_mat = evo_algorithm.vector_to_mat(pop_weights_vector, pop_weights_mat)

    # Measuring the fitness of each chromosome in the population.
    fitness = [0] * sol_per_pop

    # Split the launch in half to provide enough power
    for j in range(2):
        procs = []
        for i in range(j * half_part_of_sol, half_part_of_sol * (j + 1)):
            print("i: ", i)
            f = open("weights\\generation_" + str(generation) +
                     "_weights_" +  str(i) + ".pkl", "wb")
            pickle.dump(pop_weights_mat[i], f)
            f.close()
            proc = subprocess.Popen([
                sys.executable, 
                'bots_controller.py', 
                str(i), 
                str(generation)
                ])
            procs.append(proc)

        for proc in procs:
            proc.wait()

    # Fitness calculation based on results
    for i in range(sol_per_pop):
        f = open("results\\generation_" + str(generation) + "_player_" +
                 str(i) + ".txt", "r")
        result_str = f.readline()
        f.close()
        
        if result_str == "Result.Victory\n":
            fitness[i] = 1
        else:
            fitness[i] = 0

    print("Fitness")
    print(fitness)

    # Selecting the best parents in the population for mating.
    parents = evo_algorithm.select_mating_pool(pop_weights_vector, fitness.copy(),
                                    num_parents_mating)

    # Generating next generation using crossover.
    offspring_crossover = evo_algorithm.crossover(
        parents,
        offspring_size=(pop_weights_vector.shape[0] - parents.shape[0],
                        pop_weights_vector.shape[1]))

    # Adding some variations to the offsrping using mutation.
    offspring_mutation = evo_algorithm.mutation(
        offspring_crossover, mutation_percent=mutation_percent)

    # Creating the new population based on the parents and offspring.
    pop_weights_vector[0:parents.shape[0], :] = parents
    pop_weights_vector[parents.shape[0]:, :] = offspring_mutation

# Saving the final results for this launch
print("Final weights for generation",
      (num_generations + LOAD_GENERATION_NUMBER), "saving...")
for i in range(0, sol_per_pop):
    print("saved weight: ", i)
    f = open(
        "weights\\generation_" + str(num_generations + LOAD_GENERATION_NUMBER)
        + "_weights_" + str(i) + ".pkl", "wb")
    pickle.dump(pop_weights_mat[i], f)
    f.close()
