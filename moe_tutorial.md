## Preview

In this example we will solve a MOE problem using our library.

## Setting the experiment

First, we need to import the components we want to use in our experiment. If you are not familiar with the basics of evolutionary algorithms, here's a [short intro](https://drive.google.com/file/d/0B6G3tbmMcpR4WVBTeDhKa3NtQjg/view?resourcekey=0-zLNbQBpLQ7jC_HVVQGLrzA).

```python
from eckity.creators.ga_creators.simple_vector_creator import GAVectorCreator
from eckity.multi_objective_evolution.moe_evolution import MOEvolution
from eckity.multi_objective_evolution.moe_breeder import MOEBreeder
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.genetic_operators.crossovers.vector_k_point_crossover import VectorKPointsCrossover
from eckity.genetic_operators.mutations.vector_random_mutation import FloatVectorUniformNPointMutation
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.multi_objective_evolution.moe_fitness import MOEFitness
from eckity.multi_objective_evolution.moe_plot import MOEPlot
from eckity.population import Population
from eckity.statistics.minimal_print_statistics import MinimalPrintStatistics
from eckity.subpopulation import Subpopulation
from eckity.genetic_encodings.ga.float_vector import FloatVector
from eckity.multi_objective_evolution.crowding_termination_checker import CrowdingTerminationChecker
```

Now we can create our experiment.

## expirament goal
in the following expirament we will try to 

### Initializing the multi-objective evolutionary algorithm

The Evolution object is the main part of the experiment. It receives the parameters of the experiment and runs the evolutionary process:

```python
	algo = MOEvolution(
		Population([Subpopulation(
			creators=GAVectorCreator(length=3, bounds=(-4, 4), fitness_type=MOEFitness, vector_type=FloatVector),
			population_size=150,
			# user-defined fitness evaluation method
			evaluator=MOEBasicTestEvaluator(),
			# maximization problem (fitness is sum of values), so higher fitness is better
			higher_is_better=False,
			elitism_rate=1 / 300,
			# genetic operators sequence to be applied in each generation
			operators_sequence=[
				VectorKPointsCrossover(probability=0.7, k=1),
				FloatVectorUniformNPointMutation(probability=0.3, n=3)  # maybe chnge mutation
			],
			selection_methods=[
				# (selection method, selection probability) tuple
				(TournamentSelection(tournament_size=3, higher_is_better=True), 1)
			]
		)]),
		breeder=MOEBreeder(),
		max_workers=4,
		max_generation=150,

		termination_checker=CrowdingTerminationChecker(0.01),
		statistics=MinimalPrintStatistics()
	)

```

Let's break down the parts and understand their meaning.

### Initializing Subpopulation

Sub-population is the object that holds the individuals, and the objects that are responsible for treating them (a Population can include a list of multiple Subpopulations but it is not needed in this case).

### Creating individuals

We have chosen to use bit vectors with a length equal to the number of entries we want to fill. We chose our population to contain 300 individuals.

```python
algo = SimpleEvolution(
        Subpopulation(creators=GABitStringVectorCreator(length=100),
                      population_size=300,
```

### Evaluating individuals

Next we set the parameters for evaluating the individuals. We will elaborate on this later on.

```python
                      # user-defined fitness evaluation method
                      evaluator=OneMaxEvaluator(),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
```

### Breeding process

Now we will set the (hyper)parameters for the breeding process. We declared that one of the individuals will be considered as elite (and pass to the next generation without a change).

Then, we defined the genetic operators to be applied in each generation:

-   1 Point Crossover with a probability of 50%
-   Bit Flip Mutation with a probability of 20%, and on every entry of a chosen individual we flip the bit with a probability of 5% (probabilities don't sum to 1 since operators are simply applied sequentially, in a pipeline manner)
-   Tournament Selection with a probability of 1 and with tournament size of 3

```python
                      elitism_rate=1/300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          VectorKPointsCrossover(probability=0.5, k=1),
                          BitStringVectorNFlipMutation(probability=0.2, probability_for_each=0.05, n=100)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
                      ]),
```

Now that we are done with our Subpopulation, we will finish setting the evolutionary algorithm.

We define our breeder to be the standard simple Breeder (appropriate for the simple case of a single sub-population), and our max number of worker nodes to compute the fitness values is 4.

```python
        breeder=SimpleBreeder(),
        max_workers=4,
```

### Termination condition and statistics

We define max number of generations (iterations). We define a `TerminationChecker` (early termination mechanism) to be with a Threshold of 0 from 100 (which means fitness 100 - optimal fitness for our problem)

```python
        termination_checker=ThresholdFromTargetTerminationChecker(optimal=100, threshold=0),
        statistics=BestAverageWorstStatistics()
    )
```

Finally, we set our statistics to be the default form of best-average-worst statistics which prints the next format in each generation of the evolutionary run:

```
generation #(generation number)
subpopulation #(subpopulation number)
best fitness (some fitness which is the best)
worst fitness (some fitness which is average)
average fitness (some fitness which is just the worst)
```

Another possible keyword argument to the program is a seed value. This enables to replicate results across different runs with the same parameters.

## Evolution stage

After setting up the evolutionary algorithm, we can finally begin the run:

```python
    # evolve the generated initial population
    algo.evolve()
```

## Execution stage

After the evolve stage has finished (by exceeding the maximal number of generations), we can execute the Algorithm and show the best-of-run vector (solution) to check the evolutionary results:

```python
    # Execute (show) the best solution
    print(algo.execute())
```

## The one Max Evaluator

## What is an Evaluator?

Problem-specific fitness evaluation method.

For each problem we need to supply a mechanism to compute an individual's fitness score, which determines how "good" (fit) this particular individual is.

Let's go through the parts of the one max problem Evaluator:

Simple version evaluator (SimpleIndividualEvaluator) sub-classes compute fitness scores of each individual separately, while more complex Individual Evaluators may compute the fitness of several individuals at once.

## Defining the One Max Evaluator

Our evaluator extends the SimpleIndividualEvaluator class, thus computing each individual's fitness separately. This evaluator summs the values if the vector.

```python
class OneMaxEvaluator(SimpleIndividualEvaluator):
    def _evaluate_individual(self, individual):
        return sum(individual.vector)
```