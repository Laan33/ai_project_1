# Genetic Algorithm for Travelling Salesman Problem

This project implements a genetic algorithm to solve the Travelling Salesman Problem (TSP). The algorithm includes various crossover and mutation methods, and supports grid search for hyperparameter optimization.

## Requirements

- Python 3.x
- numpy
- matplotlib

## Setup

1. Clone the repository:
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. Install the required packages:
    ```sh
    pip install numpy matplotlib
    ```

## Usage

1. **Run the Genetic Algorithm**:
    ```sh
    python main.py
    ```

2. **Grid Search for Hyperparameter Optimization**:
    The script will automatically perform a grid search over predefined mutation rates, crossover rates, and population sizes. The results will be saved to a CSV file in the [results] directory.

3. **View Best Result**:
    The best result from the grid search will be printed and plotted.

## Configuration

- **Datasets**:
    - The dataset files should be placed in the [datasets] directory.
    - Update the [filename] variable in [main.py] to select the dataset.

- **Hyperparameters**:
    - Modify the [MUTATION_RATES], [CROSSOVER_RATES], and [POPULATION_SIZES] lists in [main.py] to change the grid search parameters.

## Example

To run the algorithm with the `berlin52` dataset:
```sh
python main.py