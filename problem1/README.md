# Problem 1: SIRENs

In this problem, you will learn the theory underlying Sine Representation Networks (SIRENs), and implement the model yourself. This exercise will serve as an introduction to both neural fields and computer vision research, as you will implement the model (along with many baselines) from scratch.

## Part 1: Understanding the theory

For this part, start by reading the SIRENs paper:
> [*Implicit Neural Representations with Periodic Activation Functions*](https://www.google.com/search?client=firefox-b-1-d&q=Implicit+Neural+Representations+with+Periodic+Activation+Functions) by Sitzmann and Martel et al., NeurIPS 2020

Next, answer all the multiple choice questions in `multiple_choice.yml` to test your comprehension.

## Part 2: Implementing SIRENs (and an MLP baseline)

### Instructions

Using your knowledge from the paper, do the following:

- Implement all missing functions in the `problem_1_*.py` files
  - Implement a SIREN (`problem_1_siren.py`)
  - Implement a bog-standard feed-forward MLP (`problem_1_mlp.py`)
  - Implement derivative-calculating functions (`problem_1_gradient.py`)
  - Implement a flexible MLP/SIREN training loop (`problem_1_train.py`)
- Run the notebook `benchmark.ipynb` to check your functions and visualize the training process

### Short-answer questions

Answer these questions in a file called `problem_1.pdf`:

- Why are the MLP reconstructions so much less detailed than those produced by the SIREN?
- The image Laplacian produced by the MLP looks strange... what's happening here and why?

## Part 3: Eeking out better performance!

While the reconstruction from our SIREN looks a lot better than the MLP's, it's still not great (especially compared to the results shown in the paper -- notice those weird blobby artifacts!). In this part, you'll experiment with hyperparameters in the models you've already written to achieve better reconstructions with SIREN.

### Specific deliverables

Include plots and analyses for the following investigations in `problem_1.pdf`:

- Benchmark at least 10 additional SIREN/MLP models by varying multiple hyperparameters (e.g., activation functions, model hyperparameters, training parameters). Plot the results of these benchmarks and include in the writeup.
- Optimize the hyperparameters in the SIREN model to achieve the highest PSNR you can on the `astronaut` image. Include the optimal hyperparameters in your writeup.

**For extra credit:** implement another (nontrivial) model from the literature and add it to your benchmarks! If you do this, please describe the model you implemented in your writeup.


## Submission Instructions

Submit the following files:

- `multiple_choice.yml`
- `problem_1_gradients.py`
- `problem_1_mlp.py`
- `problem_1_siren.py`
- `problem_1_train.py`
- `problem_1.pdf`
