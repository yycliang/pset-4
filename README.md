- Run the notebook `benchmark.ipynb` to check your functions and visualize the training process
`pytest .`
`pytest . -k "not initialization"`.
___
## Part 3: better performance!

 In problem_1.pdf`:

- Benchmark at least 10 additional SIREN/MLP models by varying multiple hyperparameters (e.g., activation functions, model hyperparameters, training parameters). Plot the results of these benchmarks and include in the writeup.
- Optimize the hyperparameters in the SIREN model to achieve the highest PSNR you can on the `astronaut` image. Include the optimal hyperparameters in your writeup.

Your visualization should look like what is in `should_look_like/` (not in the minutiae, but it should display all of your `problem1_gradients.py` implementations over the course of training along with the image as output by your model based on the coordinates, as well as a PSNR curve comparing different outputs).

## Problem 2

## Part A: Camera Rays

## Part B: Sphere Tracing

Implement the `sphere_tracing` function, which should:
1. Initialize each ray with a starting **t** value 
2. March along each ray by querying the SDF at the current point
3. For each step, advance the ray by a distance equal to the SDF value (this is safe as the SDF tells us the distance to the closest surface)
4. Stop marching when either:
   - The ray hits the surface (SDF < epsilon)
   - The maximum number of iterations is reached
   - The ray goes too far (beyond **t_far**)
5. For each ray that hits a surface, record the color at the hit point

## Hints

2. **Implementation Steps**:
   - Initialize a tensor `t` for each ray's current position
   - Track which rays have hit a surface with a boolean mask
   - For each iteration:
     - Sample points using the current **t** values: `points = origins + t * directions`
     - Query the SDF model to get distance values and colors
     - Determine which rays hit a surface (SDF < epsilon)
     - Record colors for rays that hit
     - For rays that haven't hit yet, advance t by the SDF value (optionally using a relaxation factor like 0.5)

3. **Batched Processing**: Process all rays in parallel for efficiency.

4. **Error Checking**: Handle cases where rays might never hit any surface.

### Testing

python problem2/problem_2_sphere_tracing.py

`expected_renders_for_debug/sphere_tracing.png`.

## Problem 2.2: Differentiable Volume Rendering

#### Part A: Camera Ray Generation

Reuse your implementation of `camera_param_to_rays` from the sphere tracing problem. 

#### Part B: Ray Marching
- Generate `num_samples` points along each ray between `t_near` and `t_far`.
- The sample points are defined as: `point = origin + t * direction` for each value of `t`.
- Return an array of sampled points with shape `[H, W, num_samples, 3]`.

#### Part C: Volumetric Rendering
- Implement the classical volume rendering equation: 
  C = ∑ T_i * α_i * c_i, where:
  - T_i = ∏_{j=1}^{i-1} (1 - α_j) is the accumulated transmittance
  - α_i = 1 - exp(-σ_i * δ_i) is the alpha value at sample i
  - σ_i is the density at sample i
  - δ_i is the distance between adjacent samples
  - c_i is the color at sample i
For more details elaboration on this equation, please refer to NeRF's original paper: https://arxiv.org/abs/2003.08934 (Eqs. 1, 2, 3, and 5 would be helpful).

Hint: Iterate through samples front-to-back (from **t_near** to **t_far**), accumulating color and updating transmittance.

### Testing
```
python problem2/problem_2_volumetric_rendering.py
```
`problem2/expected_renders_for_debug/volume_rendering.png`.

### Please answer the following questions in problem_2.pdf

# Problem-3 Playing with a NeRF

#### Positional Encodings

Please fill in several lines of code in  `src/components/positional_encoding.py` to implement a Positional Encoding described in Equation (4) of paper:  [Neural Radiance Fields, Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934).  Attach the filled-in code in the submission report.

You need also to copy the SineLayer implementation in problem-1 to `problem3/src/components/sine_layer.py`. 

After your implementation, please run to test it.

`python -m tests.test_positional_encoding`

Neural fields can be categorized into two types—explicit and implicit—as follows:

- **Implicit:** Coordinates are mapped to values via neural networks.
- **Explicit:** Coordinates are mapped to values via spatial data structures (grids, octrees, point clouds, etc.).

We will play with an implicit one in this problem. It’s already implemented in `src/field/field_mlp.py`. Please read it. 

## Training Neural Radiance Fields (NeRFs)

In this part of the assignment, you’ll train a neural radiance field (NeRF), mostly as described in [Neural Radiance Fields, Mildenhall et al., 2020](https://arxiv.org/abs/2003.08934). 

Since you have implemented basic volumetric rendering in Problem-2, you don’t need to implement it again. It's mostly done in `src/nerf.py`, but there is one line that needs to be filled in. Copy the line you added into the final report. 

Please also read the`src/nerf.py` code and answer the question:

The rendering model used in our code is slightly different from the one used in the original NeRF paper. Please identify at least one difference. (Not the size of the model, not the initialization, etc. Hint: look at rendering and ray sampling!) 

## Training

`python -m scripts.train_nerf`

Attach at least two final images from the `outputs/spin` folder in the final report. It’s fine if your final images have blurred results. 

I put three resulting images in `expected_results/` for your reference. They were obtained after 10 minutes of training. 

## Submission Instructions

for problem 1
- `multiple_choice.yml`
- `problem_1_gradients.py`
- `problem_1_mlp.py`
- `problem_1_siren.py`
- `problem_1_train.py`
- `problem_1.pdf`

problem 2

1. For the code, submit the two Python files. Also, include the edited code in the report PDF file. (either paste formatted code or screenshot)

2. Please submit your rendered images for both problems in the report pdf file. Please make sure that you used the intrinsics corresponding to the `submit` mode.

3. For the questions, write in English and submit the report PDF file.

For Problem-3, you don’t need to submit code. Please ensure you have passed the provided test for positional encoding. You only need to submit a PDF report containing the code and answers for questions mentioned in: 

1. Positional encoding.
2. NeRF forward pass code.
3. Difference between our implemented NeRF and the original NeRF paper. 
4. At least two images showing your final rendering.

Otherwise, you will lose points. Submit your work using Canvas.