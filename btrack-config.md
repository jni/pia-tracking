# Notes on btrack
## Configuration file
---
### `MotionModel`
The motion model is a Kalman filter. The filter uses a model and `m` measurements of `n` state variables to estimate the true state of the system. Both the model and the measurements are sensitive to peturbation by noise, which is usually modeled as gaussian white noise. By solving an optimisation problem whereby the square expected error is minimised (time complexity O(n^3) - traktable only for small `n`) the 

`"dt"`: change in time. In the example file this is 1.0 indicating that t corresponds one-to-one with frames in movie.

`"measurements"`: number of measurements taken about the system. In the example this is 3, because we can observe only the x, y, z coordinates of the object - velocity cannot be directly measured. 

`"states"`: The number of states required to fully describe the motion model. In the example configuration file 6 states are given coresponding to x-transpose where `x = [x, y, z, del_x, del_y, del_z]`. This assumes that these are the only state values that can affect the transition of the system.

`"accuracy"`: (7.5)

`"prob_not_assign"`: (0.001)

`"max_lost"`: (5)


`"A"`: This is an nxn matrix giving the state transition model that encodes the physical model for object motion. In the example configuration file this is a 6x6 matrix indicating that objects have constant velocity on each axis (shown in the code block below).

```Python
x = np.array([x, y, z, del_x, del_y, del_z]).T
A = np.array([[1, 0, 0, 1, 0, 0], 
              [0, 1, 0, 0, 1, 0], 
              [0, 0, 1, 0, 0, 1],
              [0, 0, 0, 1, 0, 0],
              [0, 0, 0, 0, 1, 0], 
              [0, 0, 0, 0, 0, 1]])
x_hat = A @ x
# x_hat.T --> [x + del_x, y + del_y, z + del_z, del_x, del_y, del_z]
# yes I did use code because markdown equations are a pain :)
``` 

`"H"`: This is the observation matrix. This matrix determines how states are transformed into measurments. In the example a 6x3 matrix selecting only coordinate variables is supplied. 

`"P"`: P condists of two fields: `"sigma"` and `"matrix"`. Sigma gives the Gaussian (presumably mean = 0) from which noise is drawn and matrix gives a matrix that is used to scale the noise for each state. In the example, a sigma of `150.0` is given and this is scaled by `0.1` for coordinates and `1` for velocities.

`"G"`: G condists of two fields: `"sigma"` and `"matrix"`. Sigma gives the Gaussian (presumably mean = 0) from which noise is drawn and matrix gives a matrix that is used to scale the noise for each state. __I do not know what this is used for__. In the example, sigma `15.0` is given and this is scaled by `0.5` for coords and `1` for velocities.

`"R"`: R condists of two fields: `"sigma"` and `"matrix"`. Sigma gives the Gaussian (presumably mean = 0) from which noise is drawn and matrix gives a matrix that is used to scale the noise for each measurment. In the example sigma `5` is given and matrix `np.eye(3)`


*The following is from* `btrack/include/motion.h` *and shows the beginning of the definition of* `MotionModel` *class*
```C++
// Implements a Kalman filter for motion modelling in the tracker. Note that we
// do not implement the 'control' updates from the full Kalman filter
class MotionModel
{
  public:
    // Default constructor for MotionModel
    MotionModel() {};

    // Initialise a motion model with matrices representing the motion model.
    // A: State transition matrix
    // H: Observation matrix
    // P: Initial covariance estimate
    // Q: Estimated error in process
    // R: Estimated error in measurements
    // Certain parameters are inferred from the shapes of the matrices, such as
    // the number of states and measurements
    MotionModel(const Eigen::MatrixXd &A,
                const Eigen::MatrixXd &H,
                const Eigen::MatrixXd &P,
                const Eigen::MatrixXd &R,
                const Eigen::MatrixXd &Q);
```
NOTE: this is where I found out where some of these things were initially defined. 

---
### `ObjectModel`

---
### `HypothesisModel`

`"name"`:

`"hypotheses"`: refers to the track hypotheses that will be used in the optimisation. In the example configuration file the following list is given `["P_FP", "P_init", "P_term", "P_link", "P_branch", "P_dead"]`, the meaning of each is outlined in the comments in the following code block:

*The below is from* `btrack/include/hypothesis.h`
```C++
// HypothesisEngine
//
// Hypothesis generation for global track optimisation. Uses the tracks from
// BayesianTracker to generate hypotheses.
//
// Generates these different hypotheses, based on the track data provided:
//
//   1. P_FP: a false positive trajectory (probably very short)
//   2. P_init: an initialising trajectory near the edge of the screen or
//      beginning of the movie
//   3. P_term: a terminating trajectory near the edge of the screen or
//      end of movie
//   4. P_link: a broken trajectory which needs to be linked. Probably a
//      one-to-one mapping
//   5. P_branch: a division event where two new trajectories initialise
//   6. P_dead: an apoptosis event
//   7. P_extrude: a cell extrusion event. A cell is removed from the tissue.
```
But also:
```C++
// hypothesis and state types
// ['P_FP','P_init','P_term','P_link','P_branch','P_dead','P_merge']
```

