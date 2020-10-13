# Notes on BayesianTracker
---
BayesianTracker (`btrack`), not so surprisingly, uses a bayesian tracking paradigms. This means that for each object being tracked a model is used to represent the probable movements of the object. Using Bayes' Theorem, given a single track the posterior probability for each putative match in a successive frame can be calculated. The tracking assignments are made by composing a tracks x objects Bayesian belief matrix. In `btrack` the posterior probabilities arise from a predictive motion model. Initially, this model is used to assemble and grow tracklets. Once the tracklets have been assembled, a series of hypotheses about the possible origins and fates of each tracklet are produced and the log likelihood of each is calculated. This optimal solution, the one that is most probable across all tracks, is found by solving an integer linear program.

BTW, congratulate me, for (after much digging and consultation of a tutorial) I can kind of sort of almost read C/C++ =P

*__NB__: Apologies for the blocks of copied code, they are there for reference and out of laziness.  

---
## Configuration file
---
### `MotionModel`
The motion model is a Kalman filter (which can presented as a simple Bayesian estimator). The filter uses a model and `m` measurements of `n` state variables (vector `x`) to estimate the true state of the system. Both the model and the measurements are sensitive to perturbation by noise, which is usually modeled as gaussian white noise. This often has two phases, prediction and update. The filter is a recursive estimator. The model is instantiated with a state transition model (`A`), information about which of the state variables are measured (`H`), estimates for the noise perturbing the transition model (`Q`) and the measurements (`R`)(usually supplied as covariance matrices representing multivariate Gaussian distributions), and a priors for the state prediction (`x_hat`) and it's covariance matrix(`P`), which represents the accuracy. At successive time steps, there are two stages of changes to `x_hat` and `P`. In the __prediction__ stage, *a priori* estimates are made based on the transition model and predicted noise. In the __update__ stage, *a posterior* estimates are made on the basis of the measurements and predicted noise. 

`"dt"`: change in time. In the example file this is 1.0 indicating that t corresponds one-to-one with frames in movie.

`"measurements"`: number of measurements taken about the system. In the example this is 3, because we can observe only the x, y, z coordinates of the object - velocity cannot be directly measured. 

`"states"`: The number of states required to fully describe the motion model. In the example configuration file 6 states are given corresponding to x-transpose where `x = [x, y, z, del_x, del_y, del_z]`. This assumes that these are the only state values that can affect the transition of the system.

`"accuracy"`: Used when finding the probability that a given object in frame t+1 belongs to a particular tracklet (for the Bayesian belief matrix) based on the prediction by the Kalman filter.  The accuracy determines the interval on the gaussian over which the probability is calculated.  

`"prob_not_assign"`: Presumably the probability that the track is not assigned an object at the next time step?

`"max_lost"`: The maximum number of tracks that can be lost (i.e., not assigned a next object) per time step.


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

`"H"`: This is the observation matrix. This matrix determines how states are transformed into measurements. In the example a 6x3 matrix selecting only coordinate variables is supplied. 

`"P"`: The initial estimate of covariance matrix, P consists of two fields: `"sigma"` and `"matrix"`. Sigma is multiplied by matrix to give the initial covariance matrix, which represents the accuracy of the state estimate x_hat. 

`"G"`: The `"sigma"` and `"matrix"` fields are multiplied to give G, a vector of length n, which is used to produce the matrix Q (see code below). Q is the covariance matrix of the estimated noise that impacts the quality of the model predictions given by A. This noise is modeled as a multivariate gaussian distribution where random variables are system states (coordinates and velocities). 

`"R"`: TThe `"sigma"` and `"matrix"` fields are multiplied to give R. R is the covariance of the noise affecting the coordinate measurements, which again is modeled as a multivariate Gaussian distribution, this time with variables representing coordinates. 

To understand how the matrices are used (and the necessary dimensions) consult the following code block.

*Below is a pythonised altered version of part of the implementation of the Kalman filter in* `btrack/src/motion.cc`
```Python
# A Priori Predictions
# --------------------
# predict x_hat on the basis of the model
x_hat_new = A @ x_hat
# new covariance estimate 
P = A @ P @ A.T + Q
# NOTE: G = G['sigma'] * G['matrix']; Q =  G @ G.T

# A Posteriori Predictions
# ------------------------
# pre-fit residual of covariance - difference from the optimal forecast (a.k.a., innovation [apparently])
S = (H @ P @ H.T + R) # R = R['sigma] * R[matrix]
# kalman gain
K = (P @ H.T) @ np.linalg.inv(S)
# pre-fit residual of prediction. z = measurement vector 
y_tida = z - H @ x_hat_new
# update state prediction
x_hat_new = x_hat_new + K @ y_tida
# update covariance
P = (I - K @ H) @ P
```

*The following is from* `btrack/include/motion.h` *and shows the beginning of the definition of* `MotionModel` *class*
```C
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
```C
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
```C
// hypothesis and state types
// ['P_FP','P_init','P_term','P_link','P_branch','P_dead','P_merge']
```

`"lambda_link"`: The cost associated with linking two tracklets is log proportional to displacement, which may be scaled by a linking penalty. The penalty defaults to `link_penalty = 1.0` unless `DISALLOW_METAPHASE_ANAPHASE_LINKING = true` or `DISALLOW_PROMETAPHASE_ANAPHASE_LINKING = true`. These are used when cells are known to be cycling and mitosis stage is a known variable. which is scaled by the `lambda_link` parameter as follows:
```C++
return std::exp(-(d*link_penalty)/m_params.lambda_link)
```

My `btrack` deep dive only got this far :).  

