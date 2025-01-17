# Train Neural ODEs with Direct Collocation

This repository implements a method to optimize Neural ODEs by approximating their integrated paths with local polynomials.
Instead of relying on backpropagation through the integrator, we jointly optimize the neural network weights and the coefficients of these polynomials.

This approach increases the optimization space and includes constraints to approximate the underlying differential equations and any state constraints.

**Key Idea:**
* Joint optimization of neural network parameters and polynomial path approximations.
* Avoid backpropagation through the numerical ODE solver.
* Include constraints for ODE satisfaction and state limits.

**Example:**
This repository showcases the constrained control van der Pol problem from the paper *Neural ODEs as Feedback Policies for Nonlinear Optimal Control* ([IFAC 2023](https://doi.org/10.1016/j.ifacol.2023.10.1248)).
