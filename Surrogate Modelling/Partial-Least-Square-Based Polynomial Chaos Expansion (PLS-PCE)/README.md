# Partial-Least-Square-Based Polynomial Chaos Expansion (PLS-PCE)

Polynomial chaos expansions (PCE) is a powerful surrogate model for efficient uncertainty propagation. However, there is a major drawback of standard PCE: Its predictive ability decreases with increase of the problem dimension for a fixed computational budget. This is related to the fact that the number of terms in the expansion increases fast with the input variable dimension. The proposed PCE-driven PLS algorithm identifies the directions with the largest predictive significance in the PCE representation based on a set of samples from the input random variables and corresponding response variable. This approach does not require gradient evaluations, which makes it efficient for high dimensional problems with black-box numerical models.


## Algorithms

- Linear version (pls_pce_R_Lin): Assuming **linear relationship between response and latent variables**.
- Non-Linear version (pls_pce_R_NLin): Assuming **non-linear relationship between response and latent variables**.
- Combined version (pls_pce_R_combi): Combine both versions above, assuming Non-Linear relationship only in the first 3 PLS components.


## Example

main_bar: A 1D bar example with KLE-discretized random field.


## Requirements

- MATLAB, incl. Statistical toolbox
- Python 3
    * numpy
    * scipy
    * pyDOE


## Version: 

Last change: 01/2020


## Documentation & background

Papaioannou, I., Ehre, M., & Straub, D. (2019). PLS-based adaptation for efficient PCE representation in high dimensions. Journal of Computational Physics, 387, 186–204. doi: 10.1016/j.jcp.2019.02.046


