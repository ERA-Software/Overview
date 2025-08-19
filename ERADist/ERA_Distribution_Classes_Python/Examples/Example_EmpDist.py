import numpy as np
import matplotlib.pyplot as plt

from EmpiricalDist import EmpDist

'''
---------------------------------------------------------------------------
Example file: Creation and use of an EmpDist object
---------------------------------------------------------------------------
In this example a lognormal distribution is defined by its parameters,
moments and data.Furthermore the different methods of ERADist are
illustrated.
For other distributions and more information on ERADist please have a look
at the provided documentation.
---------------------------------------------------------------------------
Developed by: 
Michael Engel

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Initial Version 2025-07
---------------------------------------------------------------------------
'''

def sample_bimodal_gaussian(n_samples=1000, mix_weights=(0.4, 0.6),
                            means=(-2, 3), stds=(0.7, 1.2)):
    comps = np.random.choice([0, 1], size=n_samples, p=mix_weights)
    data = np.where(
        comps == 0,
        np.random.normal(loc=means[0], scale=stds[0], size=n_samples),
        np.random.normal(loc=means[1], scale=stds[1], size=n_samples),
    )
    return data

if __name__ == "__main__":
    np.random.seed(2025)
    # 1. Generate a bimodal Gaussian mixture dataset
    N = 100
    data = sample_bimodal_gaussian(n_samples=N,
                                   mix_weights=(0.2, 0.8),
                                   means=(-2, 3),
                                   stds=(0.5, 1.0))
    weights = np.ones_like(data)  # uniform empirical weights

    # 2. Fit the empirical distribution
    dist_emp = EmpDist(data, weights=weights, pdfMethod="kde", pdfPoints=None, bw_method=0.1)

    # 3. Plot histogram + estimated PDF
    x_grid = np.linspace(data.min() - 1, data.max() + 1, 1000)
    pdf_vals = dist_emp.pdf(x_grid)

    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=40, density=True, alpha=0.5, color="gray", label="Original data")
    plt.plot(x_grid, pdf_vals, 'r-', linewidth=2, label="Estimated PDF")
    plt.title("Original Histogram with Estimated PDF Overlay")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    # 4. Plot CDF and inverse CDF (PPF)
    cdf_vals = dist_emp.cdf(x_grid)
    y_grid = np.linspace(0, 1, 1000)
    icdf_vals = dist_emp.icdf(y_grid)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(x_grid, cdf_vals, 'b-')
    axes[0].set_title("Estimated CDF")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("F(x)")

    axes[1].plot(y_grid, icdf_vals, 'g-')
    axes[1].set_title("Estimated Inverse CDF (PPF)")
    axes[1].set_xlabel("Quantile")
    axes[1].set_ylabel("x")

    plt.tight_layout()
    plt.show()

    # 5. Draw new samples from the empirical distribution
    M = 2000
    sampled = dist_emp.random(size=M)

    # 6. Compare histograms: original vs resampled
    plt.figure(figsize=(8, 4))
    plt.hist(data, bins=40, density=True, alpha=0.4, label="Original data")
    plt.hist(sampled, bins=40, density=True, alpha=0.4, label="DistME samples")
    plt.title("Comparison of Original and DistME Histograms")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()