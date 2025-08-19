from ERADist import ERADist
import numpy as np
import matplotlib.pylab as plt

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERADist empirical distribution object
---------------------------------------------------------------------------
In this example an empirical distribution is defined by a set of datapoints.
Furthermore the different methods of ERADist are illustrated. 
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
References:
1. Documentation of the ERA Distribution Classes
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
    np.random.seed(2025) #initializing random number generator
    n = 2000 #number of data points
    
    # generate a bimodal Gaussian mixture dataset
    data = sample_bimodal_gaussian(n_samples=n,
                                   mix_weights=(0.2, 0.8),
                                   means=(-2, 3),
                                   stds=(0.5, 1.0))
    weights = None
    
    ''' Definition of an ERADist object by a dataset '''
    dist = ERADist('empirical','DATA',[data, weights, "kde", None, {"bw_method":None}])
    
    # computation of the first two moments
    mean_dist = dist.mean()
    std_dist = dist.std()
    
    # generation of n random samples
    n = 2000
    samples = dist.random(n)
    
    ''' Other methods '''
    # generation of n samples x to work with
    x = dist.random(n)
    print("x", x)
    
    # computation of the PDF for the samples x
    pdf = dist.pdf(x)
    print("pdf", pdf)
    
    # computation of the CDF for the samples x
    cdf = dist.cdf(x)
    print("cdf", cdf)
    
    # computation of the inverse CDF based on the CDF values (-> initial x)
    icdf = dist.icdf(cdf)
    print("icdf", icdf)
    
    
    ''' Plot of the PDF and CDF '''
    x_plot = np.linspace(data.min()-1,data.max()+1,200);     # values for which the PDF and CDF are evaluated 
    pdf = dist.pdf(x_plot);     # computation of PDF
    cdf = dist.cdf(x_plot);     # computation of CDF
    
    fig_dist = plt.figure(figsize=[10, 10])
    
    fig_pdf = fig_dist.add_subplot(211)
    fig_pdf.hist(data, density=True, bins=int(np.sqrt(len(data))), label="data", color='r', alpha=0.3)
    fig_pdf.plot(x_plot, pdf, 'b', lw=2, label="Empirical PDF")
    fig_pdf.set_xlim([data.min(), data.max()])
    fig_pdf.set_xlabel(r'$X$')
    fig_pdf.set_ylabel(r'$PDF$')
    fig_pdf.legend()
    
    fig_cdf = fig_dist.add_subplot(212)
    fig_cdf.plot(x_plot, cdf, "b", lw=2, label="Empirical CDF")
    fig_cdf.set_xlim([data.min(), data.max()])
    fig_cdf.set_xlabel(r'$X$')
    fig_cdf.set_ylabel(r'$CDF$')
    
    plt.show()