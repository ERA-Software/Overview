import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d, PchipInterpolator

class EmpDist():
    """
    Returns a distribution object similar to scipy.stats based on a dataset
    given by the user.
    ---------------------------------------------------------------------------
    Developed by: Michael Engel
    ---------------------------------------------------------------------------
    Initial Version: 2025-07
    ---------------------------------------------------------------------------
    """
    ## initialization
    def __init__(
        self,
        data,
        weights = None, # None for equal weights
        pdfMethod = "kde", # kde, linear, slinear, quadratic, cubic, nearest, next...
        pdfPoints = None, # None or integer (only relevant if pdfMethod is not equal to 'KDE')
        **pdfMethodParams
    ):
        '''
        :param data: One dimensional data array.
        :param weights: Weights associated to the data array. The default is None.
        :param pdfMethod: Desired method for the PDF creation. The default is KDE.
        :param pdfPoints: Desired number of points for the PDF creation. The default is None (will resolve in the square root of the number of data points).
        :params pdfMethodParams: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.
        '''
        
        self.cleanData = data[~np.isnan(data)]
        self._N = len(self.cleanData)
        self.normalizedWeights = np.ones_like(self.cleanData)/self._N if weights is None else weights[~np.isnan(data)]/np.sum(weights[~np.isnan(data)])
        
        self.pdfMethod = pdfMethod
        self.pdfPoints = pdfPoints if pdfPoints is not None else np.max([1,int(np.sqrt(self._N))])
        self.pdfMethodParams = pdfMethodParams
        
        # statistics
        self._mean = np.sum(self.cleanData*self.normalizedWeights)
        self._var = np.cov(self.cleanData, aweights=self.normalizedWeights)
        self._std = np.sqrt(self._var)
        
        # cdf and inverse cdf
        self._cdf = create_weighted_cdf_interp1d(self.cleanData, self.normalizedWeights, kind="previous")
        self._ppf = create_weighted_ppf_interp1d(self.cleanData, self.normalizedWeights, kind="next")
            
        # pdf
        if self.pdfMethod.lower()=="kde":
            print("EmpDist: Using Gaussian KDE for PDF!")
            dataSorted, weightsSorted = sortDataWeights(self.cleanData, self.normalizedWeights)
            self._pdf = sps.gaussian_kde(dataset=dataSorted, weights=weightsSorted, **pdfMethodParams)
        else:
            print("EmpDist: Using numerical derivative for PDF!")
            self._pdf = create_normalized_pdf_from_cdf(self._cdf, self.cleanData.min(), self.cleanData.max(),
                num_points = self.pdfPoints,
                kind = self.pdfMethod.lower()
            )
        pass
    
    def N(self):
        return self._N
                
    def mean(self):
        return self._mean
    
    def var(self):
        return self._var
    
    def std(self):
        return self._std

    def pdf(self, x):
        return self._pdf(x)
                
    def cdf(self, x):
        return self._cdf(x)
    
    def icdf(self, y):
        return self._ppf(y)
    
    def ppf(self, y):
        return self._ppf(y)
       
    def random(self, size=None): # random samples
        rands = np.random.rand(size)
        return self.icdf(rands)
    
    def rvs(self, size=None):
        rands = np.random.rand(size)
        return self.icdf(rands)
    
def sortDataWeights(data, weights):
    data = np.asarray(data)
    weights = np.asarray(weights)
    
    sort_idx = np.argsort(data)
    sorted_data = data[sort_idx]
    sorted_weights = weights[sort_idx]
    
    return sorted_data, sorted_weights

### CDF utils
# interpolation based cdf
def create_weighted_cdf_interp1d(data, weights, kind="linear"):
    sorted_data, sorted_weights = sortDataWeights(data, weights)
    
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight

    cdf_func = interp1d(sorted_data, weighted_cdf_values,
                        kind=kind,
                        bounds_error=False,
                        fill_value=(0.0, 1.0),
                        assume_sorted=True)
    return cdf_func

def create_weighted_ppf_interp1d(data, weights, kind="linear"):
    sorted_data, sorted_weights = sortDataWeights(data, weights)

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight

    cdf_func = interp1d(weighted_cdf_values, sorted_data, 
                        kind=kind,
                        bounds_error=False,
                        fill_value=(sorted_data.min(), sorted_data.max()),
                        assume_sorted=True)
    return cdf_func

# pchip based cdf
def create_weighted_cdf_PCHIP(data, weights):
    sorted_data, sorted_weights = sortDataWeights(data, weights)

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight
    
    cdf_func = PchipInterpolator(sorted_data, weighted_cdf_values, extrapolate=False)
    return cdf_func

def create_weighted_ppf_PCHIP(data, weights):
    sorted_data, sorted_weights = sortDataWeights(data, weights)

    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight
    
    cdf_func = PchipInterpolator(weighted_cdf_values, sorted_data, extrapolate=False)
    return cdf_func

### PDF utils
def create_normalized_pdf_from_cdf(cdf_func, x_min, x_max, num_points=1000, kind="linear"):
    x_grid = np.linspace(x_min, x_max, num_points)
    cdf_vals = cdf_func(x_grid)
    
    raw_pdf_vals = np.gradient(cdf_vals, x_grid)
    area = np.trapz(raw_pdf_vals, x_grid)
    
    if area != 0:
        pdf_vals = raw_pdf_vals / area
    else:
        pdf_vals = raw_pdf_vals
    
    pdf_func = interp1d(x_grid, pdf_vals, kind=kind, bounds_error=False, fill_value=0.0)
    return pdf_func
