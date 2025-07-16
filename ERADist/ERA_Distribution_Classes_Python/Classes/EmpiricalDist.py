#### Michael Engel ### 2025-07-16 ### EmpiricalDist.py ###
import warnings
import numpy as np
import scipy.stats as sps
from scipy.interpolate import interp1d, PchipInterpolator

class DistME():
    """
    Returns a distribution object similar to scipy.stats based on a dataset
    given by the user.
    """
    #%% initialization
    def __init__(
        self,
        data,
        weights = None, # None for equal weights
        cdfMethod = "pchip", # pchip, linear, slinear...
        pdfMethod = "kde", # kde, linear, slinear, quadratic, cubic...
        **pdfMethodParams
    ):
        '''
        :param data: One dimensional data array.
        :param weights: Weights associated to the data array. The default is None.
        :param cdfMethod: Desired method for the CDF creation. The default is PCHIP.
        :param pdfMethod: Desired method for the PDF creation. The default is KDE.
        :params pdfMethodParams: Optional scipy.stats.gaussian_kde keyword arguments for the case of KDE based PDF creation.
        '''
        
        self.cleanData = data[~np.isnan(data)]
        self._N = len(self.cleanData)
        self.normalizedWeights = np.ones_like(self.cleanData)/self._N if weights is None else weights[~np.isnan(data)]/np.sum(weights[~np.isnan(data)])
        
        self.cdfMethod = cdfMethod
        self.pdfMethod = pdfMethod
        self.pdfMethodParams = pdfMethodParams
        
        # statistics
        self._mean = np.sum(self.cleanData*self.normalizedWeights)
        self._var = np.cov(self.cleanData, aweights=self.normalizedWeights)
        self._std = np.sqrt(self._var)
        
        # cdf and inverse cdf
        if self.cdfMethod.lower()=="pchip":
            print("DistME: Using PCHIP for CDF!")
            self._cdf = create_weighted_cdf_PCHIP(self.cleanData, self.normalizedWeights)
            self._ppf = create_weighted_ppf_PCHIP(self.cleanData, self.normalizedWeights)
            
        elif self.cdfMethod.lower()=="linear" or self.cdfMethod.lower()=="slinear":
            print(f"DistME: Using {self.cdfMethod.lower()} interpolation for CDF!")
            self._cdf = create_weighted_cdf_interp1d(self.cleanData, self.normalizedWeights, kind=self.cdfMethod.lower())
            self._ppf = create_weighted_ppf_interp1d(self.cleanData, self.normalizedWeights, kind=self.cdfMethod.lower())
        else:
            warnings.warn(f"DistME: Using {self.cdfMethod.lower()} interpolation for CDF! Please note that we do recommend either PCHIP or somewhat linear interpolation!")
            self._cdf = create_weighted_cdf_interp1d(self.cleanData, self.normalizedWeights, kind=self.cdfMethod.lower())
            self._ppf = create_weighted_ppf_interp1d(self.cleanData, self.normalizedWeights, kind=self.cdfMethod.lower())
            
        # pdf
        if isinstance(self.pdfMethod, str) and self.pdfMethod.lower()=="kde":
            print("DistME: Using Gaussian KDE for PDF!")
            dataSorted, weightsSorted = sortDataWeights(self.cleanData, self.normalizedWeights)
            self._pdf = sps.gaussian_kde(dataset=dataSorted, weights=weightsSorted, **pdfMethodParams)
            
        elif isinstance(self.pdfMethod, str) and self.pdfMethod.lower()=="pchip":
            if not (self.cdfMethod.lower()=="pchip"):
                raise RuntimeError("DistME: cdfMethod has to be PCHIP as well in order to use its derivative for the PDF!")
            else:
                print("DistME: Using PCHIP derivative for PDF!")
                self._pdf = self._cdf.derivative(1)
                
        else:
            print("DistME: Using numerical derivative for PDF!")
            self._pdf = create_normalized_pdf_from_cdf(self._cdf, self.cleanData.min(), self.cleanData.max(),
                num_points = self.pdfMethod if isinstance(self.pdfMethod, int) else int(1+3.3*np.log10(self._N)) if self.pdfMethod.lower()=="log" else np.max([1,int(np.sqrt(self._N))]),
                kind = "linear" if self.pdfMethod.lower()=="log" or isinstance(self.pdfMethod, int) else self.pdfMethod.lower()
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
    
    def ppf(self, y): # inverse cdf
        return self._ppf(y)
       
    def rvs(self, size=None): # random samples
        rands = np.random.rand(size)
        return self.ppf(rands)
    
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
    
    cum_weights = np.cumsum(sorted_weights)-sorted_weights[0]
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

    cum_weights = np.cumsum(sorted_weights)-sorted_weights[0]
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

    cum_weights = np.cumsum(sorted_weights)-sorted_weights[0]
    total_weight = cum_weights[-1]
    weighted_cdf_values = cum_weights / total_weight
    
    cdf_func = PchipInterpolator(sorted_data, weighted_cdf_values, extrapolate=False)
    return cdf_func

def create_weighted_ppf_PCHIP(data, weights):
    sorted_data, sorted_weights = sortDataWeights(data, weights)

    cum_weights = np.cumsum(sorted_weights)-sorted_weights[0]
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
