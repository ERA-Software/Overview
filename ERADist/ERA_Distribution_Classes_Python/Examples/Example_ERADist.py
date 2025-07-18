from ERADist import ERADist
import numpy as np
import matplotlib.pylab as plt

'''
---------------------------------------------------------------------------
Example file: Definition and use of ERADist objects
---------------------------------------------------------------------------
 
 In this example a lognormal distribution is defined by its parameters,
 moments and data.Furthermore the different methods of ERADist are
 illustrated.
 For other distributions and more information on ERADist please have a look
 at the provided documentation or execute the command 'help(ERADist)'.
 
---------------------------------------------------------------------------
Developed by:
Sebastian Geyer
Felipe Uribe
Iason Papaioannou
Daniel Straub

Assistant Developers:
Luca Sardi
Alexander von Ramm
Matthias Willer
Peter Kaplan

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Version 2021-03
---------------------------------------------------------------------------
References:
1. Documentation of the ERA Distribution Classes
---------------------------------------------------------------------------
'''

np.random.seed(2021) #initializing random number generator

''' Definition of an ERADist object by the distribution parameters '''

dist = ERADist('lognormal','PAR',[2,0.5])

# computation of the first two moments
mean_dist = dist.mean()
std_dist = dist.std()

# generation of n random samples
n = 10000
samples = dist.random(n)

''' Definition of an ERADist object by the first moments
 Based on the just determined moments a new distribution object with the
 same properties is created... '''

dist_mom = ERADist('lognormal','MOM',[mean_dist,std_dist])

''' Definition of an ERADist object by data fitting
 Using maximum likelihood estimation a new distribution object is created
 from the samples which were created above.'''

dist_data = ERADist('lognormal','DATA',samples)


''' Other methods '''

# generation of n samples x to work with
x = dist.random(n)

# computation of the PDF for the samples x
pdf = dist.pdf(x)

# computation of the CDF for the samples x
cdf = dist.cdf(x)

# computation of the inverse CDF based on the CDF values (-> initial x)
icdf = dist.icdf(cdf)


''' Plot of the PDF and CDF '''

x_plot = np.linspace(0,40,200);     # values for which the PDF and CDF are evaluated 
pdf = dist.pdf(x_plot);     # computation of PDF
cdf = dist.cdf(x_plot);     # computation of CDF

fig_dist = plt.figure(figsize=[16, 9])

fig_pdf = fig_dist.add_subplot(121)
fig_pdf.plot(x_plot, pdf)
fig_pdf.set_xlabel(r'$X$')
fig_pdf.set_ylabel(r'$PDF$')

fig_cdf = fig_dist.add_subplot(122)
fig_cdf.plot(x_plot, cdf)
fig_cdf.set_xlabel(r'$X$')
fig_cdf.set_ylabel(r'$CDF$')