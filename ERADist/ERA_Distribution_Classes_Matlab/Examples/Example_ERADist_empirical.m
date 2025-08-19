%% Example file: Definition and use of ERADist empirical distribution object
%{
In this example an empirical distribution is defined by a set of datapoints.
Furthermore the different methods of ERADist are illustrated. For other
distributions and more information on ERADist please have a look at the
provided documentation or execute the command 'help ERADist'.

Developed by: 
Michael Engel

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
--------------------------------------------------------------------------
 Initial Version 2025-07
--------------------------------------------------------------------------
%}

clear; clc; close all;

%% 1. Initialization
rng(2025);          % initializing random number generator
n = 2000;           % number of data points

% generate a bimodal Gaussian mixture dataset
data = sample_bimodal_gaussian(n, [0.2, 0.8], [-2, 3], [0.5, 1.0]);

weights = []; % None = equal weights

%% 2. Definition of an ERADist object by a dataset
dist = ERADist('empirical', 'DATA', {data, weights, 'kde', [], struct('bw_method', [])});

%% 3. Computation of first two moments
mean_dist = dist.mean();
std_dist  = dist.std();

fprintf('Mean of distribution: %.4f\n', mean_dist);
fprintf('Std of distribution: %.4f\n', std_dist);

%% 4. Generation of n random samples
n = 2000;
samples = dist.random(n);

%% 5. Other methods
x = dist.random(n);  % generate samples
disp('x ='); disp(x(1:10)); % show first 10

pdf_vals = dist.pdf(x);
disp('pdf(x) ='); disp(pdf_vals(1:10));

cdf_vals = dist.cdf(x);
disp('cdf(x) ='); disp(cdf_vals(1:10));

icdf_vals = dist.icdf(cdf_vals);
disp('icdf(cdf(x)) ='); disp(icdf_vals(1:10));

%% 6. Plot of PDF and CDF
x_plot = linspace(min(data)-1, max(data)+1, 200);
pdf_plot = dist.pdf(x_plot);
cdf_plot = dist.cdf(x_plot);

figure('Position',[200 200 800 800]);

subplot(2,1,1);
histogram(data, round(sqrt(length(data))), 'Normalization','pdf', ...
    'FaceColor','r','FaceAlpha',0.3,'DisplayName','Data');
hold on;
plot(x_plot, pdf_plot, 'b','LineWidth',2,'DisplayName','Empirical PDF');
xlim([min(data), max(data)]);
xlabel('X'); ylabel('PDF');
legend show;

subplot(2,1,2);
plot(x_plot, cdf_plot, 'b','LineWidth',2,'DisplayName','Empirical CDF');
xlim([min(data), max(data)]);
xlabel('X'); ylabel('CDF');
legend show;

%% Helper function: sample from bimodal Gaussian
function data = sample_bimodal_gaussian(n_samples, mix_weights, means, stds)
    comps = randsample([0, 1], n_samples, true, mix_weights);
    data = zeros(n_samples,1);
    idx0 = comps == 0;
    idx1 = comps == 1;
    data(idx0) = normrnd(means(1), stds(1), sum(idx0), 1);
    data(idx1) = normrnd(means(2), stds(2), sum(idx1), 1);
end
