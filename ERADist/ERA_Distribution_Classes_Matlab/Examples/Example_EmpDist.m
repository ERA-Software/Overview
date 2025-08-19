%% Example file: Creation and use of an EmpDist object
%{
---------------------------------------------------------------------------
Developed by: 
Michael Engel

Engineering Risk Analysis Group
Technische Universitat Munchen
www.bgu.tum.de/era
---------------------------------------------------------------------------
Initial Version 2025-07
---------------------------------------------------------------------------
%}

clear; clc; close all;

%% Main Script
rng(2025);

% 1. Generate a bimodal Gaussian mixture dataset
N = 100;
data = sample_bimodal_gaussian(N, [0.2, 0.8], [-2, 3], [0.5, 1.0]);
weights = ones(size(data)); % uniform empirical weights

% 2. Fit the empirical distribution
dist_emp = EmpDist(...
    data, 'weights', weights, 'pdfMethod', 'kde', ...
    'pdfPoints', [], 'pdfMethodParams', struct('Bandwidth', 0.1));

% 3. Plot histogram + estimated PDF
x_grid = linspace(min(data)-1, max(data)+1, 1000);
pdf_vals = dist_emp.pdf(x_grid);

figure('Position', [100 100 800 400]);
histogram(data, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.5, ...
          'FaceColor', [0.5 0.5 0.5], 'DisplayName', 'Original data');
hold on;
plot(x_grid, pdf_vals, 'r-', 'LineWidth', 2, 'DisplayName', 'Estimated PDF');
title('Original Histogram with Estimated PDF Overlay');
xlabel('x'); ylabel('Density');
legend('Location','best');
grid on;

% 4. Plot CDF and inverse CDF (PPF)
cdf_vals = dist_emp.cdf(x_grid);
y_grid = linspace(0,1,1000);
icdf_vals = dist_emp.icdf(y_grid);

figure('Position', [100 100 1200 400]);
subplot(1,2,1);
plot(x_grid, cdf_vals, 'b-', 'LineWidth', 1.5);
title('Estimated CDF');
xlabel('x'); ylabel('F(x)'); grid on;

subplot(1,2,2);
plot(y_grid, icdf_vals, 'g-', 'LineWidth', 1.5);
title('Estimated Inverse CDF (PPF)');
xlabel('Quantile'); ylabel('x'); grid on;

% 5. Draw new samples from the empirical distribution
M = 2000;
sampled = dist_emp.random(M);

% 6. Compare histograms: original vs resampled
figure('Position', [100 100 800 400]);
histogram(data, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.4, ...
          'DisplayName', 'Original data');
hold on;
histogram(sampled, 40, 'Normalization', 'pdf', 'FaceAlpha', 0.4, ...
          'DisplayName', 'DistME samples');
title('Comparison of Original and DistME Histograms');
xlabel('x'); ylabel('Density');
legend('Location','best');
grid on;


%% Helper function: sample from bimodal Gaussian
function data = sample_bimodal_gaussian(n_samples, mix_weights, means, stds)
    comps = randsample([0, 1], n_samples, true, mix_weights);
    data = zeros(n_samples,1);
    idx0 = comps == 0;
    idx1 = comps == 1;
    data(idx0) = normrnd(means(1), stds(1), sum(idx0), 1);
    data(idx1) = normrnd(means(2), stds(2), sum(idx1), 1);
end
