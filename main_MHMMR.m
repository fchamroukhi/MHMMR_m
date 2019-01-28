% Segmentation of multivariate time series with a Multiple Hidden Markov Model Regression MHMMR).
%
%
% Multiple Hidden Markov Model Regression (HMMR) for segmentation of multivariate time series
% with regime changes. The model assumes that the time series is
% governed by a sequence of hidden discrete regimes/states, where each
% regime/state has Gaussian regressors as observations.
% The model parameters are estimated by MLE via the EM algorithm
%
% Devoloped and written by Faicel Chamroukhi
%
%% Please cite the following papers for this code:
%
% 
% @article{Chamroukhi-FDA-2018,
%  	Journal = {Wiley Interdisciplinary Reviews: Data Mining and Knowledge Discovery},
%  	Author = {Faicel Chamroukhi and Hien D. Nguyen},
%  	Note = {DOI: 10.1002/widm.1298.},
%  	Volume = {},
%  	Title = {Model-Based Clustering and Classification of Functional Data},
%  	Year = {2019},
%  	Month = {to appear},
%  	url =  {https://chamroukhi.com/papers/MBCC-FDA.pdf}
% }
% 
% @article{Chamroukhi-MHMMR-2013,
% 	Author = {Trabelsi, D. and Mohammed, S. and Chamroukhi, F. and Oukhellou, L. and Amirat, Y.},
% 	Journal = {IEEE Transactions on Automation Science and Engineering},
% 	Number = {10},
% 	Pages = {829--335},
% 	Title = {An unsupervised approach for automatic activity recognition based on Hidden Markov Model Regression},
% 	Volume = {3},
% 	Year = {2013},
% 	url  = {https://chamroukhi.com/papers/Chamroukhi-MHMMR-IeeeTase.pdf}
% 	}
% 
% 
% (c) Faicel Chamroukhi (since 2010).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc

% model specification
K = 5; % nomber of regimes (states)
p = 3; % dimension of beta' (order of the polynomial regressors)

% options
%type_variance = 'homoskedastic';
type_variance = 'hetereskedastic';
nbr_EM_tries = 3;
max_iter_EM = 1500;
threshold = 1e-6;
verbose = 1;
type_algo = 'EM';
% type_algo = 'CEM';
% type_algo = 'SEM';

%% toy multivariate time series with regime changes
% y = [[randn(100,1); 7+randn(120,1);4+randn(200,1); -1+randn(100,1); 3.5+randn(150,1)] ...
%     [1+randn(100,1); 5+randn(120,1);6+randn(200,1); -2+randn(100,1); 2+randn(150,1)] ...
%     [-2+randn(100,1); 10+randn(120,1);8+randn(200,1); randn(100,1); 5+randn(150,1)]]
% n = length(y);
% x = linspace(0,1,n);
%

load simulated_time_series;

%y = y(:,1)';

MHMMR = learn_mhmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);

%     %if model selection
%     current_BIC = -inf;
%     for K=1:8
%         for p=0:4
%             MHMMR_Kp = learn_hmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose)
% 
%             if MHMMR_Kp.stats.BIC>current_BIC
%                 MHMMR=MHMMR_Kp;
%                 current_BIC = MHMMR_Kp.stats.BIC;
%             end
%                 bic(K,p+1) = MHMMR_Kp.stats.BIC;
%         end
%     end
show_MHMMR_results(x,y, MHMMR)


%% real multivariate time series with regime changes
% an example of human activity acceleration time series

load real_time_series;
%y = y(:,1)';

MHMMR = learn_mhmmr(x, y, K, p, type_variance, nbr_EM_tries, max_iter_EM, threshold, verbose);

show_MHMMR_results(x,y, MHMMR)


% sample an MHMMR
%[y, states, Z, mean_function] = sample_mhmmr(x, MHMMR.prior, MHMMR.trans_mat, MHMMR.reg_param.betak,MHMMR.reg_param.sigma2k);

