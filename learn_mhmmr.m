function mhmmr = learn_mhmmr(x, y, K, p,...
    type_variance, total_EM_tries, max_iter_EM, threshold, verbose)

% learn_mhmmr learn a Regression model with a Hidden Markov Process (MHMMR)
% for modeling and segmentation of a time series with regime changes.
% The learning is performed by the EM (Baum-Welch) algorithm.
%
%
% Inputs :
%
%          (x,y) : a time series composed of m points : dim(y)=[m d]
%                * Each curve is observed during the interval [0,T], i.e x =[t_1,...,t_m]
%
%           K : Number of polynomial regression components (regimes)
%          	p : degree of the polynomials
%
% Outputs :
%
%         mhmmr: the estimated MHMMR model. a structure composed of:
%
%         prior: [Kx1]: prior(k) = Pr(z_1=k), k=1...K
%         trans_mat: [KxK], trans_mat(\ell,k) = Pr(z_t = k|z_{t-1}=\ell)
%         reg_param: the paramters of the regressors:
%                 betak: regression coefficients
%                 sigmak (or sigma2) : the variance(s)
%         Stats:
%           tau_tk: smoothing probs: [nxK], tau_tk(t,k) = Pr(z_i=k | y1...yn)
%           alpha_tk: [nxK], forwards probs: Pr(y1...yt,zt=k)
%           beta_tk: [nxK], backwards probs: Pr(yt+1...yn|zt=k)
%           xi_tkl: [(n-1)xKxK], joint post probs : xi_tk\elll(t,k,\ell)  = Pr(z_t=k, z_{t-1}=\ell | Y) t =2,..,n
%           X: [nx(p+1)] regression design matrix
%           nu: model complexity
%           parameter_vector
%           f_tk: [nxK] f(yt|zt=k)
%           log_f_tk: [nxK] log(f(yt|zt=k))
%           loglik: log-likelihood at convergence
%           stored_loglik: stored log-likelihood values during EM
%           cputime: for the best run
%           cputime_total: for all the EM runs
%           klas: [nx1 double]
%           Zik: [nxK]
%           state_probs: [nxK]
%           BIC: -2.1416e+03
%           AIC: -2.0355e+03
%           regressors: [nxK]
%           predict_prob: [nxK]: Pr(zt=k|y1...y_{t-1})
%           predicted: [nx1]
%           filter_prob: [nxK]: Pr(zt=k|y1...y_t)
%           filtered: [nx1]
%           smoothed_regressors: [nxK]
%           smoothed: [nx1]
%
%
%Faicel Chamroukhi, sept 2008
%
%% Please cite the following papers for this code:
%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
warning off

if nargin<9, verbose =0; end
if nargin<8, threshold = 1e-6; end
if nargin<7, max_iter_EM = 1500;end
if nargin<6, total_EM_tries = 1;end
if nargin<5, total_EM_tries = 1;type_variance='hetereskedastic';end
switch type_variance
    case 'homoskedastic'
        homoskedastic =1;
    case 'hetereskedastic'
        homoskedastic=0;
    otherwise
        error('The type of the model variance should be : ''homoskedastic'' ou ''hetereskedastic''');
end

if size(y,2)>size(y,1), y=y'; end %

[m, d] = size(y);

X = designmatrix(x,p);%design matrix
P = size(X,2);% here P is p+1
I = eye(P);% define an identity matrix, in case of a Bayesian regularization for regression

%
best_loglik = -inf;
nb_good_try=0;
total_nb_try=0;

while (nb_good_try < total_EM_tries)
    if total_EM_tries>1,fprintf(1, 'EM try n∞  %d \n ',nb_good_try+1); end
    total_nb_try=total_nb_try+1;
    
    time = cputime;
    
    %% EM Initializaiton step
    %% Initialization of the Markov chain params, the regression coeffs, and the variance(s)
    mhmmr =  init_mhmmr(X, y, K, type_variance, nb_good_try+1);
    
    % calculare the initial post probs  (tau_tk) and joint post probs (xi_ikl)
    
    %f_tk = mhmmr.stats.f_tk; % observation component densities: f(yt|zt=k)
    prior = mhmmr.prior;
    trans_mat = mhmmr.trans_mat;
    Mask = mhmmr.stats.Mask;
    
    betak = mhmmr.reg_param.betak;
    sigmak = mhmmr.reg_param.sigmak;
    
    %
    iter = 0;
    prev_loglik = -inf;
    converged = 0;
    top = 0;
    
    %
    log_f_tk = zeros(m,K);
    %
    %% EM
    while ((iter <= max_iter_EM) && ~converged)
        
        %% E step : calculate tge tau_tk (p(Zt=k|y1...ym;theta)) and xi t_kl (and the log-likelihood) by
        %  forwards backwards (computes the alpha_tk et beta_tk)
        
        % observation likelihoods
        for k=1:K
            mk = X*betak(:,:,k);  % the regressors means
            if homoskedastic; sk = sigmak ;  else; sk = sigmak(:,:,k);end
            z =((y-mk)*inv(sk)).*(y-mk);
            mahalanobis = sum(z,2);
            denom = (2*pi)^(d/2)*(det(sk))^(1/2);
            
            log_f_tk(:,k) = - ones(m,1)*log(denom)- 0.5*mahalanobis;
        end
        
        log_f_tk  = min(log_f_tk,log(realmax));
        log_f_tk = max(log_f_tk ,log(realmin));
        f_tk = exp(log_f_tk);
        
        %fprintf(1, 'forwards-backwards ');
        [tau_tk, xi_tkl, alpha_tk, beta_tk, loglik] = forwards_backwards(prior, trans_mat , f_tk );
        
        %% M step
        %  updates of the Markov chain parameters
        % initial states prob: P(Z_1 = k)
        prior = normalise(tau_tk(1,:)');
        % transition matrix: P(Zt=i|Zt-1=j) (A_{k\ell})
        trans_mat = mk_stochastic(squeeze(sum(xi_tkl,1)));
        % for segmental MHMMR: p(z_t = k| z_{t-1} = \ell) = zero if k<\ell (no back) of if k >= \ell+2 (no jumps)
        trans_mat = mk_stochastic(Mask.*trans_mat);
        
        %%  update of the regressors (reg coefficients betak and the variance(s) sigmak)
        
        if homoskedastic, s = 0; end
        for k=1:K
            weights = tau_tk(:,k);
            nk = sum(weights);% expected cardinal nbr of state k
            
            Xk = X.*(sqrt(weights)*ones(1,P));%[n*(p+1)]
            yk=  y.*(sqrt(weights)*ones(1,d));% dimension :(nxd).*(nxd) = (nxd)
            % reg coefficients
            lambda = 1e-5;% if a bayesian prior on the beta's
            bk = pinv(Xk'*Xk)*Xk'*yk;
            betak(:,:,k) = bk;
            % covariance matrix
            z = (y-X*bk).*(sqrt(weights)*ones(1,d));
            sk = z'*z;
            if homoskedastic, s = (s+sk); sigmak = s/m;else; sigmak(:,:,k) = sk/nk + lambda*eye(d);end
        end
        
        %% En of an EM iteration
        iter =  iter + 1;
        
        % test of convergence
        loglik = loglik + log(lambda);
        
        if verbose, fprintf(1, 'HMM_regression | EM   : Iteration : %d   Log-likelihood : %f \n',  iter, loglik); end
        
        if prev_loglik-loglik > 1e-4
            top = top+1;
            if (top==10)
            %    fprintf(1, '!!!!! The loglikelihood is decreasing from %6.4f to %6.4f!\n', prev_loglik, loglik);
                break;
            end
        end
        converged = abs(loglik - prev_loglik)/abs(prev_loglik) < threshold;
        stored_loglik(iter) = loglik;
        prev_loglik = loglik;
        
    end% end of an EM run
    
    cputime_total(nb_good_try+1) = cputime-time;
    
    mhmmr.prior = prior;
    mhmmr.trans_mat = trans_mat;
    mhmmr.reg_param.betak = betak;
    
    mhmmr.reg_param.sigmak = sigmak;
    
    % Estimated parameter vector (Pi,A,\theta)
    parameter_vector=[prior(:); trans_mat(Mask~=0); betak(:);sigmak(:)];
    %        nu = K-1 + K*(K-1) + K*(p+1) + K;%length(parameter_vector);%
    nu = length(parameter_vector);
    
    mhmmr.stats.nu = nu;
    mhmmr.stats.parameter_vector= parameter_vector;
    
    mhmmr.stats.tau_tk = tau_tk; % posterior (smoothing) probs
    mhmmr.stats.alpha_tk = alpha_tk;%forward probs
    mhmmr.stats.beta_tk = beta_tk;%backward probs
    mhmmr.stats.xi_ikl = xi_tkl;% joint posterior (smoothing) probs
    
    mhmmr.stats.f_tk = f_tk;% obs likelihoods
    mhmmr.stats.log_f_tk = log_f_tk;% log obs likelihoods
    
    mhmmr.stats.loglik= loglik;
    mhmmr.stats.stored_loglik= stored_loglik;
    %
    mhmmr.stats.X = X;%design matrix
    %
    if total_EM_tries>1,   fprintf(1,'loglik_max = %f \n',loglik); end
    %
    %
    if ~isempty(mhmmr.reg_param.betak)
        nb_good_try=nb_good_try+1;
        total_nb_try=0;
        if loglik > best_loglik
            best_mhmmr = mhmmr;
            best_loglik = loglik;
        end
    end
    %
    if total_nb_try > 500
        fprintf('can''t obtain the requested number of classes \n');
        mhmmr=[];
        return;
    end
    
end%End of the EM runs

mhmmr = best_mhmmr;

%
if total_EM_tries>1,    fprintf(1,'best_loglik:  %f\n',mhmmr.stats.loglik);end
%
%
mhmmr.stats.cputime = mean(cputime_total);
mhmmr.stats.cputime_total = cputime_total;

%% Smoothing state sequences : argmax(smoothing probs), and corresponding binary allocations partition
[mhmmr.stats.klas, mhmmr.stats.Zik ] =  MAP(mhmmr.stats.tau_tk);
% %  compute the sequence with viterbi
%[path, ~] = viterbi_path(mhmmr.prior, mhmmr.trans_mat, mhmmr.stats.fik');
%mhmmr.stats.viterbi_path = path;
%mhmmr.stats.klas = path;
%%%%%%%%%%%%%%%%%%%

% %% determination des temps de changements (les fonti√®tres entres les
% %% classes)
% nk=sum(mhmmr.stats.Zik,1);
% for k = 1:K
%     tk(k) = sum(nk(1:k));
% end
% mhmmr.stats.tk = [1 tk];

%% sate sequence prob p(z_1,...,z_n;\pi,A)
state_probs = hmm_process(mhmmr.prior, mhmmr.trans_mat, m);
mhmmr.stats.state_probs = state_probs;


%%% BIC, AIC, ICL
mhmmr.stats.BIC = mhmmr.stats.loglik - (mhmmr.stats.nu*log(m)/2);
mhmmr.stats.AIC = mhmmr.stats.loglik - mhmmr.stats.nu;
% % CL(theta) : Completed-data loglikelihood
% sum_t_log_Pz_ftk = sum(mhmmr.stats.Zik.*log(state_probs.*mhmmr.stats.f_tk), 2);
% comp_loglik = sum(sum_t_log_Pz_ftk(K:end));
% mhmmr.stats.comp_loglik = comp_loglik;
% mhmmr.stats.ICL = comp_loglik - (nu*log(m)/2);


%% predicted, filtered, and smoothed time series

for k = 1:K
    mhmmr.stats.regressors(:,:,k) = X*mhmmr.reg_param.betak(:,:,k);
end

% prediction probs   = Pr(z_t|y_1,...,y_{t-1})
predict_prob = zeros(m,K);
predict_prob(1,:) = mhmmr.prior;%t=1 p (z_1)
predict_prob(2:end,:) = (mhmmr.stats.alpha_tk(1:end-1,:)*mhmmr.trans_mat)./(sum(mhmmr.stats.alpha_tk(1:end-1,:), 2)*ones(1,K));%t =2,...,n
mhmmr.stats.predict_prob = predict_prob;
% predicted observations
for k = 1:K
    predicted(:,:,k) = (predict_prob(:,k)*ones(1,d)).*mhmmr.stats.regressors(:,:,k);%pond par les probas de prediction
end
% mhmmr.stats.predicted = sum(predict_prob.*mhmmr.stats.regressors,3);%pond par les probas de prediction
mhmmr.stats.predicted = sum(predicted, 3);%pond par les probas de prediction

% filtering probs  = Pr(z_t|y_1,...,y_t)
filter_prob = mhmmr.stats.alpha_tk./(sum(mhmmr.stats.alpha_tk, 2)*ones(1,K));%normalize(alpha_tk,2);
mhmmr.stats.filter_prob = filter_prob;
% filetered observations
for k = 1:K
    filtered(:,:,k) = (filter_prob(:,k)*ones(1,d)).*mhmmr.stats.regressors(:,:,k);%pond par les probas de filtrage
end
% mhmmr.stats.filtered = sum(filter_prob.*mhmmr.stats.regressors, 3);%pond par les probas de filtrage
mhmmr.stats.filtered = sum(filtered, 3);%pond par les probas de filtrage

%%% smoothed observations
%mhmmr.stats.smoothed_regressors = (mhmmr.stats.tau_tk).*(mhmmr.stats.regressors);
for k = 1:K
    smoothed(:,:,k) = (mhmmr.stats.tau_tk(:,k)*ones(1,d)).*mhmmr.stats.regressors(:,:,k);%pond par les probas de filtrage
end
% mhmmr.stats.smoothed = sum(mhmmr.stats.smoothed_regressors, 3);
mhmmr.stats.smoothed = sum(smoothed, 3);












