function [u,v,omega] = JMD(signal, alpha, tau, beta, b_bar, K, init, tol)
% Jump Plus Mode Decomposition
% Author: Mojtaba Nazari
% m.nazari@ece.au.dk
% Initial release 07-2024
%
% Input and Parameters:
% ---------------------
% signal  - the time domain signal (1D) to be decomposed
% alpha   - the balancing parameter of the mode bandwidth
% beta    - the balancing parameter of the jump constraint (1/expected number of jumps)
% tau2    - the balancing parameter related to the β parameter (greater than 1)
% tau     - the dual ascent step (set to 0 for noisy signal)
% K       - the number of modes to be recovered
% init    - 0 = all omegas start at 0
%           1 = all omegas start uniformly distributed
%           2 = all omegas initialized randomly
% tol     - tolerance of convergence criterion; typically around 1e-6
%
% Output:
% -------
% u       - the collection of decomposed modes
% v       - jump component
%
%
% Copyright (c) 2024, version 1.0,
%
% This program is free software: you can redistribute it and/or modify it 
% under the terms of the GNU Affero General Public License as published by 
% the Free Software Foundation, either version 3 of the License, or 
% (at your option) any later version.
%
% This program is distributed in the hope that it will be useful, 
% but WITHOUT ANY WARRANTY; without even the implied warranty of 
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
% See the GNU Affero General Public License for more details.
%
% You should have received a copy of the GNU Affero General Public License 
% along with this program. If not, see <http://www.gnu.org/licenses/>.%
%
%
%
%
%	Acknowledgments: This code has been developed by extending the
%   VMD and JOT codes that have been made public at the following links:
%
%   https://www.mathworks.com/matlabcentral/fileexchange/44765-variational-mode-decomposition
%   by K. Dragomiretskiy, D. Zosso.
%
%   https://www.ipol.im/pub/art/2023/417/?utm_source=doi
%   by M. Huska.
%   
%
%
%   
%
%
% When using this code, please do cite:
% -----------------------------------------------
% Mojtaba Nazari, Anders Rosendal Korshøj, Naveed Ur Rehman, Jump Plus 
% AM-FM Mode Decomposition, IEEE Trans. on Signal Processing (in press)
% https://doi.org/10.48550/arXiv.2407.07800
%
% K. Dragomiretskiy, D. Zosso, Variational Mode Decomposition, IEEE Trans.
% on Signal Processing (in press) 
% http://dx.doi.org/10.1109/TSP.2013.2288675
% 
% M. Huska, A. Cicone, S.H. Kang, S. Morigi, 
% A two-stage signal decomposition into Jump, Oscillation and Trend using ADMM,
%  Image Processing On Line (IPOL), 
% https://doi.org/10.5201/ipol.2023.417



%% ---------- Preparations
shift=mean(signal);
signal=signal-mean(signal);
% Period and sampling frequency of input signal
save_T = length(signal);
fs = 1/save_T;

% extend the signal by mirroring
T = save_T;
f_mirror(1:T/2) = signal(T/2:-1:1);
f_mirror(T/2+1:3*T/2) = signal;
f_mirror(3*T/2+1:2*T) = signal(T:-1:T/2+1);
f = f_mirror;

% Time Domain 0 to T
T = length(f);
t = (1:T)/T;

% Spectral Domain discretization
freqs = t-0.5-1/T;

% Maximum number of iterations 
N = 2000;

% Alpha = alpha*ones(1,K);
a2=50;
t2=0.01:0.001:sqrt(2/a2);
phi1=(-a2/2) * (t2.^2) + (sqrt(2*a2)*t2);
phi=[phi1 ones(1,N-length(phi1))];
Alpha = alpha.*(phi);

% Construct and center f_hat
f_hat = fftshift((fft(f)));
f_hat_plus = f_hat;
f_hat_plus(1:T/2) = 0;

% matrix keeping track of every iterant // could be discarded for mem
u_hat_plus = zeros(N, length(freqs), K);

% Initialization of omega_k
omega_plus = zeros(N, K);
switch init
    case 1
        for i = 1:K
            omega_plus(1,i) = (0.5/K)*(i-1);
        end
    case 2
        omega_plus(1,:) = sort(exp(log(fs) + (log(0.5)-log(fs))*rand(1,K)));
    otherwise
        omega_plus(1,:) = 0;
end


% other inits
uDiff = tol+eps; % update step
n = 1; % loop counter
sum_uk = 0; % accumulator



% ===================== Jump part
% Define and calculate b using b_bar
b = 2 / (b_bar^2);
% Compute gamma using tau2 and b
gamma = tau * (0.5 * b * beta);
% Initialize vector v with zeros, of length T
v = zeros(1, T);
% Create a column vector d filled with ones, of length T
d = ones(T, 1);
% Construct a sparse diagonal matrix D with sub-diagonal and super-diagonal
D = spdiags([-d, d], [0, 1], T, T);
% Apply zero boundary condition to the last row of D
D(end, :) = 0;
% Compute the matrix product D' * D
DTD = D' * D;
% Initialize vector tt with zeros, of same size as vector f
x = zeros(T,1);
% Initialize rho with the same size as x
rho = x;
% Calculate the reciprocal of gamma
coef1 = 1 / gamma;
% Compute mu using gamma and beta
mu = 2 * beta / gamma;
% Create a sparse diagonal matrix SPDiag with d as the diagonal
SPDiag = spdiags(d, 0, T, T);
% matrix keeping track of every iterant // could be discarded for mem
j_hat_plus = zeros(N, length(freqs));



%%  ----------- Main loop for iterative updates


while ( uDiff > tol &&  n < N ) % not converged and below iterations limit

        % update first mode accumulator
% Initialize sum_uk for k=1
sum_uk = u_hat_plus(n,:,K) + sum_uk - u_hat_plus(n,:,1);

for k = 1:K
    if k > 1
        % Update the accumulator for k > 1
        sum_uk = u_hat_plus(n+1,:,k-1) + sum_uk - u_hat_plus(n,:,k);
    end

    % Update the spectrum of the mode through Wiener filter of residuals
    u_hat_plus(n+1,:,k) = (f_hat_plus - sum_uk - j_hat_plus(n,:))./(1 + Alpha(1,n)*(freqs - omega_plus(n,k)).^2);

    % Update omega 
        omega_plus(n+1,k) = (freqs(T/2+1:T) * (abs(u_hat_plus(n+1, T/2+1:T, k)).^2)') / sum(abs(u_hat_plus(n+1, T/2+1:T, k)).^2);

end



    % Back to time domain

    u_hat = zeros(T, K);

    for k = 1:K
        u_hat((T/2+1):T,k) = squeeze(u_hat_plus(n+1,(T/2+1):T,k));
        u_hat((T/2+1):-1:2,k) = squeeze(conj(u_hat_plus(n+1,(T/2+1):T,k)));
        u_hat(1,k) = conj(u_hat(end,k));
    end

    u = zeros(K,length(t));

    for k = 1:K
        u(k,:)=real(ifft(ifftshift(u_hat(:,k))));
    end


    % Update jump
    v = (1*SPDiag + gamma * DTD) \ ((gamma.*D'*x - D' * rho) + 1*f' - 1*(sum(u))');


    % Update variable x 
    Dv = D * v(:);

    h = (Dv + coef1.*rho);

    x = min( max(((1/(1 - mu*b)) * ones(size(abs(h)))) - ...
        ((mu*sqrt(2*b)/(1 - mu*b)) * ones(size(abs(h)))) ./ abs(h), 0), 1) .* h;




    % Dual ascent rho 
    rho = rho - gamma.*(x - Dv);

    v=v';
    v = v - (mean(v(:)) - mean(f(:)));


    % To frequency domain
    j_hat_plus(n+1,:) = fftshift(fft(v));
    j_hat_plus(n+1,1:T/2) = 0;


    
    % loop counter
    n = n+1;

    % convergence
    % Option 1:
    uDiff = eps;
    for i=1:K
        uDiff = uDiff + (1/T)*(u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i))*conj((u_hat_plus(n,:,i)-u_hat_plus(n-1,:,i)))';
    end
    uDiff=uDiff+((1/T)*(j_hat_plus(n,:)-j_hat_plus(n-1,:))*conj((j_hat_plus(n,:)-j_hat_plus(n-1,:)))');
    uDiff = abs(uDiff);
  

end


%%  Cleanup


% discard empty space if converged early
N = min(N,n);
omega = omega_plus(1:N,:);

% Signal reconstruction
u_hat = zeros(T, K);
u_hat((T/2+1):T,:) = squeeze(u_hat_plus(N,(T/2+1):T,:));
u_hat((T/2+1):-1:2,:) = squeeze(conj(u_hat_plus(N,(T/2+1):T,:)));
u_hat(1,:) = conj(u_hat(end,:));

u = zeros(K,length(t));

for k = 1:K
    u(k,:)=real(ifft(ifftshift(u_hat(:,k))));
end

% remove mirror part
u = u(:,T/4+1:3*T/4);
v = v(:,T/4+1:3*T/4);v=v+shift;


[~,loc]=sort(omega(end,:));
u=u(loc,:);
omega=omega(:,loc);

% % recompute spectrum
% clear u_hat;
% for k = 1:K
%     u_hat(:,k)=fftshift(fft(u(k,:)))';
% end

end
