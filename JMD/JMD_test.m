% Clear the workspace, close all figures, and clear the command window
clc
clear
close all

% Load the synthetic jump data (borrowed from the JOT toolbox at the
% following link https://www.ipol.im/pub/art/2023/417/?utm_source=doi)
load('jump2.mat');

% Set the sampling frequency
SampFreq = length(f_vGT);

% Define the time vector
t = 0:1/SampFreq:1-1/SampFreq;

% Create synthetic signals with intersecting IFs
IF1 = 4;
Sig1 = 4 * cos(2 * IF1 * pi * t);
IF2 = 80;
Sig2 = 2 * cos(2 * IF2 * pi * t);
IF3 = 200;
Sig3 = 2 * cos(2 * IF3 * pi * t);

% Combine the signals and add noise
jump = 5 * f_vGT';
Sig = Sig1 + Sig2 + Sig3 + jump;
Sig = Sig + 0.1 * randn(size(jump)); % to add noise
%% Parameter settings for the JMD method
alpha = 5000; 
tol = 1e-6;
beta = 0.03;
b_bar = 0.45;

tau = 5;
  

% Perform Jump Plus Mode Decomposition (JMD)
tic
[u, v] = JMD(Sig, alpha, tau, beta, b_bar, 3, 0, tol);
toc


%% Visualization
figure

% Plot the input signal
subplot(511)
plot(t, Sig, 'linewidth', 1)
% xlabel('t (sec)')


% Plot the extracted modes and the original signals
subplot(512)
plot(t, u(1,:,end), 'r', 'linewidth', 0.5)
hold on
plot(t, Sig1, 'k-.', 'linewidth', 1)
% xlabel('t (sec)')

subplot(513)
plot(t, u(2,:,end), 'r', 'linewidth', 0.5)
hold on
plot(t, Sig2, 'k-.', 'linewidth', 1)
% xlabel('t (sec)')

subplot(514)
plot(t, u(3,:,end), 'r', 'linewidth', 0.5)
hold on
plot(t, Sig3, 'k-.', 'linewidth', 0.5)
% xlabel('t (sec)')

% Plot the extracted jump component
subplot(515)
plot(t, v, 'r', 'linewidth', 0.5)
hold on
plot(t, jump, 'k-.', 'linewidth', 1)
xlabel('t (sec)')

fontsize(figure(1), 10, "points")