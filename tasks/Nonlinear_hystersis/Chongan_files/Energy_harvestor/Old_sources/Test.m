clear; clc; close all;

%% Parameters
T = 1000;       % Number of time steps
dt = 0.01;      % Time step size
sigma = 1;      % Brownian motion intensity
gamma = 0.1;    % Damping coefficient
omega = 1.0;    % Natural frequency of the oscillator

%% Brownian Motion Simulation
xi = sigma * sqrt(dt) * randn(1, T); % Gaussian noise
x_bm = cumsum(xi); % Cumulative sum for Brownian motion

%% Damped Harmonic Oscillator Simulation
x_osc = zeros(1, T);
v = 0; % Initial velocity

for t = 2:T
    a = -omega^2 * x_osc(t-1) - gamma * v; % Acceleration
    v = v + a * dt; % Update velocity
    x_osc(t) = x_osc(t-1) + v * dt; % Update position
end

%% Plot Time Series
figure;
plot(1:T, x_bm, 'b', 'LineWidth', 1.5); hold on;
plot(1:T, x_osc, 'r', 'LineWidth', 1.5);
xlabel('Time Steps');
ylabel('Displacement');
legend('Brownian Motion', 'Damped Oscillation');
title('Comparison of Brownian Motion and Damped Oscillatory Motion');
grid on;

%% Compute Power Spectral Density (PSD)
fs = 1/dt; % Sampling frequency
[psd_bm, f_bm] = pwelch(x_bm, [], [], [], fs);
[psd_osc, f_osc] = pwelch(x_osc, [], [], [], fs);

%% Plot PSD
figure;
loglog(f_bm, psd_bm, 'b', 'LineWidth', 1.5); hold on;
loglog(f_osc, psd_osc, 'r', 'LineWidth', 1.5);
xlabel('Frequency (Hz)');
ylabel('Power Spectral Density');
legend('Brownian Motion', 'Damped Oscillation');
title('PSD Comparison: Brownian Motion vs. Oscillations');
grid on;
