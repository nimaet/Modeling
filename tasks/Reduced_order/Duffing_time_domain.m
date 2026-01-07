clear; clc
%% ===================== Geometry =====================
L_b = 0.3185;                 % beam length [m]
w_p = 10e-3;                  % patch width
Q   = 31;
S   = Q;

xL = (0:S-1)*w_p;
xR = xL + w_p;

%% ===================== Beam Material =====================
rho_p = 8500;
rho_s = 2700;
E_p   = 100e9;
E_s   = 60e9;
b     = 10e-3;
hp    = 0.31e-3;
hs    = 0.607e-3;

m = b*(rho_s * hs + 2 * rho_p * hp);

term1 = E_s * hs^3 / 8;
term2 = E_p * ((hp + hs/2)^3 - hs^3/8);
YI = 2*b/3 * (term1 + term2);

e31  = -14.76;
eps33 = 14.8e-9;

A_term = (hp + 0.5*hs)^2 - (hs^2)/4;
theta_mech = e31 * b / hp * A_term;

%% ===================== Capacitance =====================
eps0 = 8.854e-12;
eps_r = 1930;
Cp_scalar = 2 * eps33 * w_p * b / hp;
Cp = Cp_scalar * ones(1,S);

%% ===================== Eigenvalues =====================
N = 50;

eigen_eq = @(lam) 1 + cos(lam).*cosh(lam);
lambda_vals = zeros(1,N);
for i = 1:N
    lam0 = pi*(i - 0.5);
    lambda_vals(i) = fsolve(eigen_eq, lam0, optimoptions('fsolve','Display','off'));
end

beta = lambda_vals / L_b;
omega = beta.^2 * sqrt(YI / m);
zeta  = 0.01 * ones(1,N);

%% ===================== Stable Sigma =====================
sigma_r = @(lam) ...
    (2*sin(lam).*exp(-lam) - 0.5 + 0.5*exp(-2*lam)) ./ ...
    (2*cos(lam).*exp(-lam) + 0.5 + 0.5*exp(-2*lam));

sigma_vals = sigma_r(lambda_vals);

%% ===================== Stable exp-term =====================
stable_exp_term = @(lam,x) ...
    0.5 * ( ...
        (sin(lam) + cos(lam) + exp(-lam)) ./ ...
        (2*cos(lam)*exp(-lam) + 0.5 + 0.5*exp(-2*lam)) ...
    ) .* exp(lam*(x./L_b - 1));

%% ===================== Mode Shapes =====================
mode_shape = @(r,x) ...
    sqrt(1/(m*L_b)) * ( ...
        cos(lambda_vals(r)*x/L_b) + ...
        sigma_vals(r)*sin(lambda_vals(r)*x/L_b) - ...
        stable_exp_term(lambda_vals(r),x) - ...
        0.5*(1 - sigma_vals(r)).*exp(-lambda_vals(r)*x/L_b) ...
    );

mode_shape_dx = @(r,x) ...
    sqrt(1/(m*L_b)) * ( ...
        (-sin(lambda_vals(r)*x/L_b))*(lambda_vals(r)/L_b) + ...
        sigma_vals(r)*cos(lambda_vals(r)*x/L_b)*(lambda_vals(r)/L_b) - ...
        stable_exp_term(lambda_vals(r),x)*(lambda_vals(r)/L_b) - ...
        (-0.5*(1 - sigma_vals(r)).*exp(-lambda_vals(r)*x/L_b))*(lambda_vals(r)/L_b) ...
    );

%% ===================== Coupling Matrix =====================
Gamma = zeros(N,S);
for r = 1:N
    for j = 1:S
        Gamma(r,j) = mode_shape_dx(r,xR(j)) - mode_shape_dx(r,xL(j));
    end
end

%% ===================== Excitation =====================
j_exc = 29;
A_exc = 50;
f0 = 1e3;
f1 = 5e3;
t_end = 0.1;

v_exc = @(t) A_exc * sin( 2*pi*(f0 + t*(f1-f0)/t_end) .* t );

%% ===================== Duffing + ODE =====================
R_b = 1e6;
R_c = 1e3;
K_p = 0.0003;
K_i = 0;
K_c = 0;

omega2 = omega.^2;
damp   = 2*zeta.*omega;



%% ===================== Solve =====================
x0 = zeros(2*N + 2*S,1);
dt = 1/(f1*20);
t_eval = 0:dt:t_end;

sol = ode45(@(t,x) odefun(t,x,N,S,omega2,damp,theta_mech,Gamma,Cp,K_p,K_i,R_c,v_exc,j_exc), ...
            [0 t_end],  x0);

y = deval(sol, t_eval);
eta = y(1:N,:);
eta_dot =  y(N+1:2*N,:);

%% ===================== Plot =====================
figure;
plot(t_eval, eta(1,:), 'LineWidth', 1.2);
xlabel('t [s]'); ylabel('\eta_1(t)');
grid on;
%% ===================== Reconstruct displacement and velocity =====================

x_eval = linspace(0, L_b, 50).';    % 50 spatial points (column)
M = length(x_eval);                 % number of spatial samples
T = length(t_eval);                 % number of time samples
disp_mat  = zeros(M, T);            % displacement
vel_mat   = zeros(M, T);            % velocity

for r = 1:N
    phi  = mode_shape(r, x_eval);      % M×1
    disp_mat  = disp_mat  + phi  * eta(r,:);      % M×T (outer product)
    vel_mat   = vel_mat   + phi  * eta_dot(r,:);  % M×T
end

%% ===================== Plot example =====================
figure;
plot(t_eval, vel_mat(10,:), 'LineWidth', 1.3);
xlabel('t [s]');
ylabel('Velocity');
grid on;
legend('Velocity at x = x_{10}');
%% ===================== Compute and plot spectrum =====================

% choose spatial index (same as Python's veloc[10,:])
idx_x = 5;
amp_mean = mean(abs(fft(vel_mat,[],2)), 1);  % 1×T mean spectrum
% y = vel_mat(idx_x, :);       % 1×T velocity signal
T = length(t_eval);
dt = t_eval(2) - t_eval(1);
fs = 1/dt;

% FFT
% Y = fft(y);
freq = (0:T-1)*(fs/T);       % MATLAB FFT frequency vector

% keep only positive frequencies
% pos = freq >= 0;
% freq_pos = freq(pos);
% Y_pos = Y(pos);

% amplitude spectrum
% amp = abs(Y_pos);

% Plot
figure;
semilogy(freq, amp_mean, 'b-', 'LineWidth', 1.5);
xlabel('Frequency [Hz]');
ylabel('Amplitude');
title('Velocity Spectrum at x = x_{eval}(10)');
xlim([1000 5000]);
grid on;

