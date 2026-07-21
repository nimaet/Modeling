%% Structural Parameters
rho_p = 7800;
rho_s = 8500;
c_11_E = 66e9;
cs = 100e9;
e31 = -14.76;
varepsilon_33_s = 14.8e-9;
% Cp = 52e-9;
l = 58.5e-3;
b = 31.75e-3;
hp = 0.267e-3;
hs = 0.127e-3;
Cp = 1/2 * varepsilon_33_s * b * l/2 / hp; % In series, half length
% vartheta = e31 * b / 2 * (hp + hs);
m = b * (rho_s * hs + 2 * rho_p * hp);
EI = 2/3 * b * (cs * hs^3/8 + c_11_E * ((hp + hs/2)^3 - hs^3/8));
fun = @(x) (1 + cos(x) * cosh(x)); % function
x0 = 1; % initial point
lambda1 = fzero(fun,x0);
omega_1 = lambda1^2 * sqrt(EI/(m * l^4));
%% Circuit Parameters
Rc = 1e4;
Rb = 1e6;
Kp = -0.0066;
Ki = 185/2;
Kc = 5e5;
L_lin = Rc/Ki;
Lc = Rc/Kc;
R = Rc * Rb / (Kp * Rb + Rc);
omega_t = sqrt(1/(Cp * L_lin));
% Electro-mechanical coupling
vartheta = b * e31 / 2 * (hp + hs); % In series
% Mode shape
lambda_r = lambda1;
phi_r_prime_l_2 = sqrt(1/(m*l^3)) * ...
    (- lambda_r * sin(lambda_r/2) ...
    - lambda_r * sinh(lambda_r/2) ...
    + lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cos(lambda_r/2) ...
    - lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cosh(lambda_r/2));
phi_r_prime_l = sqrt(1/(m*l^3)) * ...
    (- lambda_r * sin(lambda_r) ...
    - lambda_r * sinh(lambda_r) ...
    + lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cos(lambda_r) ...
    - lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cosh(lambda_r));
Chi_11 = vartheta * phi_r_prime_l_2;
Theta = Chi_11/(omega1 * sqrt(Cp));
% Parameters
% omega_t = 599.4816;
% omega_1 = 566.4404;
% Theta = 0.4093;

% Coefficients of the quadratic eigenvalue equation
b = 1 + (omega_t^2 / omega_1^2) + Theta^2;
c = omega_t^2 / omega_1^2;

% Solve the quadratic equation for mu (lambda^2)
mu = roots([1, b, c]);  % Coefficients: a=1, b, c

% Compute eigenvalues (lambda)
lambda = [sqrt(mu), -sqrt(mu)];  % Handle both positive and negative roots

% Display results
disp('Eigenvalues (lambda):');
disp(lambda);

% Compute lambda^2 * omega_1
lambda_squared_contributions = abs(lambda) * omega_1;

disp('Values of lambda * omega_1:');
disp(lambda_squared_contributions);
