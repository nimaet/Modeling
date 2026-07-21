load FEM_modes_norm.mat
%% Structural Parameters
rho_p = 7800; % Density of piezos
rho_s = 8500; % Density of substrate
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
% omega1 = lambda1^2 * sqrt(EI/(m * l^4));
omega1 = omega1_FEM;
%% Circuit Parameters
Rc = 1e4;
Rb = 1e6;
Kp = -0.0066;
Ki = 185/2;
Kc = 5e5;
L_lin = Rc/Ki;
% Lc = Rc/Kc;
Lc = 0.017;
R = Rc * Rb / (Kp * Rb + Rc);
omega_t = sqrt(1/(Cp * L_lin));
% Electro-mechanical coupling
vartheta = b * e31 / 2 * (hp + hs); % In series
% Mode shape
lambda_r = lambda1;
phi_r_prime_l_2 = - phi_r_prime_l_2_FEM;
phi_r_prime_l = - phi_r_prime_l_FEM;
% phi_r_prime_l_2 = sqrt(1/(m*l^3)) * ...
%     (- lambda_r * sin(lambda_r/2) ...
%     - lambda_r * sinh(lambda_r/2) ...
%     + lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cos(lambda_r/2) ...
%     - lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cosh(lambda_r/2));
% phi_r_prime_l = sqrt(1/(m*l^3)) * ...
%     (- lambda_r * sin(lambda_r) ...
%     - lambda_r * sinh(lambda_r) ...
%     + lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cos(lambda_r) ...
%     - lambda_r * (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * cosh(lambda_r));
Chi_11 = vartheta * phi_r_prime_l_2;
Theta = Chi_11/(omega1 * sqrt(Cp));
Chi_12 = vartheta * (phi_r_prime_l - phi_r_prime_l_2);
lambda0 = sqrt(Lc * Cp * omega1^2);
eta0 = lambda0 * sqrt(Cp);
t0 = 1/omega1;
v0 = 0.8; % Amplitude of the voltage on the second piezo
% reference value: lambda0 * omega1
w0 = 0.5e-6; % Amplitude of the base excitation
% reference value: eta0 * sqrt(1/(m*l))
omega_b = 500; % Frequency of base excitation
omega_v = 502; % Frequency of modulator
% int_0_l_phi_r = 2/lambda_r * sqrt(l/m) * (sin(lambda_r) - sinh(lambda_r))/(cos(lambda_r) + cosh(lambda_r));
int_0_l_phi_r = - int_0_l_phi_r_FEM;
Fb = m * omega_b^2 * w0 / (omega1^2 * eta0) * int_0_l_phi_r;
Fv = Theta * Chi_12 * v0 / (Chi_11 * omega1 * lambda0);
% reference value: eta0 * sqrt(1/m*l)

% Normalized parameters
zeta = 1/(Cp * R * omega1);
Omega_f = omega_b/omega1;
Omega_f2 = omega_v/omega1;
Omega_t = omega_t/omega1;
%%
% A = zeta/2 * (1 - Omega_f^2)/(2 * 1j * Omega_f) - (Omega_t^2 - Omega_f^2) * (1 - Omega_f^2) / (4 * Omega_f^2) + Theta^2/4;
% B = - Theta/2 * Fb;
% C = - 3 * (1 - Omega_f^2) / (16 * Omega_f^4);
% Lambda_square = 0:0.001:0.35;
% Lambda_square_poly = C^2 * Lambda_square.^3 + (2 * real(A) * C) * Lambda_square.^2 + abs(A)^2 * Lambda_square;
% B_sol = sqrt(Lambda_square_poly);
% varphi_Lambda_1 = B_sol./(A + C * Lambda_square);
% plot(Lambda_square,abs(varphi_Lambda_1).^2)
%%
% for i_lambda = 1:size(Lambda_square,2)
%     varphi_Lambda_1_temp = varphi_Lambda_1(i_lambda);
%     Mat_A = - [(1 - Omega_f^2)/(2 * 1j * Omega_f) -Theta/2; Theta/2 zeta/2 + (Omega_t^2 - Omega_f^2)/(2 * 1j * Omega_f) - 3 * abs(varphi_Lambda_1_temp)^2 / (4 * Omega_f^3) * 1j];
%     Mat_B = - [0 0; 0 -3*(varphi_Lambda_1_temp)^2 / (8 * Omega_f^3) * 1j];
%     flag(i_lambda) = max(real(eig([Mat_A Mat_B;conj(Mat_B) conj(Mat_A)])))<0;
% end
%%
% figure(2)
% Fb_sol = B_sol / (- Theta/2);
% index = find(flag == 0);
% hold on
% h1 = plot(abs(Fb_sol),abs(varphi_Lambda_1),'k-',LineWidth=1.5);
% h2 = scatter(abs(Fb_sol(index)),abs(varphi_Lambda_1(index)));
% legend([h1 h2],'Analytical Results','Unstable branches')
% xlabel(['$|F_b|$'], 'Interpreter', 'latex')
% ylabel('${|\varphi_{\Lambda1}|}$', 'Interpreter', 'latex');
% set(gca,'fontsize',12)
% varphi_Lambda_1_vec = varphi_Lambda_1;

%% Single Fb lower branch IC

% Lambda_square_poly = C^2 * Lambda_square.^3 + (2 * real(A) * C) * Lambda_square.^2 + abs(A)^2 * Lambda_square - abs(B)^2 ;
Lambda_square_all = roots([C^2 2 * real(A) * C abs(A)^2 -abs(B)^2]);
Lambda_square_lower = Lambda_square_all(end);
varphi_Lambda_1 = B/(A + C * Lambda_square_lower);
varphi_Eta_1 = (Fb + Theta/2 * varphi_Lambda_1) / ((1 - Omega_f^2)/(2 * 1j * Omega_f));
H1_0 = (varphi_Eta_1 - conj(varphi_Eta_1))/(2 * 1j * Omega_f);
dH1_dT_0 = (varphi_Eta_1 + conj(varphi_Eta_1))/2;
Lambda1_0 = (varphi_Lambda_1 - conj(varphi_Lambda_1))/(2 * 1j * Omega_f);
dLambda1_dT_0 = (varphi_Lambda_1 + conj(varphi_Lambda_1))/2;
% x0 = [H1_0; dH1_dT_0; Lambda1_0; dLambda1_dT_0]; % [H1, dH1/dT, Lambda1, dLambda1/dT]
x0 = [0;0;0;0]; % [H1, dH1/dT, Lambda1, dLambda1/dT]
Tspan_simple = 0:0.2:10000;
% ODE function
% odefun = @(T, x) [
%     x(2); % dx1/dT = x2
%     -x(1) + Theta * x(4) + 2 * Fb * cos(Omega_f * T) + 2 * Fv * cos(Omega_f2 * T); % dx2/dT
%     x(4); % dx3/dT = x4
%     -zeta * x(4) - Omega_t^2 * x(3) - x(3)^3 - Theta * x(2) % dx4/dT
% ];
odefun = @(T, x) [
    x(2); % dx1/dT = x2
    -x(1) + Theta * x(4) + 2 * Fb * cos(Omega_f * T) + 2 * Fv * sin(Omega_f2 * T); % dx2/dT
    x(4); % dx3/dT = x4
    -zeta * x(4) - Omega_t^2 * x(3) - x(3)^3 - Theta * x(2) % dx4/dT
];

% Solve the system
[T, X] = ode45(odefun, Tspan_simple, x0);
figure(44)
hold on
plot(T,2 * Fb * cos(Omega_f * T) + 2 * Fv * sin(Omega_f2 * T))
% plot(T,(2 * Fb + 2 * Fv * sin((Omega_f2 - Omega_f) * T) .* cos(Omega_f * T) + 2 * Fv * cos((Omega_f2 - Omega_f) * T) .* sin(Omega_f * T))
% plot(T,((2 * Fb + 2 * Fv * sin((Omega_f2 - Omega_f) * T)) .^2 + (2 * Fv * cos((Omega_f2 - Omega_f) * T)).^2)
plot(T,sqrt(((2 * Fb + 2 * Fv * sin((Omega_f2 - Omega_f) * T)) .^2 + (2 * Fv * cos((Omega_f2 - Omega_f) * T)).^2)))
xlabel(['Time'])
ylabel(['Equivalent force'])
title(['Forcing'])
set(gca,'fontsize',12)
% Plot results
figure(4);
subplot(2, 1, 1);
plot(T, X(:, 2), 'LineWidth', 2);
title('$\frac{dH_1}{dT}$(T)', 'Interpreter', 'latex');
xlabel('Time T');
ylabel('$\frac{dH_1}{dT}$(T)', 'Interpreter', 'latex');
subplot(2, 1, 2);
plot(T, X(:, 4), 'LineWidth', 2);
title('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
xlabel('Time T');
ylabel('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
grid on;

% figure(42)
% varphi_Lambda_1_num = X(:, 4) + 1j * Omega_f * X(:, 3);
% plot(T, abs(varphi_Lambda_1_num))
% title('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
% xlabel('Time T');
% ylabel('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
% grid on;
%% Power flow
% P1_mag = eta0^2/t0^3 * zeta;
% P2_mag = eta0^2/t0^3 * Fv;
dEta1_dT = X(:, 2);
dLambda1_dT = X(:, 4);
P1 = zeta * dLambda1_dT.^2;
P2 = Fv * dEta1_dT * 2 * Fv .* sin(Omega_f2 * T) + Omega_f2 * (Fv * Chi_11/Theta/Chi_12)^2 * 4 * sin(Omega_f2 * T) .* cos(Omega_f2 * T);
figure(43)
hold on
h431 = plot(T,cumtrapz(T,P1),'LineWidth',1.5);
h432 = plot(T,cumtrapz(T,P2),'LineWidth',1.5);
legend([h431 h432],'Normalized Harvesting Energy','Normalized Modulation Input')
xlabel(['Normalized time'])
ylabel(['Normalized energy'])
set(gca,'fontsize',14)
ylim([-5 30])
