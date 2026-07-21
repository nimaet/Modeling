load FEM_modes_norm.mat
xi = 0.02;
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
v0 = 0.5; % Amplitude of the voltage on the second piezo
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
A = zeta/2 * (xi + (1 - Omega_f^2)/(2 * 1j * Omega_f)) ...
    + (Omega_t^2 - Omega_f^2)/(2 * 1j * Omega_f) * (xi + (1 - Omega_f^2)/(2 * 1j * Omega_f)) + Theta^2/4;
B = - Theta/2 * Fb;
% C = - 3 * (1 - Omega_f^2) / (16 * Omega_f^4);
C = - 3/(8 * Omega_f^3) * 1j * (xi + (1 - Omega_f^2)/(2 * 1j * Omega_f));
Lambda_square = 0:0.001:0.50;
Lambda_square_poly = (C*C') * Lambda_square.^3 + (C * A' + A * C') * Lambda_square.^2 + (A*A') * Lambda_square;
B_sol = sqrt(Lambda_square_poly);
varphi_Lambda_1 = B_sol./(A + C * Lambda_square);
% plot(Lambda_square,abs(varphi_Lambda_1).^2)
%%
for i_lambda = 1:size(Lambda_square,2)
    varphi_Lambda_1_temp = varphi_Lambda_1(i_lambda);
    Mat_A = - [(1 - Omega_f^2)/(2 * 1j * Omega_f)+xi -Theta/2; Theta/2 zeta/2 + (Omega_t^2 - Omega_f^2)/(2 * 1j * Omega_f) - 3 * abs(varphi_Lambda_1_temp)^2 / (4 * Omega_f^3) * 1j];
    Mat_B = - [0 0; 0 -3*(varphi_Lambda_1_temp)^2 / (8 * Omega_f^3) * 1j];
    flag(i_lambda) = max(real(eig([Mat_A Mat_B;conj(Mat_B) conj(Mat_A)])))<0;
end
%%
% figure(1)
% hold on
% plot(Lambda_square,Lambda_square_poly)
% plot(Lambda_square,ones(size(Lambda_square,2),1) * abs(B)^2)
% % xlabel(['\phi_{\Lambda1}^2'])
% % xlabel('\varphi', 'Interpreter', 'tex');
% xlabel('${|\varphi_{\Lambda1}|}^2$', 'Interpreter', 'latex'); % Displays ϕ
% ylabel('$B\bar{B}$', 'Interpreter', 'latex')
%%
figure(2)
Fb_sol = B_sol / (- Theta/2);
index = find(flag == 0);
hold on
h1 = plot(abs(Fb_sol),abs(varphi_Lambda_1),'k-',LineWidth=1.5);
h2 = scatter(abs(Fb_sol(index)),abs(varphi_Lambda_1(index)));
xlabel(['$|F_b|$'], 'Interpreter', 'latex')
ylabel('${|\varphi_{\Lambda1}|}$', 'Interpreter', 'latex');
title(['Hysteresis, \xi = ',num2str(xi)])
set(gca,'fontsize',12)
varphi_Lambda_1_vec = varphi_Lambda_1;
ylim_up = 0.8;
ylim([0 ylim_up])
Force_amp_eff_min = abs(abs(Fb)-abs(Fv));
Force_amp_eff_max = abs(abs(Fb)+abs(Fv));
L1 = line([Force_amp_eff_min Force_amp_eff_min],[0 ylim_up],'color','k','linestyle','--');
L2 = line([Force_amp_eff_max Force_amp_eff_max],[0 ylim_up],'color','k','linestyle','--');
legend([h1 h2 L1 L2],'Analytical Results','Unstable branches','Forcing amplitude lower bound','Forcing amplitude higher bound','location','southeast')
