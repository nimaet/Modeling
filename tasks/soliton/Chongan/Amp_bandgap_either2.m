Lj = 9.224; % Linear Inductance (H)
Lcj = 0.01; % Cubic nonlinear gain (inductance, Wb^3/A)
% Q_sgn = sign(Lcj);
omega_ext = 2.4109e+03; % Frequency (ra
%% Beam parameters
b = 21e-3; % Width
h_p = 0.55e-3; % Piezo thickness
h_s = 0.51e-3; % Substrate thickness
rho_p = 7750; % Piezo Density
rho_s = 2700; % Substrate Density
Ys = 70e9; % Substrate modulus
epsilon_0 = 8.854e-12;
epsilon_33_T_rel = 2100; % Relative permittivity
s_11_E = 15.8e-12; % Piezo elastic compliance constant
d_31 = -270e-12; % strain coefficient
Delta_l = 1e-3; % Length: piezo
Delta_L = 21e-3; % Length: gap between piezos
%% Calculations
h_pc = (h_p + h_s)/2;
m = b * (rho_s * h_s + 2 * rho_p * h_p);
epsilon_33_T = epsilon_0 * epsilon_33_T_rel;
e_31 = d_31/s_11_E;
epsilon_33_s = epsilon_0 * epsilon_33_T_rel - (d_31)^2/s_11_E;
% Cp_eq_p = 2 * epsilon_33_s * b * L_unit / h_p;
% Cp = Cp_eq_p/L_unit;
Cp = 2 * epsilon_33_s * b / h_p;
vartheta_p = 2 * e_31 * b * h_pc;
EI = 2/3 * b * (Ys * h_s^3/8 + 1/s_11_E * ((h_p + h_s/2)^3 - h_s^3/8));
% vartheta_tilda = vartheta_p * (Cp * EI)^(-1/2);
% alpha = vartheta_tilda^2;
Cpj = Cp * Delta_L;
% omega_t = 350 * 2 * pi;
L_ind = Lj * Delta_L;
%% Effective Parameters
mt = m;
ms = b * rho_s * h_s;
EI_s = 2/3 * b * Ys * h_s^3/8;
EI_t = EI;
C1 = EI_t * (Delta_l + Delta_L) / (EI_s * Delta_L + EI_t * Delta_l);
C2 = EI_s * (Delta_l + Delta_L) / (EI_s * Delta_L + EI_t * Delta_l);
theta1 = vartheta_p * Delta_L / (EI_s * Delta_L + EI_t * Delta_l);
theta2 = vartheta_p * Delta_l / (EI_s * Delta_L + EI_t * Delta_l);
m_eff = (ms * Delta_l + mt * Delta_L) / (Delta_l + Delta_L);
EI_eff = EI_s * Delta_l / (Delta_l + Delta_L) * C1^2 + EI_t * Delta_L / (Delta_l + Delta_L) * C2^2;
vartheta_eff = C2 * vartheta_p;
Cp_eff = Cpj / (Delta_l + Delta_L) + 2 * vartheta_p * theta2 * Delta_L / (Delta_l + Delta_L) - ...
    EI_s * Delta_l / (Delta_l + Delta_L) * theta1^2 - EI_t * Delta_L / (Delta_l + Delta_L) * theta2^2;
% Cp_eff = Cpj / (Delta_l + Delta_L) + EI_s * Delta_l / (Delta_l + Delta_L) * theta1^2 + EI_t * Delta_L / (Delta_l + Delta_L) * theta2^2;
L_eff = Lj * (Delta_l + Delta_L);
omega_t_eff = 1/sqrt(Cp_eff * L_eff);
Lc_eff = Lcj * (Delta_l + Delta_L);
x0 = (EI_eff * Cp_eff * L_eff / m_eff)^(1/4);
t0 = (Cp_eff * L_eff)^(1/2);
%% Compute amp-dependent values
% Amp_norm = 0.2;
alpha_eff =0.03;
Amp_norm_vec = 0:0.001:0.5;
Omega_Bandgap_up = 1;
Omega_Bandgap_down = sqrt(1/(1+alpha_eff));
Q_sgn = 1;
Omega = Omega_Bandgap_up;
q4 = (Omega.^4 - Omega.^2)./((1 + alpha_eff) * Omega.^2 - 1);
% Q_Bandgap_up = Q_sgn * 3./(2 * Omega .* (- Omega.^2 + 1) .* (1./(- Omega.^2 + q4) + 1./(Omega.^2) + 1./(- Omega.^2 + 1)));
Q_Bandgap_up = Q_sgn * 3/2;
Omega = Omega_Bandgap_down;
q4 = (Omega.^4 - Omega.^2)./((1 + alpha_eff) * Omega.^2 - 1);
Q_Bandgap_down = Q_sgn * 3./(2 * Omega .* (- Omega.^2 + 1) .* (1./(- Omega.^2 + q4) + 1./(Omega.^2) + 1./(- Omega.^2 + 1)));
Q_Bandgap_up_amp_stiff = Omega_Bandgap_up + 1/2 * Amp_norm_vec.^2 .* Q_Bandgap_up;
Q_Bandgap_down_amp_stiff = Omega_Bandgap_down + 1/2 * Amp_norm_vec.^2 .* Q_Bandgap_down;
Q_sgn = -1;
Omega = Omega_Bandgap_up;
q4 = (Omega.^4 - Omega.^2)./((1 + alpha_eff) * Omega.^2 - 1);
% Q_Bandgap_up = Q_sgn * 3./(2 * Omega .* (- Omega.^2 + 1) .* (1./(- Omega.^2 + q4) + 1./(Omega.^2) + 1./(- Omega.^2 + 1)));
Q_Bandgap_up = Q_sgn * 3/2;
Omega = Omega_Bandgap_down;
q4 = (Omega.^4 - Omega.^2)./((1 + alpha_eff) * Omega.^2 - 1);
Q_Bandgap_down = Q_sgn * 3./(2 * Omega .* (- Omega.^2 + 1) .* (1./(- Omega.^2 + q4) + 1./(Omega.^2) + 1./(- Omega.^2 + 1)));
Q_Bandgap_up_amp_soft = Omega_Bandgap_up + 1/2 * Amp_norm_vec.^2 .* Q_Bandgap_up;
Q_Bandgap_down_amp_soft = Omega_Bandgap_down + 1/2 * Amp_norm_vec.^2 .* Q_Bandgap_down;
Amp_vec = 2 * lambda0 * Amp_norm_vec;
if Lcj > 0
    lambda0 = sqrt(Lc_eff/L_eff);
    w0 = sqrt(Cp_eff * Lc_eff / (m_eff * L_eff));
    vartheta_tilda_eff = vartheta_eff * (Cp_eff * EI_eff)^(-1/2);
    alpha_eff = vartheta_tilda_eff^2;
    Omega_ext = omega_ext / omega_t_eff;
elseif Lcj < 0
    lambda0 = sqrt(-Lc_eff/L_eff);
    w0 = sqrt(Cp_eff * Lc_eff / (m_eff * L_eff));
    vartheta_tilda_eff = vartheta_eff * (Cp_eff * EI_eff)^(-1/2);
    alpha_eff = vartheta_tilda_eff^2;
    Omega_ext = omega_ext / omega_t_eff;
end
%%
figure(101)
if Lcj > 0
    hold on
    plot(Amp_vec * lambda0,Q_Bandgap_up_amp_stiff * omega_t_eff)
    plot(Amp_vec * lambda0,Q_Bandgap_down_amp_stiff * omega_t_eff)
    fill([Amp_vec * lambda0, fliplr(Amp_vec * lambda0)],...
    [Q_Bandgap_down_amp_stiff * omega_t_eff,...
    fliplr(Q_Bandgap_up_amp_stiff * omega_t_eff)],...
    [0.2, 0.2, 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
elseif Lcj < 0
    hold on
    plot(Amp_vec * lambda0,Q_Bandgap_up_amp_soft * omega_t_eff)
    plot(Amp_vec * lambda0,Q_Bandgap_down_amp_soft * omega_t_eff)
    fill([Amp_vec * lambda0, fliplr(Amp_vec * lambda0)],...
    [Q_Bandgap_down_amp_stiff * omega_t_eff,...
    fliplr(Q_Bandgap_up_amp_stiff * omega_t_eff)],...
    [0.2, 0.2, 0.2], 'FaceAlpha', 0.2, 'EdgeColor', 'none');
end
xlabel(['Soliton amplitude in flux linkage(Wb)'])
ylabel(['Frequency bandgap\omega (rad/s)'])
