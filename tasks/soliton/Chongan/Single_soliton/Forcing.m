%% Plane wave solutions normalized
clc; clear
Tspan_nm = 0:0.01:1000;
% Tspan = Tspan_nm * t0;
Lcj = 0.01;
Lj = 9.2237;
Amp = 0.2;
%% Beam parameters
b = 21e-3;
h_p = 0.55e-3;
h_s = 0.51e-3;
h_pc = (h_p + h_s)/2;
% L_unit = 22e-3;
rho_p = 7750;
rho_s = 2700;
m = b * (rho_s * h_s + 2 * rho_p * h_p);
Ys = 70e9;
epsilon_0 = 8.854e-12;
epsilon_33_T_rel = 2100;
epsilon_33_T = epsilon_0 * epsilon_33_T_rel;
s_11_E = 15.8e-12;
d_31 = -270e-12;
e_31 = d_31/s_11_E;
epsilon_33_s = epsilon_0 * epsilon_33_T_rel - (d_31)^2/s_11_E;
% Cp_eq_p = 2 * epsilon_33_s * b * L_unit / h_p;
% Cp = Cp_eq_p/L_unit;
Cp = 2 * epsilon_33_s * b / h_p;
vartheta_p = 2 * e_31 * b * h_pc;
EI = 2/3 * b * (Ys * h_s^3/8 + 1/s_11_E * ((h_p + h_s/2)^3 - h_s^3/8));
% vartheta_tilda = vartheta_p * (Cp * EI)^(-1/2);
% alpha = vartheta_tilda^2;
Delta_l = 1e-3;
Delta_L = 21e-3;
Cpj = Cp * Delta_L;
omega_t = 350 * 2 * pi;
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
lambda0 = sqrt(Lc_eff/L_eff);
w0 = sqrt(Cp_eff * Lc_eff / (m_eff * L_eff));
vartheta_tilda_eff = vartheta_eff * (Cp_eff * EI_eff)^(-1/2);
alpha_eff = vartheta_tilda_eff^2;
% Currently, we hit the optical branch with q = 0.95
q = 0.95;
Omega = sqrt(((1+alpha_eff)*q^4 + 1 + sqrt(((1+alpha_eff)*q^4 + 1).^2 - 4 * q^4))/2);
dOmega_dq = (4*q.^3 * (1+alpha_eff) .* Omega.^2 - 4*q.^3)...
    ./(4 * Omega.^3 - 2 * Omega .* ((1+alpha_eff) * q.^4 + 1));
vg = dOmega_dq;
ddOmega_dqdq = (16 * (1+alpha_eff) * q.^3 .* Omega .* dOmega_dq ...
    - 12 * Omega.^2 .* dOmega_dq.^2 ...
    + 2 * ((1+alpha_eff) * q.^4 + 1) .* dOmega_dq.^2 ...
    + 12 * (1+alpha_eff) * q.^2 .* Omega.^2 - 12 * q.^2)...
    ./(4 * Omega.^3 - 2 * Omega .* ((1+alpha_eff) * q.^4 + 1));
P = ddOmega_dqdq/2;
Q = 3./(2 * Omega .* (- Omega.^2 + 1) .* (1./(q.^4 - Omega.^2) + 1./(Omega.^2) + 1./(1 - Omega.^2)));
C_w1_abs = sqrt((1 - Omega^2)/(q^4 - Omega^2));
Tspan_nm_start_1 = -70;
Coeff_w = 1;
% Amp_dimen = Coeff_w * 2 * w0 * Amp * C_w1_abs;
Amp_dimen = Coeff_w * 2 * w0 * Amp;
k_spre_1 = Amp * sqrt(Q/(2*P)) * vg / t0;
Tspan = Tspan_nm * t0;
Tspan_dimen = Tspan / t0;
omega = Omega / t0;
phi1 = 0;
x01 = Amp * sqrt(Q/(2*P)) * vg * Tspan_nm_start_1; % Initial position is negative
Envelope_soliton_1 = sech(k_spre_1 * Tspan + x01);
Harmonic_soliton_1 = cos(omega * Tspan + phi1);
d_dt_Envelope_soliton_1 = - k_spre_1 * tanh(k_spre_1 * Tspan + x01) .* sech(k_spre_1 * Tspan + x01);
d_dt_Harmonic_soliton_1 = - omega * sin(omega * Tspan + phi1);
d2_dt2_Envelope_soliton_1 = - k_spre_1^2 * sech(k_spre_1 * Tspan + x01) .* (1 - 2 * sech(k_spre_1 * Tspan + x01));
d2_dt2_Harmonic_soliton_1 = - omega^2 * cos(omega * Tspan + phi1);
w0_boundary_1 = Amp_dimen * Envelope_soliton_1 .* Harmonic_soliton_1;
d_dt_w0_boundary_1 = Amp_dimen * d_dt_Envelope_soliton_1 .* Harmonic_soliton_1 + ...
    Amp_dimen * Envelope_soliton_1 .* d_dt_Harmonic_soliton_1;
d2_dt2_w0_boundary_1 = Amp_dimen * d2_dt2_Envelope_soliton_1 .* Harmonic_soliton_1 + ...
    2 * Amp_dimen * d_dt_Envelope_soliton_1 .* d_dt_Harmonic_soliton_1 + ...
    Amp_dimen * Envelope_soliton_1 .* d2_dt2_Harmonic_soliton_1;
x02 = x01 - 4; % Initial position is negative
phi2 = pi;
Envelope_soliton_2 = sech(k_spre_1 * Tspan + x02);
Harmonic_soliton_2 = cos(omega * Tspan + phi2);
d_dt_Envelope_soliton_2 = - k_spre_1 * tanh(k_spre_1 * Tspan + x02) .* sech(k_spre_1 * Tspan + x02);
d_dt_Harmonic_soliton_2 = - omega * sin(omega * Tspan + phi2);
d2_dt2_Envelope_soliton_2 = - k_spre_1^2 * sech(k_spre_1 * Tspan + x02) .* (1 - 2 * sech(k_spre_1 * Tspan + x02));
d2_dt2_Harmonic_soliton_2 = - omega^2 * cos(omega * Tspan + phi2);
w0_boundary_2 = Amp_dimen * Envelope_soliton_2 .* Harmonic_soliton_2;
d_dt_w0_boundary_2 = Amp_dimen * d_dt_Envelope_soliton_2 .* Harmonic_soliton_2 + ...
    Amp_dimen * Envelope_soliton_2 .* d_dt_Harmonic_soliton_2;
d2_dt2_w0_boundary_2 = Amp_dimen * d2_dt2_Envelope_soliton_2 .* Harmonic_soliton_2 + ...
    2 * Amp_dimen * d_dt_Envelope_soliton_2 .* d_dt_Harmonic_soliton_2 + ...
    Amp_dimen * Envelope_soliton_2 .* d2_dt2_Harmonic_soliton_2;
% w0_boundary = w0_boundary_1 + w0_boundary_2;
% d_dt_w0_boundary = d_dt_w0_boundary_1 + d_dt_w0_boundary_2;
% d2_dt2_w0_boundary = d2_dt2_w0_boundary_1 + d2_dt2_w0_boundary_2;
w0_boundary = w0_boundary_1;
d_dt_w0_boundary = d_dt_w0_boundary_1;
d2_dt2_w0_boundary = d2_dt2_w0_boundary_1;
figure(1)
plot(Tspan,w0_boundary)
figure(2)
plot(Tspan,d_dt_w0_boundary)
figure(3)
plot(Tspan,d2_dt2_w0_boundary)

% save disp_BC_new.mat x0 w0_boundary d_dt_w0_boundary d2_dt2_w0_boundary Tspan

save Parameters_tunable_full.mat x0 w0_boundary d_dt_w0_boundary d2_dt2_w0_boundary Tspan