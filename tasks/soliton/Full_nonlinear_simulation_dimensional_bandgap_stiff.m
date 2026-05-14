load ..\..\Parameters_tunable_full.mat Tspan M_boundary V_boundary Lj Lcj Omega t0
Lj_old = Lj;
Lcj_old = Lcj;
clear Lcj Lj
% clear Lcj
Lj = 6;
Lcj = 0.01;
omega_ext = Omega / t0;
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
%% FEM settings
L_unit = Delta_l + Delta_L;
dx = 5e-4; % dx must be divisible by (Delta_l/2), currently 5e-4
x_vec_seg = - L_unit/2 : dx : L_unit/2;
x_vec_l = - L_unit/2 : dx : - Delta_L/2;
x_vec_m = - Delta_L/2 : dx : Delta_L/2;
x_vec_r = Delta_L/2 : dx : L_unit/2;
N_nodes_l = size(x_vec_l,2);
N_nodes_m = size(x_vec_m,2);
N_nodes_r = size(x_vec_r,2);
N_nodes_seg = size(x_vec_seg,2);
N_ele_l = N_nodes_l - 1;
N_ele_m = N_nodes_m - 1;
N_ele_r = N_nodes_r - 1;
N_ele_seg = N_ele_l + N_ele_m + N_ele_r;
N_dof_seg = 2 * N_nodes_seg + N_ele_m;
index_ele_dof_l = [(1:2:2*N_ele_l-1);(2:2:2*N_ele_l);...
    (3:2:2*N_ele_l+1);(4:2:2*N_ele_l+2)];
index_ele_dof_m_beam = 2 * N_ele_l + [(1:2:2*N_ele_m-1);(2:2:2*N_ele_m);...
    (3:2:2*N_ele_m+1);(4:2:2*N_ele_m+2)];
index_ele_dof_r = 2 * (N_ele_l + N_ele_m) + [(1:2:2*N_ele_r-1);(2:2:2*N_ele_r);...
    (3:2:2*N_ele_r+1);(4:2:2*N_ele_r+2)];
index_ele_dof_m_circuit = 2 * N_nodes_seg + [1:N_ele_m];
index_ele_dof_m = [index_ele_dof_m_beam;index_ele_dof_m_circuit];

L_ele = dx;
%% Element matrices 1
zeta = 0.01;
tau_0 = 2 * zeta / omega_ext;
M_mat_ele_beam_1 = ms * (L_ele / 420) * [
    156,   22*L_ele,  54,   -13*L_ele;
    22*L_ele,  4*L_ele^2, 13*L_ele, -3*L_ele^2;
    54,    13*L_ele,  156,  -22*L_ele;
   -13*L_ele, -3*L_ele^2, -22*L_ele, 4*L_ele^2
];
K_mat_ele_beam_1 = (EI_s / L_ele^3) * [
    12   6*L_ele  -12  6*L_ele;
    6*L_ele  4*L_ele^2 -6*L_ele  2*L_ele^2;
   -12  -6*L_ele  12  -6*L_ele;
    6*L_ele  2*L_ele^2 -6*L_ele  4*L_ele^2
];
C_mat_ele_beam_1 = tau_0 * K_mat_ele_beam_1;
M_mat_ele_1 = M_mat_ele_beam_1;
K_mat_ele_1 = K_mat_ele_beam_1;
% C_mat_ele_1 = zeros(4,4);
C_mat_ele_1 = C_mat_ele_beam_1;

%% Element Matrices 2
M_mat_ele_beam_2 = mt * (L_ele / 420) * [
    156,   22*L_ele,  54,   -13*L_ele;
    22*L_ele,  4*L_ele^2, 13*L_ele, -3*L_ele^2;
    54,    13*L_ele,  156,  -22*L_ele;
   -13*L_ele, -3*L_ele^2, -22*L_ele, 4*L_ele^2
];
K_mat_ele_beam_2 = (EI_t / L_ele^3) * [
    12   6*L_ele  -12  6*L_ele;
    6*L_ele  4*L_ele^2 -6*L_ele  2*L_ele^2;
   -12  -6*L_ele  12  -6*L_ele;
    6*L_ele  2*L_ele^2 -6*L_ele  4*L_ele^2
];
M_mat_ele_circuit_2 = Cp * L_ele;
K_mat_ele_circuit_2 = 1/L_ind * L_ele;

M_mat_ele_2 = zeros(5,5);
K_mat_ele_2 = zeros(5,5);
C_mat_ele_2 = zeros(5,5);
M_mat_ele_2(1:4,1:4) = M_mat_ele_beam_2;
M_mat_ele_2(5,5) = M_mat_ele_circuit_2;
K_mat_ele_2(1:4,1:4) = K_mat_ele_beam_2;
K_mat_ele_2(5,5) = K_mat_ele_circuit_2;
C_mat_ele_beam_2 = tau_0 * K_mat_ele_beam_2;
vartheta_mat_ele_2 = - vartheta_p * [0;-1;0;1];
C_mat_ele_2(1:4,1:4) = C_mat_ele_beam_2;
C_mat_ele_2(5,1:4) = - vartheta_mat_ele_2';
C_mat_ele_2(1:4,5) = vartheta_mat_ele_2;
%% Assembly Unit
stiffness_M_seg = sparse(N_dof_seg,N_dof_seg);
mass_M_seg = sparse(N_dof_seg,N_dof_seg);
damping_eff_M_seg = sparse(N_dof_seg,N_dof_seg);

for i_ele = 1:N_ele_l
    for i=1:4
        ii=index_ele_dof_l(i,i_ele);
        for j=1:4
            jj=index_ele_dof_l(j,i_ele);
            stiffness_M_seg(ii,jj)=stiffness_M_seg(ii,jj)+K_mat_ele_1(i,j);
            mass_M_seg(ii,jj)=mass_M_seg(ii,jj)+M_mat_ele_1(i,j);
            damping_eff_M_seg(ii,jj)=damping_eff_M_seg(ii,jj)+C_mat_ele_1(i,j);
        end
    end
end
clear i j ii jj i_ele

for i_ele = 1:N_ele_m
    for i=1:5
        ii=index_ele_dof_m(i,i_ele);
        for j=1:5
            jj=index_ele_dof_m(j,i_ele);
            stiffness_M_seg(ii,jj)=stiffness_M_seg(ii,jj)+K_mat_ele_2(i,j);
            mass_M_seg(ii,jj)=mass_M_seg(ii,jj)+M_mat_ele_2(i,j);
            damping_eff_M_seg(ii,jj)=damping_eff_M_seg(ii,jj)+C_mat_ele_2(i,j);
        end
    end
end
clear i j ii jj i_ele

for i_ele = 1:N_ele_r
    for i=1:4
        ii=index_ele_dof_r(i,i_ele);
        for j=1:4
            jj=index_ele_dof_r(j,i_ele);
            stiffness_M_seg(ii,jj)=stiffness_M_seg(ii,jj)+K_mat_ele_1(i,j);
            mass_M_seg(ii,jj)=mass_M_seg(ii,jj)+M_mat_ele_1(i,j);
            damping_eff_M_seg(ii,jj)=damping_eff_M_seg(ii,jj)+C_mat_ele_1(i,j);
        end
    end
end
clear i j ii jj i_ele
%% Transfer Matrix
N_dof_seg_eff = 2 * (N_ele_seg + 1) + 1;
index_i = 1:N_dof_seg;
index_j = [(1 : 2 * (N_ele_seg + 1)) N_dof_seg_eff * ones(1,N_ele_m)];
index_s = [ones(1 , 2 * (N_ele_seg + 1)) ones(1,N_ele_m)];
T = sparse(index_i,index_j,index_s,N_dof_seg,N_dof_seg_eff,N_dof_seg);
% I may need to use Cholesky decomposition to increase the accuracy
% R_K = chol(stiffness_M, 'lower');
% R_M = chol(mass_M, 'lower');
% norm(R_M*R_M' - full(mass_M))
stiffness_M_unit_eff = T' * stiffness_M_seg * T;
mass_M_unit_eff = T' * mass_M_seg * T;
damping_eff_M_unit_eff = T' * damping_eff_M_seg * T;
%% Assembly Global
N_segs = 31;
N_dof_total = (N_dof_seg_eff - 2) * N_segs + 2;
N_dof_beam_total = (N_dof_seg_eff - 3) * N_segs + 2;
N_dof_circuit_total = N_segs;
stiffness_M = sparse(N_dof_total,N_dof_total);
mass_M = sparse(N_dof_total,N_dof_total);
damping_eff_M = sparse(N_dof_total,N_dof_total);
for i_segs = 1:N_segs
    Beam_dof = (i_segs - 1) * N_ele_seg * 2 + (1 : (N_nodes_seg * 2));
    Circuit_dof = N_segs * N_ele_seg * 2 + 2 + i_segs;
    Seg_dof = [Beam_dof Circuit_dof];
    stiffness_M(Seg_dof,Seg_dof) = stiffness_M(Seg_dof,Seg_dof) + stiffness_M_unit_eff;
    mass_M(Seg_dof,Seg_dof) = mass_M(Seg_dof,Seg_dof) + mass_M_unit_eff;
    damping_eff_M(Seg_dof,Seg_dof) = damping_eff_M(Seg_dof,Seg_dof) + damping_eff_M_unit_eff;
end
%% DOF index
index_dof = zeros(2,N_dof_total);
index_dof(1,:) = 1:N_dof_total;
index_dof(2,1:2:N_dof_beam_total-1) = 1; % DoF of beam deflection
index_dof(2,2:2:N_dof_beam_total) = 2; % Dof of beam slope
index_dof(2,N_dof_beam_total + (1:N_dof_circuit_total)) = 3; % Dof of Flux linkage
%% Boundary condition
% The current BC is fixed-free
bcdof = 2 * N_ele_seg * N_segs + [1 2];
% bcdof = sort([bcdof_beam bcdof_circuit],'ascend');
stiffness_M(bcdof,:) = [];
stiffness_M(:,bcdof) = [];
mass_M(bcdof,:) = [];
mass_M(:,bcdof) = [];
damping_eff_M(bcdof,:) = [];
damping_eff_M(:,bcdof) = [];
N_dof_eff = N_dof_total - size(bcdof,2);
index_dof(:,bcdof) = [];
index_deflection_dof = find(index_dof(2,:)==1);
index_slope_dof = find(index_dof(2,:)==2);
index_circuits_dof = find(index_dof(2,:)==3);
% N_dof_eff = N_dof_total - size(bcdof,2);
%% Initial condition
X0 = zeros(N_dof_eff, 1);
v0 = zeros(N_dof_eff, 1);
Lambda_0 = zeros(N_dof_circuit_total,1);
f_nl_circuit_0 = - 1/Lcj * Lambda_0.^3;
f_nl_full_0 = zeros(N_dof_eff,1);
f_nl_full_0(index_circuits_dof) = f_nl_circuit_0';
%% Circuit nonlinearity
index_deflection_dof = find(index_dof(2,:)==1);
index_slope_dof = find(index_dof(2,:)==2);
index_circuits_dof = find(index_dof(2,:)==3);
N_circuits_dof_eff = size(index_circuits_dof,2);
N_dof_circuit = N_segs;
J_nl_circuit_0_diag = - 1/Lcj * 3 * Lambda_0.^2;
J_nl_circuit_0 = sparse(1:N_circuits_dof_eff,1:N_circuits_dof_eff,J_nl_circuit_0_diag,N_circuits_dof_eff,N_circuits_dof_eff,N_circuits_dof_eff);
J_nl_full_0 = spalloc(N_dof_eff,N_dof_eff,N_dof_circuit);
J_nl_full_0(index_circuits_dof,index_circuits_dof) = J_nl_circuit_0;
%% Boundary Forces
% External force
f_ext_full_0 = zeros(N_dof_eff,1);
M_boundary_1 = M_boundary(1);
V_boundary_1 = V_boundary(1);
f_ext_full_0(1) = - V_boundary_1;
f_ext_full_0(2) = M_boundary_1;
a0 = mass_M \ ( - damping_eff_M * v0 - stiffness_M * X0 - f_nl_full_0 + f_ext_full_0);
%% Numerical integration
% Newmark-beta Parameters
beta = 0.25;
gamma = 0.5;
% Tspan = 0:0.005:100;
max_iter = 500;
dt = mean(diff(Tspan));
XT = zeros(size(X0,1),size(Tspan,2));
VT = zeros(size(v0,1),size(Tspan,2));
aT = zeros(size(a0,1),size(Tspan,2));
XT(:,1) = X0;
VT(:,1) = v0;
aT(:,1) = a0;
tol = 1e-12;
f_ext_full_temp = zeros(N_dof_eff,1);
for i_n = 1:size(Tspan,2)-1
    i_n
    Xn = XT(:,i_n);
    Vn = VT(:,i_n);
    an = aT(:,i_n);
    X_hat = Xn + dt * Vn + (0.5 - beta) * dt^2 * an;
    V_hat = Vn + (1 - gamma) * dt * an;
    Xn1_temp = Xn;
    % External force
    M_boundary_n1 = M_boundary(i_n + 1);
    V_boundary_n1 = V_boundary(i_n + 1);
    f_ext_full_temp(1) = - V_boundary_n1;
    f_ext_full_temp(2) = M_boundary_n1;
    for i_iter = 1:max_iter
        % Extract circuit variables
        Lambda_temp = Xn1_temp(index_circuits_dof);
        % Nonlinear force full
        f_nl_circuit_temp = - 1/Lcj * Lambda_temp.^3;
        f_nl_full_temp = zeros(N_dof_eff,1);
        f_nl_full_temp(index_circuits_dof) = f_nl_circuit_temp;
        % Nonlinear force Jacobian
        J_nl_circuit_temp_diag = - 1/Lcj * 3 * Lambda_temp.^2;
        J_nl_full_temp_diag = zeros(N_dof_eff,1);
        J_nl_full_temp_diag(N_dof_eff - N_dof_circuit + 1:N_dof_eff) = J_nl_circuit_temp_diag;
        J_nl_full_temp = sparse(1:N_dof_eff,1:N_dof_eff,J_nl_full_temp_diag,N_dof_eff,N_dof_eff,N_dof_eff);        % Residue and Jacobian
        R = mass_M * (Xn1_temp - X_hat) + gamma * dt * damping_eff_M * (Xn1_temp - X_hat) ...
            + beta * dt^2 * (damping_eff_M * V_hat + stiffness_M * Xn1_temp + f_nl_full_temp - f_ext_full_temp);
        J = mass_M + gamma * dt * damping_eff_M + beta * dt^2 * (stiffness_M + J_nl_full_temp);
        % Update
        % delta_u = - inv(J) * R;
        delta_Xn1 = - (J \ R);
        Xn1_temp = Xn1_temp + delta_Xn1;
        % Convergence Check
        if norm(delta_Xn1) < tol
            break
        end
        
        % Iteration check
        if i_iter == max_iter
            max_iter
            return
        end
    end
    % Proceed to the next time step
    XT(:,i_n + 1) = Xn1_temp;
    VT(:,i_n + 1) = V_hat + gamma/(beta*dt) * (Xn1_temp - X_hat);
    aT(:,i_n + 1) = (Xn1_temp - X_hat)/(beta*dt*dt);
end

x_vec_beam = - N_segs/2 * L_unit : dx : N_segs/2 * L_unit;
plot(x_vec_beam,[XT(index_deflection_dof,20000);0])
% plot(1:300,[XT(index_circuits_dof,20000)])

Tspan_sample = Tspan(1:100:end);
XT_sample = XT(:,1:100:size(Tspan,2));
VT_sample = VT(:,1:100:size(Tspan,2));
aT_sample = aT(:,1:100:size(Tspan,2));
save soft_cantilever_Boundary_optical.mat Tspan_sample XT_sample VT_sample aT_sample index_deflection_dof index_slope_dof index_circuits_dof Lj Lcj