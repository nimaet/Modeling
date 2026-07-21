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
omega1 = lambda1^2 * sqrt(EI/(m * l^4));
%% Circuit Parameters
Rc = 1e4;
Rb = 1e6;
Kp = -0.0066;
Ki = 185/2;
Kc = 5e5;
L_lin = Rc/Ki;
% Lc = Rc/Kc;
Lc = 0.017;
% Lc = 0.02
R = Rc * Rb / (Kp * Rb + Rc);
omega_t = sqrt(1/(Cp * L_lin));
% Electro-mechanical coupling
vartheta = b * e31 / 2 * (hp + hs); % In series
%%
Delta_L = l/2;
R_ind = R * Delta_L;
L_ind = L_lin * Delta_L;
Cp_unit = 1/2 * varepsilon_33_s * b / hp;
%% FEM Settings
N_ele = 40;
l_ele = l/N_ele; 
x_vec = 0 : l_ele : l;
N_nodes = size(x_vec,2);
N_dof = 2 * N_nodes + N_ele;
index_ele_dof_beam = [(1:2:2*N_ele-1);(2:2:2*N_ele);...
    (3:2:2*N_ele+1);(4:2:2*N_ele+2)];
index_ele_dof_circuit = 2*N_ele+2 + (1:N_ele);
index_ele_dof = [index_ele_dof_beam;index_ele_dof_circuit];

%% FEM element matrix
M_mat_ele_beam = m * (l_ele / 420) * [
    156,   22*l_ele,  54,   -13*l_ele;
    22*l_ele,  4*l_ele^2, 13*l_ele, -3*l_ele^2;
    54,    13*l_ele,  156,  -22*l_ele;
   -13*l_ele, -3*l_ele^2, -22*l_ele, 4*l_ele^2
];
K_mat_ele_beam = (EI / l_ele^3) * [
    12   6*l_ele  -12  6*l_ele;
    6*l_ele  4*l_ele^2 -6*l_ele  2*l_ele^2;
   -12  -6*l_ele  12  -6*l_ele;
    6*l_ele  2*l_ele^2 -6*l_ele  4*l_ele^2
];
M_mat_ele_circuit = Cp_unit * l_ele;
C_mat_ele_circuit = 1/R_ind * l_ele;
K_mat_ele_circuit = 1/L_ind * l_ele;
M_mat_ele = zeros(5,5);
K_mat_ele = zeros(5,5);
C_mat_ele = zeros(5,5);
M_mat_ele(1:4,1:4) = M_mat_ele_beam;
M_mat_ele(5,5) = M_mat_ele_circuit;
K_mat_ele(1:4,1:4) = K_mat_ele_beam;
K_mat_ele(5,5) = K_mat_ele_circuit;
% C_mat_ele_beam_2 = tau_0 * K_mat_ele_beam_2;
vartheta_mat_ele = - vartheta * [0;-1;0;1];
% C_mat_ele_2(1:4,1:4) = C_mat_ele_beam_2;
C_mat_ele(5,1:4) = - vartheta_mat_ele';
C_mat_ele(1:4,5) = vartheta_mat_ele;
C_mat_ele(5,5) = C_mat_ele_circuit;

%% FEM Integration
stiffness_M = sparse(N_dof,N_dof);
mass_M = sparse(N_dof,N_dof);
damping_eff_M = sparse(N_dof,N_dof);
for i_ele = 1:N_ele
    for i=1:5
        ii=index_ele_dof(i,i_ele);
        for j=1:5
            jj=index_ele_dof(j,i_ele);
            stiffness_M(ii,jj)=stiffness_M(ii,jj)+K_mat_ele(i,j);
            mass_M(ii,jj)=mass_M(ii,jj)+M_mat_ele(i,j);
            damping_eff_M(ii,jj)=damping_eff_M(ii,jj)+C_mat_ele(i,j);
        end
    end
end
%% Transfer Matrix
N_dof_eff = 2 * (N_ele + 1) + 2;
index_i = 1:N_dof;
index_j = [(1 : 2 * (N_ele + 1)) (N_dof_eff-1) * ones(1,N_ele/2) (N_dof_eff) * ones(1,N_ele/2)];
index_s = [ones(1 , N_dof)];
T = sparse(index_i,index_j,index_s,N_dof,N_dof_eff,N_dof);
% I may need to use Cholesky decomposition to increase the accuracy
% R_K = chol(stiffness_M, 'lower');
% R_M = chol(mass_M, 'lower');
% norm(R_M*R_M' - full(mass_M))
stiffness_M_eff = T' * stiffness_M * T;
mass_M_eff = T' * mass_M * T;
damping_eff_M_eff = T' * damping_eff_M * T;
%% DOF index
index_dof = zeros(2,N_dof_eff);
index_dof(1,:) = 1:N_dof_eff;
index_dof(2,1:2:2 * N_ele + 2-1) = 1; % DoF of beam deflection
index_dof(2,2:2:2 * N_ele + 2) = 2; % Dof of beam slope
index_dof(2,2 * N_ele + 2 + (1:2)) = 3; % Dof of Flux linkage
%% Boundary condition
Tspan = 0:1e-4:20;
w0 = 0.5e-6;
v0 = 0.5;
index_dof_total = 1:N_dof_eff;
index_boundary = [1 2 N_dof_eff];
N_dof_free = N_dof_eff - size(index_boundary,2);
index_dof_free = index_dof_total;
index_dof_free(index_boundary) = [];
omega_b = 500;
omega_v = 501;
wb = 2 * w0 * cos(omega_b * Tspan);
lambda2 = -2 * v0/omega_v * cos(omega_v * Tspan);
dot_wb = - 2 * omega_b * w0 * sin(omega_b * Tspan);
dot_lambda2 = 2 * v0 * sin(omega_v * Tspan);
ddot_wb = - 2 * omega_b^2 * w0 * cos(omega_b * Tspan);
ddot_lambda2 = 2 * omega_v * v0 * cos(omega_v * Tspan);
stiffness_M_free = stiffness_M_eff(index_dof_free,index_dof_free);
mass_M_free = mass_M_eff(index_dof_free,index_dof_free);
damping_eff_M_free = damping_eff_M_eff(index_dof_free,index_dof_free);
stiffness_M_boundary = stiffness_M_eff(index_dof_free,index_boundary);
mass_M_boundary = mass_M_eff(index_dof_free,index_boundary);
damping_eff_M_boundary = damping_eff_M_eff(index_dof_free,index_boundary);
%%
[eigenvectors, eigenvalues] = eig(full(stiffness_M_free), full(mass_M_free));

% Extract eigenvalues as a vector
eigenvalues = diag(eigenvalues);

% Find the smallest eigenvalue
[smallest_eigenvalue, index] = min(eigenvalues);
stiffness_M_free(end,end)
mass_M_free(end,end)
%% Initial condition
X0_deflection = 2 * w0 * ones(N_nodes, 1);
V0_deflection = zeros(N_nodes, 1);
X0_slope = zeros(N_nodes, 1);
V0_slope = zeros(N_nodes, 1);
X0_flux = zeros(2, 1);
V0_flux = zeros(2, 1);
index_deflection_dof = (find(index_dof(2,:) == 1));
index_slope_dof = (find(index_dof(2,:) == 2));
index_circuit_dof = (find(index_dof(2,:) == 3));
X0_total = zeros(N_dof_eff,1);
V0_total = zeros(N_dof_eff,1);
X0_total(index_deflection_dof) = X0_deflection;
X0_total(index_slope_dof) = X0_slope;
X0_total(index_circuit_dof) = X0_flux;
V0_total(index_deflection_dof) = V0_deflection;
V0_total(index_slope_dof) = V0_slope;
V0_total(index_circuit_dof) = V0_flux;
X0_IC = X0_total;
V0_IC = V0_total;
X0_IC(index_boundary) = [];
V0_IC(index_boundary) = [];
%% External force
f_ext_full_0 = zeros(N_dof_free,1);
wb_0 = wb(1);
lambda2_0 = lambda2(1);
dot_wb_0 = dot_wb(1);
dot_lambda2_0 = dot_lambda2(1);
ddot_wb_0 = ddot_wb(1);
ddot_lambda2_0 = ddot_lambda2(1);
X_boundary = [wb_0;0;lambda2_0];
dot_X_boundary = [dot_wb_0;0;dot_lambda2_0];
ddot_X_boundary = [ddot_wb_0;0;ddot_lambda2_0];
f_ext_full_0 = - mass_M_boundary * ddot_X_boundary - damping_eff_M_boundary * dot_X_boundary - stiffness_M_boundary * X_boundary;
% stiffness_M_boundary * X_boundary + stiffness_M_free * X0_IC
%% Nonlinear Force
Lambda_0 = 0;
f_nl_circuit_0 = 1/Lc * Lambda_0.^3;
f_nl_full_0 = zeros(N_dof_free,1);
f_nl_full_0(end) = f_nl_circuit_0';
J_nl_circuit_0 = 1/Lc * 3 * Lambda_0.^2;
J_nl_full_0 = spalloc(N_dof_free,N_dof_free,1);
J_nl_full_0(N_dof_free,N_dof_free) = J_nl_circuit_0;
a0_IC = mass_M_free \ ( - damping_eff_M_free * V0_IC - stiffness_M_free * X0_IC - f_nl_full_0 + f_ext_full_0);
%% Integration
%% Numerical integration
% Newmark-beta Parameters
beta = 0.25;
gamma = 0.5;
% Tspan = 0:0.005:100;
max_iter = 500;
dt = mean(diff(Tspan));
XT = zeros(size(X0_IC,1),size(Tspan,2));
VT = zeros(size(V0_IC,1),size(Tspan,2));
aT = zeros(size(a0_IC,1),size(Tspan,2));
XT(:,1) = X0_IC;
VT(:,1) = V0_IC;
aT(:,1) = a0_IC;
tol = 1e-12;
f_ext_full_temp = zeros(N_dof_free,1);
for i_n = 1:size(Tspan,2)-1
    i_n
    Xn = XT(:,i_n);
    Vn = VT(:,i_n);
    an = aT(:,i_n);
    X_hat = Xn + dt * Vn + (0.5 - beta) * dt^2 * an;
    V_hat = Vn + (1 - gamma) * dt * an;
    Xn1_temp = Xn;
    % External force
    wb_temp = wb(i_n + 1);
    lambda2_temp = lambda2(i_n + 1);
    dot_wb_temp = dot_wb(i_n + 1);
    dot_lambda2_temp = dot_lambda2(i_n + 1);
    ddot_wb_temp = ddot_wb(i_n + 1);
    ddot_lambda2_temp = ddot_lambda2(i_n + 1);
    X_boundary = [wb_temp;0;lambda2_temp];
    dot_X_boundary = [dot_wb_temp;0;dot_lambda2_temp];
    ddot_X_boundary = [ddot_wb_temp;0;ddot_lambda2_temp];
    f_ext_full_temp = - mass_M_boundary * ddot_X_boundary - damping_eff_M_boundary * dot_X_boundary - stiffness_M_boundary * X_boundary;

    for i_iter = 1:max_iter
        % Extract circuit variables
        Lambda_temp = Xn1_temp(end);
        % Nonlinear force full
        f_nl_circuit_temp = 1/Lc * Lambda_temp.^3;
        f_nl_full_temp = zeros(N_dof_free,1);
        f_nl_full_temp(end) = f_nl_circuit_temp;
        % Nonlinear force Jacobian
        J_nl_circuit_temp = 1/Lc * 3 * Lambda_temp.^2;
        J_nl_full_temp = spalloc(N_dof_free,N_dof_free,1);
        J_nl_full_temp(N_dof_free,N_dof_free) = J_nl_circuit_temp;
        R = mass_M_free * (Xn1_temp - X_hat) + gamma * dt * damping_eff_M_free * (Xn1_temp - X_hat) ...
            + beta * dt^2 * (damping_eff_M_free * V_hat + stiffness_M_free * Xn1_temp + f_nl_full_temp - f_ext_full_temp);
        J = mass_M_free + gamma * dt * damping_eff_M_free + beta * dt^2 * (stiffness_M_free + J_nl_full_temp);
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
plot(Tspan,VT(end,:))
plot(Tspan,XT(end,:))

plot(Tspan,XT(end-2,:))
lambda_ref = sqrt(Lc * Cp * omega1^2);
v_ref = omega1 * lambda_ref;
plot(Tspan * omega1,VT(end,:)/v_ref)
xlim([0 10000])
voltage = VT(end,:);
save 22.mat Tspan omega1 voltage v_ref