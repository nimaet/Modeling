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
Lc = 0.022;
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
% C_mat_ele_circuit = 1/R_ind * l_ele;
C_mat_ele_circuit = 0;
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
index_dof_total = 1:N_dof_eff;
index_boundary = [1 2 N_dof_eff];
N_dof_free = N_dof_eff - size(index_boundary,2);
index_dof_free = index_dof_total;
index_dof_free(index_boundary) = [];
stiffness_M_free = stiffness_M_eff(index_dof_free,index_dof_free);
mass_M_free = mass_M_eff(index_dof_free,index_dof_free);
damping_eff_M_free = damping_eff_M_eff(index_dof_free,index_dof_free);
%% Assemble matrices
% mass_M_augmented = sparse(N_dof_free * 2,N_dof_free * 2);
% mass_M_augmented(1:N_dof_free,1:N_dof_free) = speye(N_dof_free);
% mass_M_augmented(N_dof_free + (1:N_dof_free),N_dof_free + (1:N_dof_free)) = mass_M_free;
% stiffness_M_augmented = sparse(N_dof_free * 2,N_dof_free * 2);
% stiffness_M_augmented(1:N_dof_free,N_dof_free + (1:N_dof_free)) = speye(N_dof_free);
% stiffness_M_augmented(N_dof_free + (1:N_dof_free),1:N_dof_free) = stiffness_M_free;
% stiffness_M_augmented(N_dof_free + (1:N_dof_free),N_dof_free + (1:N_dof_free)) = 1j * damping_eff_M_free;
mass_M_augmented = sparse(N_dof_free * 2,N_dof_free * 2);
mass_M_augmented(1:N_dof_free,1:N_dof_free) = mass_M_free;
mass_M_augmented(1:N_dof_free,N_dof_free + (1:N_dof_free)) = damping_eff_M_free;
mass_M_augmented(N_dof_free + (1:N_dof_free),N_dof_free + (1:N_dof_free)) = mass_M_free;
stiffness_M_augmented = sparse(N_dof_free * 2,N_dof_free * 2);
stiffness_M_augmented(1:N_dof_free,N_dof_free + (1:N_dof_free)) = -stiffness_M_free;
stiffness_M_augmented(N_dof_free + (1:N_dof_free),1:N_dof_free) = mass_M_free;
N_modes = 5;
x_beam_vec_FEM = 0:l_ele:l;
sigma = 1e-10; % Small shift for better numerical conditioning
[V, D] = eigs(stiffness_M_augmented, mass_M_augmented, N_modes*2, sigma, 'Tolerance', 1e-15);
% [V, D] = eigs(stiffness_M_augmented,mass_M_augmented,N_modes*2, 'smallestabs','Tolerance',1e-15);
% V_Deflection_raw = [0;real(V(1:2:79,1))];
% V_Slope_raw = [0;real(V(2:2:80,1))];
% plot(x_beam_vec_FEM,V_Deflection_raw)
%% Scaling factor
V_Deflection_1_raw = [0;real(V(1:2:79,1))];
V_Slope_1_raw = [0;real(V(2:2:80,1))];
Mod2 = 0;
for i_ele = 1:N_ele
    deflection_raw_1 = V_Deflection_1_raw(i_ele);
    deflection_raw_2 = V_Deflection_1_raw(i_ele+1);
    Slope_raw_1 = V_Slope_1_raw(i_ele);
    Slope_raw_2 = V_Slope_1_raw(i_ele+1);
    Mod2_ele = int_phi2(0,l_ele,deflection_raw_1,deflection_raw_2,Slope_raw_1,Slope_raw_2);
    Mod2 = Mod2 + Mod2_ele;
end
% V_Deflection = V_Deflection_raw / sqrt(m * Mod2);
% V_Slope = V_Slope_raw / sqrt(m * Mod2);
%% Beam frequency at the mode
stiffness_M_beam = sparse(N_ele * 2 + 2,N_ele * 2 + 2);
mass_M_beam = sparse(N_ele * 2 + 2,N_ele * 2 + 2);
for i_ele = 1:N_ele
    for i=1:4
        ii=index_ele_dof(i,i_ele);
        for j=1:4
            jj=index_ele_dof(j,i_ele);
            stiffness_M_beam(ii,jj)=stiffness_M_beam(ii,jj)+K_mat_ele_beam(i,j);
            mass_M_beam(ii,jj)=mass_M_beam(ii,jj)+M_mat_ele_beam(i,j);
        end
    end
end
V_disp_raw = [zeros(2,5);real(V(1:80,1:2:(2*N_modes-1)))];
V_disp = V_disp_raw/sqrt(m * Mod2);
V_Deflection = V_disp(1:2:(2*N_ele+1),:);
V_Slope = V_disp(2:2:(2*N_ele+2),:);
Mass_eff = V_disp' * mass_M_beam * V_disp;
Stiffness_eff = V_disp' * stiffness_M_beam * V_disp;
omega1_FEM = sqrt((V_disp' * stiffness_M_beam * V_disp)/(V_disp' * mass_M_beam * V_disp));
%%
phi_r_prime_l_2_FEM = V_Slope(1 + N_ele/2,:);
phi_r_prime_l_FEM = V_Slope(1 + N_ele,:);
% int_0_l_phi_r_FEM = 0;
int_0_l_phi_r_FEM = zeros(1,N_modes);
for i_mode = 1:N_modes
    for i_ele = 1:N_ele
        deflection_1_temp = V_Deflection(i_ele,i_mode);
        deflection_2_temp = V_Deflection(i_ele+1,i_mode);
        Slope_1_temp = V_Slope(i_ele,i_mode);
        Slope_2_temp = V_Slope(i_ele+1,i_mode);
        int_phi_ele_temp = int_phi(0,l_ele,deflection_1_temp,deflection_2_temp,Slope_1_temp,Slope_2_temp);
        int_0_l_phi_r_FEM(i_mode) = int_0_l_phi_r_FEM(i_mode) + int_phi_ele_temp;
    end
end
save FEM_modes_norm.mat int_0_l_phi_r_FEM phi_r_prime_l_FEM phi_r_prime_l_2_FEM omega1_FEM
% trapz(x_beam_vec_FEM,V_Deflection)
%% Analytical
x_beam_vec = 0:l/100:l;
x_beam_norm = lambda_r * x_beam_vec/l;
phi_r = sqrt(1/(m*l)) * (cos(x_beam_norm) - cosh(x_beam_norm) + ...
    (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * ...
    (sin(x_beam_norm) - sinh(x_beam_norm)));
phi_r_prime = lambda_r/l * sqrt(1/(m*l)) * (-sin(x_beam_norm) - sinh(x_beam_norm) + ...
    (sin(lambda_r) - sinh(lambda_r)) / (cos(lambda_r) + cosh(lambda_r)) * ...
    (cos(x_beam_norm) - cosh(x_beam_norm)));
plot(x_beam_vec,phi_r)
trapz(x_beam_vec,m * phi_r.^2)
%% Comparison
figure(1)
hold on
h11 = plot(x_beam_vec_FEM,V_Deflection(:,3),'LineWidth',1.5);
h12 = plot(x_beam_vec,-phi_r,'LineWidth',2,'LineStyle','--');
legend([h11 h12],'FEM','First mode wo piezo')
xlabel(['Position(m)'])
ylabel(['Modal amplitude'])
title(['Deflection'])

%%
figure(2)
hold on
h21 = plot(x_beam_vec_FEM,V_Slope,'LineWidth',1.5);
h22 = plot(x_beam_vec,-phi_r_prime,'LineWidth',2,'LineStyle','--');
legend([h21 h22],'FEM','First mode wo piezo')
xlabel(['Position(m)'])
ylabel(['Modal amplitude'])
title(['Slope'])
%%
% function int_phi2 = int_phi2(a,b,fa,fb,fpa,fpb)
% % Define interval [a, b] and function parameters
% % a = 0; b = 1;  % Change as needed
% % fa = 1; fb = 2;  % Function values at a and b
% % fpa = 0.5; fpb = -0.5;  % Slopes at a and b
% ba = b - a;  % Interval length
% 
% % Hermite interpolating function (symbolic for simplicity)
% syms t
% h0 = 1 - 3*t^2 + 2*t^3;
% h1 = t - 2*t^2 + t^3;
% h2 = 3*t^2 - 2*t^3;
% h3 = -t^2 + t^3;
% 
% f = h0*fa + h1*ba*fpa + h2*fb + h3*ba*fpb;
% 
% % Define Gauss quadrature nodes and weights for 4-point rule
% nodes = [-sqrt(3/7 + 2/7*sqrt(6/5)), -sqrt(3/7 - 2/7*sqrt(6/5)), ...
%           sqrt(3/7 - 2/7*sqrt(6/5)),  sqrt(3/7 + 2/7*sqrt(6/5))];
% weights = [(18 - sqrt(30))/36, (18 + sqrt(30))/36, ...
%            (18 + sqrt(30))/36, (18 - sqrt(30))/36];
% 
% % Transform nodes to [a, b]
% transformed_nodes = (b-a)/2 * nodes + (a+b)/2;
% 
% % Evaluate f(x)^2 at the transformed nodes
% f_squared = matlabFunction(f^2);  % Convert symbolic function to numeric
% values_at_nodes = f_squared(transformed_nodes);
% 
% % Compute the integral using Gauss quadrature
% integral_gauss = (b-a)/2 * sum(weights .* values_at_nodes);
% int_phi2 = integral_gauss;
% end
function int_phi2 = int_phi2(a,b,fa,fb,fpa,fpb)
ba = b - a;  % Interval length
% Hermite basis functions (normalized space t in [0, 1])
h0 = @(t) 1 - 3*t.^2 + 2*t.^3;
h1 = @(t) t - 2*t.^2 + t.^3;
h2 = @(t) 3*t.^2 - 2*t.^3;
h3 = @(t) -t.^2 + t.^3;

% Define the Hermite interpolation polynomial f(t)
f = @(t) h0(t)*fa + h1(t)*ba*fpa + h2(t)*fb + h3(t)*ba*fpb;

% Gauss quadrature nodes and weights for 4-point rule (normalized space)
nodes = [-sqrt(3/7 + 2/7*sqrt(6/5)), -sqrt(3/7 - 2/7*sqrt(6/5)), ...
          sqrt(3/7 - 2/7*sqrt(6/5)),  sqrt(3/7 + 2/7*sqrt(6/5))];
weights = [(18 - sqrt(30))/36, (18 + sqrt(30))/36, ...
           (18 + sqrt(30))/36, (18 - sqrt(30))/36];

% Transform Gauss nodes to normalized space [0, 1]
normalized_nodes = (nodes + 1) / 2;  % Map [-1, 1] to [0, 1]

% Evaluate f(t)^2 at the normalized nodes
values_at_nodes = f(normalized_nodes).^2;

% Compute the integral using Gauss quadrature (normalized space [0, 1])
integral_gauss = sum(weights .* values_at_nodes) * (b-a) / 2;
int_phi2 = integral_gauss;
end
function int_phi = int_phi(a,b,fa,fb,fpa,fpb)
ba = b - a;  % Interval length
% Hermite basis functions (normalized space t in [0, 1])
h0 = @(t) 1 - 3*t.^2 + 2*t.^3;
h1 = @(t) t - 2*t.^2 + t.^3;
h2 = @(t) 3*t.^2 - 2*t.^3;
h3 = @(t) -t.^2 + t.^3;

% Define the Hermite interpolation polynomial f(t)
f = @(t) h0(t)*fa + h1(t)*ba*fpa + h2(t)*fb + h3(t)*ba*fpb;

% Gauss quadrature nodes and weights for 4-point rule (normalized space)
nodes = [-sqrt(3/7 + 2/7*sqrt(6/5)), -sqrt(3/7 - 2/7*sqrt(6/5)), ...
          sqrt(3/7 - 2/7*sqrt(6/5)),  sqrt(3/7 + 2/7*sqrt(6/5))];
weights = [(18 - sqrt(30))/36, (18 + sqrt(30))/36, ...
           (18 + sqrt(30))/36, (18 - sqrt(30))/36];

% Transform Gauss nodes to normalized space [0, 1]
normalized_nodes = (nodes + 1) / 2;  % Map [-1, 1] to [0, 1]

% Evaluate f(t)^2 at the normalized nodes
values_at_nodes = f(normalized_nodes);

% Compute the integral using Gauss quadrature (normalized space [0, 1])
integral_gauss = sum(weights .* values_at_nodes) * (b-a) / 2;
int_phi = integral_gauss;
end