load soft_cantilever_Boundary_optical_new.mat
%%
t0 = 4.9152e-04;
vg = 1.0117;
P = 3.6655;
Q = 0.9668;
Amp = 0.2;
N_piezo = 60;
flux_sample = XT_sample(index_circuits_dof,:);
Lcj = 0.003;
lambda_0 = sqrt(Lcj/Lj);
% plot(1:240,flux_sample(:,201)/lambda_0);
k_spre_1 = Amp * sqrt(Q/(2*P)) * vg / t0;
x01 = -5.1436;
x_bias = Amp * sqrt(Q/(2*P)) * N_piezo * 0.022/0.0223;
%%
x_bias_setting = -1;
Envelope_soliton_1 = sech(k_spre_1 * Tspan_sample + x01 - x_bias - x_bias_setting);
Envelope_soliton_2 = sech(k_spre_1 * Tspan_sample + x01 - x_bias - 6 - x_bias_setting);

hold on
% plot(Tspan_sample/t0,Amp * 2 * Envelope_soliton_1);
h1 = plot(Tspan_sample,Amp * 2 * Envelope_soliton_1 * lambda_0,'color','r','linewidth',1.5);

% plot(Tspan_sample/t0,Amp * 2 * Envelope_soliton_2);
% plot(Tspan_sample/t0,flux_sample(N_piezo,:)/lambda_0);
h2 = plot(Tspan_sample,flux_sample(N_piezo,:),'color','b','linewidth',1);
legend([h1 h2],'Analytical Envelope','Solitary response','box','off')
% title(['Response of the 60th piezo'])
xlabel('Time (s)')
ylabel('Flux linkage (V*s)')
set(gca,'fontsize',20)
xlim([0 0.1])
ylim([-0.015 0.015])