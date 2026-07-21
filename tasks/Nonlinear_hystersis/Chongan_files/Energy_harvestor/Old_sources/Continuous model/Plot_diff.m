subplot(2, 1, 2);
plot(T, X(:, 4), 'LineWidth', 2);
title('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
xlabel('Time T');
ylabel('$\frac{d\Lambda_1}{dT}$(T)', 'Interpreter', 'latex');
grid on;
hold on
plot(T, X(:, 4), 'LineWidth', 2);
plot(Tspan * omega1,VT(end,:)/v_ref)
%%
figure(58)
hold on
h581 = plot(T/omega1, X(:, 4) * v_ref, 'LineWidth', 2,'LineStyle','-.');
h582 = plot(Tspan,VT(end,:), 'LineWidth', 1.5,'LineStyle','-.');
xlabel(['Time (s)'])
ylabel(['Voltage(V)'])
title(['Response of the harvestor'])
legend([h581 h582],'ROM','FEM')
set(gca,'fontsize',12)