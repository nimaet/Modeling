% VT_sample = X2;
%     XT_sample = X1;
%     index_circuits_dof = I3;
    volt_sample = VT_sample(index_circuits_dof,:);
    flux_sample = XT_sample(index_circuits_dof,:);
    Energy_piezo = 1/2 * Cpj * volt_sample.^2 + 1/2 * Lj * flux_sample.^2 + 1/4 * Lcj * flux_sample.^4;
    size(Energy_piezo)
    pcolor(1:N_segs,Tspan_sample,Energy_piezo')
    shading interp
    xlim([1 100])
    ylim([0 0.1])
    caxis([0 2e-4])
    xlabel(['Piezo index'])
    ylabel(['Time (s)'])
    colorbar
    set(gca,'fontsize',18)
%     title(['Energy in shunt circuits'])