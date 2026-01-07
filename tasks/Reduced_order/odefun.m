%% ODE function
function dx = odefun(t,x,N,S,omega2,damp,theta_mech,Gamma,Cp,K_p,K_i,R_c,v_exc,j_exc)
    x = x(:);

    eta     = x(1:N);
    eta_dot = x(N+1:2*N);
    v       = x(2*N+1 : 2*N+S);
    z       = x(2*N+S+1 : 2*N+2*S);

    v(j_exc) = v_exc(t);

    eta_ddot = -damp(:).*eta_dot - omega2(:).*eta + theta_mech*(Gamma*v);

    v_dot = zeros(S,1);
    z_dot = zeros(S,1);

    for j = 1:S
        if j == j_exc
            v(j) = v_exc(t);
            v_dot(j) = 0;
            z_dot(j) = v_exc(t);
            continue;
        end

        strain_coupling = Gamma(:,j).' * eta_dot;
        num = (K_p/R_c)*v(j) + (K_i/R_c)*z(j) + theta_mech*strain_coupling;
        v_dot(j) = -num / Cp(j);
        z_dot(j) = v(j);
    end

    dx = [eta_dot; eta_ddot; v_dot; z_dot];
end