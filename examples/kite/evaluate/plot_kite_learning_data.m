%% plotting of kite results
clear all;
close all;

%% sizes
nt = 1;
nx = 3;
np = 2;

offset = 1;
n_batches = 1;
path_to_data = '../data/2_uncertainties_NN/data_batch_';
% path_to_data = '../data/data_batch_';

% violationsl
max_viol = zeros(n_batches,1);

% Plot height constraint 
c_plot = linspace(-70*2*pi/360,70*2*pi/360,100);
h_min = 100;
L_tether = 400;
A = 300;
beta = 0;
E_0 = 5.0;
c_tilde = 0.028;

for i = offset:offset+n_batches-1
    
        im = readNPY([path_to_data,num2str(i),'.npy']);
        
        theta_real = im(:,2);
        phi_real = im(:,3);
        psi_real = im(:,4);
        
        theta_est= im(:,5);
        phi_est = im(:,6);
        psi_est = im(:,7);

        v_0 = im(1,9);
        u_tilde = im(:,8);
        E = E_0 - c_tilde .* u_tilde.^2;
        P_D = v_0^2/2.0;
        T_F_real = A * P_D .* cos(theta_real).^2 .* sqrt(E.^2+1) .* (cos(theta_real) .* cos(beta) + sin(theta_real) .* sin(beta) .* sin(phi_real));
        T_F_est = A * P_D .* cos(theta_est).^2 .* sqrt(E.^2+1) .* (cos(theta_est) .* cos(beta) + sin(theta_est) .* sin(beta) .* sin(phi_est));
        height_kite = L_tether * sin(theta_real) .* cos(phi_real);
        
        t = im(:,nt);
        
        states_real = im(:,2:4);
        states_est = im(:,5:7);
        
        u_control = im(:,8);
        
        param_real = im(:,9:10);
        param_est = im(:,11:12);
        
        %% 
        figure();hold on;box off;
        plot(states_real(:,2)*180/pi,states_real(:,1)*180/pi,'b-','Linewidth',2);
        plot(states_est(:,2)*180/pi,states_est(:,1)*180/pi,'r--','Linewidth',2);
        plot(c_plot.*360./(2.*pi),sinh(h_min./(cos(c_plot).*L_tether)).*360./(2*pi),'k','Linewidth',3)
        legend('Real','Estimated','Constraint')
        xlabel('\theta [degrees]','Fontsize',16);
        ylabel('\phi [degrees]','Fontsize',16)
        set(gca,'Fontsize',16)
%         
%         %% states and parameters
        figure();hold on;
       
        subplot(nx+np,1,1)
        plot(t,states_real(:,1),'b','Linewidth',2);hold on;
        plot(t,states_est(:,1),'r--','Linewidth',2);
        ylabel('\theta [degrees]','Fontsize',16);
        set(gca,'Fontsize',16)
        
        subplot(nx+np,1,2)
        plot(t,states_real(:,2),'b','Linewidth',2);hold on;
        plot(t,states_est(:,2),'r--','Linewidth',2);
        ylabel('\phi [degrees]','Fontsize',16);
        set(gca,'Fontsize',16)
        
        subplot(nx+np,1,3)
        plot(t,states_real(:,3),'b','Linewidth',2);hold on;
        plot(t,states_est(:,3),'r--','Linewidth',2);
        ylabel('\psi [degrees]','Fontsize',16);
        set(gca,'Fontsize',16)
        
        subplot(nx+np,1,4)
        plot(t,param_real(:,1),'b','Linewidth',2);hold on;
        plot(t,param_est(:,1),'r--','Linewidth',2);
        ylabel('E0 [-]','Fontsize',16)
        xlabel('Time [s]','Fontsize',16)
        set(gca,'Fontsize',16)
        
        subplot(nx+np,1,5)
        plot(t,param_real(:,2),'b','Linewidth',2);hold on;
        plot(t,param_est(:,2),'r--','Linewidth',2);
        legend('real','estimation')
        ylabel('wind [m/s]','Fontsize',16)
        xlabel('Time [s]','Fontsize',16)
        set(gca,'Fontsize',16)
        
        %% height kite and force
        figure();
        
        subplot(3,1,1)
        plot(t,T_F_real,'b','Linewidth',2);hold on;
        plot(t,T_F_est,'r--','Linewidth',2);
        ylabel('T_F [N]','Fontsize',16)
        set(gca,'Fontsize',16)
        
        subplot(3,1,2)
        plot(t,height_kite,'b','Linewidth',2);hold on
        plot(t,ones(size(t,1))*h_min,'r','Linewidth',3);
        ylabel('Height [m]','Fontsize',16)
        xlabel('Time [s]','Fontsize',16)
        set(gca,'Fontsize',16)
        legend('height of kite','minimal height')

        subplot(3,1,3)
        plot(t,u_control,'b','Linewidth',2);hold on
        plot(t,ones(size(t,1))*10.0,'r','Linewidth',3);
        plot(t,-ones(size(t,1))*10.0,'r','Linewidth',3);
        ylabel('u [-]','Fontsize',16)
        xlabel('Time [s]','Fontsize',16)
        set(gca,'Fontsize',16)
        
        %%
        max_viol(i+1-offset) = -min(height_kite-h_min);
        
end