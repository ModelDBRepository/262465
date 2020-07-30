function asymmetric_new(P,Ni,N_left,Nrun,amp,NEP1,NEP2,k_von_left,k_von_right,EP,...
    wmax,A3p,A2d,g,g_inh)
% Code written by Xize Xu
% Please cite:
%Xize Xu, Jianhua 'JC' Cang and Hermann Riecke
% "Development and Binocular Matching of Orientation Selectivity in Visual Cortex: 
%A Computational Model"
%Journal of Neurophysiology, January 2020
%doi:10.1152/jn.00386.2019
%This code uses published codes from
%Clopath C, Vasilaki E, B�sing L and Gerstner W.
% "Connectivity reflects coding: A model of voltage-based
% spike-timing-dependent-plasticity with homeostasis"
% Nature Neuroscience, 13, 344-352, 2010
% and
% Clopath C, Gerstner W.
% "Voltage and Spike Timing interact in STDP - a Unified Model."
% Frontiers in Synaptic Neuroscience, doi:10.3389/fnsyn.2010.00025, 2010

% This code uses the voltage-based plasticity rule developed in Clopath
% et al. 2010 to simulate the development and binocuclar matching of the
% receptive fields through both eyes under monocular vision and binocular
% vision.
%(note that all parameters were set in advance on the basis of Clopath et al. (2010)
%except for the amplitude for potentation and depression, which are increased to speed up the simulations).
%As an example run try
%"asymmetric_new(250,500,250,8,3.03,250,2000,1.7,1.7,225,1.6,0.0007,0.0012,35,40)"

% This code calls the function aEIF.m which is the neuron model
% Simulaton parameters
% EP: number of time steps per simulated input [ms]. 
% NEP1: number of random visual inputs during monocular vision
% NEP2: number of random visual inputs during binocular vision
rand('seed',100)
N_right=Ni-N_left;
pre_orientation=[linspace(0,180,N_left),linspace(0,180,N_right)];
n_neur    = 1;                                  % number of downstream neurons
dt        = 1;                                  % duration of time step [ms]
T         = EP*(NEP1+NEP2);                     % number of time steps
tau_th    = 1.2*1000.0;                         % time constant for sliding threshold

% Plasticity model parameters
% parameters set in the function call:
% A2d: Depression Amplitude.  A3p: Potentiation Amplitude.  
theta_tar=110;                                   % scale for the homeostatic process
tau_p    = 7;                                    % time constant of the voltage low-pass for the potentiation term
tau_r    = 15;                                   % time constant for the presynaptic spike low-pass
tau_d    = 10;                                   % time constant of the voltage low-pass for the depression term

% Input parameters
% parameters set in the function call:
% P: number of input patterns
% k_von_left:argument of modified
% Bessel function for synaptic input through left eye
% k_von_right=20:argument of modified
% Bessel function for synaptic input through right eye
% wmax: max synaptic weights
nu_in_max = 0.015;                              % amplitude of synaptic input
nu_in_min = 0.0001;                             % baseline of synaptic input
pattern=90;
% Upstream monocular neurons' preferred orientations are linearly determined from 0 to pi. 
gau(1:N_left) = nu_in_min+1*amp*.3*nu_in_max*exp(cos((2*pre_orientation(1:N_left)-pattern)/180*pi)*k_von_left)/(2*pi*besseli(0,k_von_left));
gau(N_left+1:Ni) = nu_in_min+1*amp*.3*nu_in_max*exp(cos((2*pre_orientation(N_left+1:Ni)-pattern)/180*pi)*k_von_right)/(2*pi*besseli(0,k_von_right));
gau_left=[gau(1:N_left),gau(1:N_left)];
gau_right=[gau(N_left+1:Ni),gau(N_left+1:Ni)];
prespikeamount=0;% record the amount of inputs's spiking
for jj=1:P
    mup = 1+(jj-1)*N_left/P;
    mupp= 1+(jj-1)*N_right/P;
    pat_left(:,jj)=gau_left(mup:mup-1+N_left)';
    pat_right(:,jj)=gau_right(mupp:mupp-1+N_right)';
end
w_total=zeros(Ni, T/250+1,Nrun);% record the evolution of synaptic strength every 250 timesteps
for irun=1:Nrun
    fprintf(' This is run number %g \n',irun)
    clear right
    clear left
    clear momentary_v
    clear w_save
    p_left         = zeros(1,T);                         % generate random pattern locations
    p_right         = zeros(1,T);                        % generate random pattern locations
    for n=1:NEP1+NEP2
        if n==1
            p_left((n-1)*EP+1:n*EP) = 1;
            p_right((n-1)*EP+1:n*EP) = 1;
        else
            p_left((n-1)*EP+1:n*EP) = floor(P*rand)+1;
            p_right((n-1)*EP+1:n*EP) = floor(P*rand)+1;
        end
    end
    % Initialisation of the variables
    E_L      = -70.6;                                % resting potential
    w       =rand(Ni,n_neur)*wmax;                   %initial condition for w
    u        = E_L*ones(n_neur, 1);                  % membrane potential
    u_md     = E_L*ones(n_neur, 1);                  % low pas of u for the depression term
    u_mp     = E_L*ones(n_neur, 1);                  % low pas of u for the potentiation term
    uy       = 0*ones(n_neur, 1);                    % voltage thresholded
    u_sig_md = 0*ones(n_neur, 1);                    % low-pass of u thresholded for the depression term
    u_sig_mp = 0*ones(n_neur, 1);                    % low-pass of u thresholded for the potentiation term
    r       = zeros(Ni,1);                           % presyn low pass;xbar
    inp      = (rand(Ni,1)<[pat_left(:,p_left(1));pat_right(:,p_right(1))]);             % generation of the inputs
    C        = 281;                                  % membrane capacitance [pF]
    theta    = 0.0*ones(n_neur, 1);                  % sliding threshold, relative to resting potential
    u1s      = E_L*ones(n_neur, 1);                  % keeping voltage in memory for 2ms
    u1ss     = E_L*ones(n_neur, 1);                  % keeping voltage in memory for 2ms
    counter  = 0*ones(n_neur, 1);                    % initial values
    w_tail   = 0*ones(n_neur, 1);                    % initial values
    wad      = 0*ones(n_neur, 1);                    % initial values
    V_T      = -50.4*ones(n_neur, 1);                % initial values
    w_save = zeros(Ni, T/250+1);                     % save the weights for plotting the data
    momentary_v=zeros(1,T);
    E_rev=0;
    I_rev=-80;
    % g=40;
    for t=2:EP*NEP1
        inp = (rand(Ni,1)<[pat_left(:,p_left(t));pat_right(:,p_right(t))]);
        prespikeamount=prespikeamount+sum(inp);
        I = w'*inp*g*(E_rev-u1s)+g_inh*(I_rev-u1s); %input currents, g: exciatory synaptic conductance
        % g_inh: inhibitory synaptic conductance
        [u, wad,w_tail, counter, V_T] = aEIF(u, wad,w_tail, I,counter, V_T); % voltage, aEIF neuron model
        uy = ((u-E_L) > 25.3).*(u-E_L-25.3); % threshold of voltage
        r = (1-dt/tau_r)*r+dt/tau_r*inp; % low pass of presynaptic spike
        u_mp = u1ss*dt/tau_p +(1-dt/tau_p)*u_mp; % low pass of postsynaptic voltage for the potentiation term
        u_md = u1ss*dt/tau_d +(1-dt/tau_d)*u_md;% low pass of postsynaptic voltage for the depression term
        u_sig_md = ((u_md-E_L) > 0.).*(u_md-E_L-0.); % threshold the low pass
        u_sig_mp = ((u_mp-E_L) > 0.).*(u_mp-E_L-0.);% threshold the low pass
        theta = (1-dt/tau_th)*theta+dt/tau_th*((u1ss-E_L).^2); % homeostasis
        w = w + A3p*r*(u_sig_mp.*uy)'-A2d*inp*(u_sig_md.*(theta./theta_tar))'; % weight update
        w(w<0) = 0; % weight lower bound
        w(w>wmax) = wmax; % weight upper bound
        w_save(:,floor(t/250)+1) = w; % save the weights for plotting%%%
        u1s = u; % save the voltage 1ms before for the update order
        u1ss = u1s; % save the voltage 2ms before for the update order
        momentary_v(t)=u;
    end
    for t=EP*NEP1+1:T
        inp = (rand(Ni,1)<[pat_left(:,p_left(t));pat_right(:,p_left(t))]); % input pattern
        prespikeamount=prespikeamount+sum(inp);
        I = w'*inp*g*(E_rev-u1s)+g_inh*(I_rev-u1s); %input currents
        [u, wad,w_tail, counter, V_T] = aEIF(u, wad,w_tail, I,counter, V_T); % voltage, aEIF neuron model
        uy = ((u-E_L) > 25.3).*(u-E_L-25.3); % threshold of voltage
        r = (1-dt/tau_r)*r+dt/tau_r*inp; % low pass of presynaptic spike
        u_mp = u1ss*dt/tau_p +(1-dt/tau_p)*u_mp; % low pass of postsynaptic voltage for the potentiation term
        u_md = u1ss*dt/tau_d +(1-dt/tau_d)*u_md;% low pass of postsynaptic voltage for the depression term
        u_sig_md = ((u_md-E_L) > 0.).*(u_md-E_L-0.); % threshold the low pass
        u_sig_mp = ((u_mp-E_L) > 0.).*(u_mp-E_L-0.);% threshold the low pass
        theta = (1-dt/tau_th)*theta+dt/tau_th*((u1ss-E_L).^2); % homeostasis
        w = w + A3p*r*(u_sig_mp.*uy)'-A2d*inp*(u_sig_md.*(theta./theta_tar))'; % weight update
        w(w<0) = 0; % weight lower bound
        w(w>wmax) = wmax; % weight upper bound
        w_save(:,floor(t/250)+1) = w; % save the weights for plotting purpose
        u1s = u; % save the voltage 2ms before for the update order
        u1ss = u1s; % save the voltage 2ms before for the update order
        momentary_v(t)=u;
    end
    w_total(:,:,irun)=w_save;
    % Now plot the evolution of synaptic strength (similar as Fig 4A1, A2 in the paper)
    if 1==1
        figure(irun)
        subplot(2,1,1)
        imagesc(w_save(1:250,:))
        ylabel('Neuron index')
        xlabel('Time (s)')
        title(['Synaptic strength to the left eye, t_{switch}=',num2str(NEP1*EP/1000),' s'])
        xticks(1:400:4401)
        xticklabels({'0','100','200','300','400','500','600','700','800','900','1000','1100','1200',})
        colorbar
        hold all
        plot([NEP1*EP/250,NEP1*EP/250],[1,N_left],'w--')
        hold off
        subplot(2,1,2)
        imagesc(w_save(251:500,:))
        title(['Synaptic strength to the right eye, t_{switch}=',num2str(NEP1*EP/1000),' s'])
        ylabel('Neuron index')
        xlabel('Time (s)')
        xticks(1:400:4401)
        xticklabels({'0','100','200','300','400','500','600','700','800','900','1000','1100','1200',})
        colorbar
        hold all
        plot([NEP1*EP/250,NEP1*EP/250],[1,N_right],'w--')
        hold off
    end
    filename4=['Synaptic_evolution.mat'];
    save(filename4, 'w_total');
end

end

%%

function [u, w,w_tail,counter,V_T] = aEIF(u,w,w_tail,I,counter,V_T)

% Code written by Claudia Clopath
% Please cite:	
% Clopath C, Vasilaki E, B�sing L and Gerstner W.
% "Connectivity reflects coding: A model of voltage-based
% spike-timing-dependent-plasticity with homeostasis"
% Nature Neuroscience, 13, 344-352, 2010
% and
% Clopath C, Gerstner W.
% "Voltage and Spike Timing interact in STDP - a Unified Model."
% Frontiers in Synaptic Neuroscience, doi:10.3389/fnsyn.2010.00025, 2010

% This is the code to simulate the neuron model

% AEIF: Simulate adex model with spike after depolarization current and
% adaptive threshold
%
% Preconditions:
%  u            Membrane potential at time t
%  w            Adaptation variable
%  w_tail       Current for spike afterdepolarization
%  V_T          Adaptive threshold 
%  I            Input Current
% counter       Counter to force the spike to be clamped at the high value
%               for 2ms


% Model parameters
th = 20;        % [mV] spike threshold
C = 281;        % [pF] membrane capacitance
g_L = 30;       % [nS] membrane conductance
E_L = -70.6;    % [mV] resting voltage
VT_rest = -50.4;% [mV] resetting voltage
Delta_T = 2;    % [mV] exponential parameters
tau_w = 144;    % [ms] time constant for adaptation variable w
a = 4;          % [nS] adaptation coupling constant
b = 0.0805;     % [nA] spike triggered adaptation
w_jump = 400;   % [pA]spike after depolarisation;spike current after a spike
tau_wtail = 40; % [ms] time constant for spike after depolarisation;tau_z
tau_VT = 50;    % [ms] time constant for VT
VT_jump = 20;   % adaptive threshold;not threshold potential after a spike

if counter ==2          % trick to force the spike to be 2ms long��spike�����ʱ�̣����һ����
    u = E_L+15+6.0984;  % resolution trick (simulation of the spike at a fine resolution - see below)
    w = w+b;
    w_tail = w_jump;
    counter = 0;
    V_T = VT_jump+VT_rest;%��
end

% Updates of the variables for the aEIF
udot = 1/C*(-g_L*(u-E_L) + g_L*Delta_T*exp((u-V_T)/Delta_T) - w +w_tail+ I);
wdot = 1/tau_w*(a*(u-E_L) - w);
u= u + udot;
w = w + wdot;
w_tail = w_tail-w_tail/tau_wtail;
V_T = VT_rest/tau_VT+(1-1/tau_VT)*V_T;

if counter == 1%spike �ĵ�1�룬u�ı仯trick��, w����
    counter = 2;
    u = 29.4+3.462; % resolution trick (simulation of the spike at a fine resolution - see below)
    w = w-wdot;
end

if (u>th && counter==0) % threshold condition for the spike,spike �ĵ�0��
    u = 29.4;
    counter = 1;
end

% numerical trick for the aEIF model: I simulated, once and for all, the spike upswing and 
% integrated to know what is the integral of the spike, with high precision. Then I used this number in the simulation and 
% I clamped the spike for 2ms at the appropriated calculated value (this is to speed up the simulations 
% since network simulation is really time consuming). Since my time step in my simulation is 1ms, 
% I wait 3ms (the spike length (2ms) plus 1 time step (1ms)) before I read the filtered version of 
% the voltage (we want to read the value of the voltage trace before this spike). 

end