%%%%%%%%%%%%%%%%%
%%%%Validate Example1%%%%
%%%%%%%%%%%%%%%%%
clear all
close all
clc
%%%%%%%%%%%%%%%%%
%EDIT BEGIN%
%%%%%%%%%%%%%%%%%

filename='rho0.csv';

NOPID=3; %PID

ts = 50 * 10^-6; %Control sampling
taud = 5*ts; %Pseudo Differential Cutoff
td = 3/10 * ts; %Input Delay

%%%%%%%%%%%%%%%%%
%EDIT END%
%%%%%%%%%%%%%%%%%
num = csvread(filename,1,0);
tsfir = ts; 
NOFIR=length(num) - NOPID;
s = tf('s');
z=tf('z', ts);

CPID = num(1) * 1 + num(2) / s + num(3) * s / (1+taud*s);
CPID = c2d(CPID, ts, 'tustin');
cfir = @(n) z^(-n);
if NOFIR == 0
    CFIR = 1;
else
    CFIR = 0;
end
for i=1:NOFIR
    CFIR = CFIR + cfir(i) * num(NOPID+i);
end
C=CFIR + CPID;

on = 2 * pi * [0, 3950, 5400, 6100, 7100];
kappa = [1, -1, 0.4, -1.2, 0.9];
zeta = [0, 0.035, 0.015, 0.015, 0.06];
Kp = 3.7 * 10 ^ 7;
P = 0;
for i=1:length(on)
    k = kappa(i);
    z = zeta(i);
    o = on(i);
    P =P +  k / (s ^ 2 + 2 * z * o * s + o ^ 2);
end
P = P * Kp *exp(-td*s);
P = c2d(P, ts);

L = P * C;
S = minreal(1/(1+L));
T = 1 - S;

fig = figure;
nyquist(L);
xlim([-3 3])
ylim([-3 3])

fig = figure;
lpf = c2d(1/(1+1.5*taud*s), ts, 'Tustin');
step(T*lpf); %L/(1+L) * shaped input
title('Shaped Step Response');
xlabel('Time (s)')

fig = figure;
step(S*P); %P/(1+L)
title('Step Disturbance Response');
xlabel('Time (s)')


fig = figure;
bode(C);
title('Controller');