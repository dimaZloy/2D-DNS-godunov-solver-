

Patm = 101325.0;
M = 1.01;
k = 1.4;
Mw = 28.966;
Rstar =  8.31432e+3;
R = Rstar/Mw;

P0 = 15.0*Patm
T0 = 1000.0;

P = P0*(1.0 + (k-1.0)*0.5*M*M)^(-k/(k-1.0))
T = T0*(1.0 + (k-1.0)*0.5*M*M)^(-1.0)
a = sqrt(k*R*T)
U = M*a
rho = P/R/T

M2 = 0.0575;
T2 = 300.0;
P2 = Patm;
a2 = sqrt(k*R*T2)
U2 = M2*a2
rho2 = P2/R/T2
