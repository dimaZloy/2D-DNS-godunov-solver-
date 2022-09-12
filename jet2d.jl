

# Patm = 101325.0;
# M = 1.01;
# k = 1.4;
# Mw = 28.966;
# Rstar =  8.31432e+3;
# R = Rstar/Mw;

# P0 = 15.0*Patm
# T0 = 1000.0;

# P = P0*(1.0 + (k-1.0)*0.5*M*M)^(-k/(k-1.0))
# T = T0*(1.0 + (k-1.0)*0.5*M*M)^(-1.0)
# a = sqrt(k*R*T)
# U = M*a
# rho = P/R/T

# M2 = 0.0575;
# T2 = 300.0;
# P2 = Patm;
# a2 = sqrt(k*R*T2)
# U2 = M2*a2
# rho2 = P2/R/T2



cur_cell = zeros(Float64,4);
bnd_cell = zeros(Float64,4);

cur_cell[1] = 1.1766766855256956;
cur_cell[2] = 0.0;
cur_cell[3] = 0.0;
cur_cell[4] = 101325.0;


 # subsonic inflow boundary condition

 gamma = 1.4;

 Minf = 0.0575;
 Tinf = 300.0;
 Pinf = 101325.0;
 Pt = Pinf/(1.0 + 0.5*(gamma-1.0)*Minf*Minf)^(-gamma/(gamma-1.0));
 Tt = Tinf/(1.0 + 0.5*(gamma-1.0)*Minf*Minf)^(-1.0);

 Ui = sqrt(cur_cell[2]*cur_cell[2] + cur_cell[3]*cur_cell[3]);
 Ht = cur_cell[4]/cur_cell[1]*gamma/(gamma-1.0) + 0.5*Ui*Ui; 
 ci = sqrt(gamma*cur_cell[4]/cur_cell[1]); 
 Rp = -Ui - 2.0*ci/(gamma-1.0); 	
 a = 1.0 + 2.0/(gamma-1.0);
 b = 2.0*Rp;
 c = 0.5*(gamma-1.0)*(Rp*Rp - 2.0*Ht);
 cbp =  0.5/a * (-b + sqrt(b*b-4.0*a*c) );
 cbm =  0.5/a * (-b - sqrt(b*b-4.0*a*c) );

 cb = max(cbp, cbm);
 U = 2.0*cb/(gamma-1.0) + Rp;
 Mb = U/cb;
 Pb = Pt/(1.0 + 0.5*(gamma-1.0)*Mb*Mb)^(-gamma/(gamma-1.0));
 Tb = Tt/(1.0 + 0.5*(gamma-1.0)*Mb*Mb)^(-1.0);
 R = 8.31432e+3/28.966;

 bnd_cell[1] = Pb/R/Tb;
 bnd_cell[2] = U;
 bnd_cell[3] = 0.0;
 bnd_cell[4] = Pb;
