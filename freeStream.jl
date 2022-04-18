
# supersonic flow over a circular cylinder at M = 3.5

using Distributed;

include("sutherland.jl");

T = 300.0;
P = 101325.0;
R = 287.058;
rho = P/R/T;
gamma = 1.4;
a = sqrt(gamma*R*T);
M = 3.55;
U = a*M;
mu = calcSutherlandViscosityOF(T)
D = 1.0; 
Re = rho*U*D/mu

BL = 1/sqrt(Re)





