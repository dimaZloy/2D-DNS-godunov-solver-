
# laminar flow over a circular cylinder at M = 0.2 and Re = 140

using Distributed
using Printf
using PyPlot

include("viscosityModels.jl");


thetaDeg = 3.75; 
thetaRad = thetaDeg*pi/180.0;

sigmaDeg = 30.8;
sigmaRad = sigmaDeg*pi/180.0;

xsep = 8e-2;
L = 0.2+2.3;

xB=[
-0.2
 0.0
 2.3
];

yB=[
0.0
0.0
0.0
];

yT = [
1.0
1.0
1.0
];


shockInX = [ 
	-0.2
	1.0/tan(sigmaRad)+-0.2
]

shockInY = [ 
	1.0
	0.0
]
wedgeX = [ 
	-0.2
	2.3
]

wedgeY = [ 
	1.0
	1.0-L*tan(thetaRad)
]


figure(1)
clf()

plot(xB,yB,"--r");
plot(xB,yT,"--r");

plot(shockInX,shockInY,"-b",linewidth = 1.5);
plot(wedgeX,wedgeY,"-g",linewidth = 1.5);

axis("equal")
xlabel("x")
ylabel("y")



T = 273.0;
P = 50000.0;
R = 287.058;
rho = P/R/T;
gamma = 1.4;
a = sqrt(gamma*R*T);
M = 2.15;
U = a*M;
mu = calcSutherlandViscosityOF(T)

D = 8e-2; 
Re = rho*U*D/mu

BL = 1/sqrt(Re)

println("The turbulent flow over a circular cylinder");
println("Free-stream conditions:");
@printf("===================================\n");
@printf("PInf:\t%.2f\n", P);
@printf("TInf:\t%.2f\n", T);
@printf("rhoInf:\t%.4f\n", rho);
@printf("muInf:\t%.7f\n", mu);
@printf("UInf:\t%.3f\n", U);
@printf("Re:\t%.0f\n", Re);
@printf("Ma:\t%.2f\n", M);
@printf("D[m]:\t%.2f\n", D);




