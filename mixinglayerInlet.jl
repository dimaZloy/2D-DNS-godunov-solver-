

## need y, t, gamma;  

using PyPlot;
using Statistics;


function testUyPerturbations()

	gamma::Float64 = 1.4;
	delta::Float64 = 1.44e-4;
	ymin::Float64 = -60.0*delta;
	ymax::Float64 = 60.0*delta;

	N::Int64 = 100;
	dy::Float64 = (ymax-ymin)/N;
	y = zeros(Float64,N);
	Uy = zeros(Float64,N);

	y[1] = ymin;

	for i = 2:N
		y[i] = y[i-1] + dy;
	end

				
	U1::Float64 = 973.0;
	U2::Float64 = 1634.0;
	rho1::Float64 = 0.6025;
	rho2::Float64 = 0.22226;
	P1::Float64 = 94232.25;
	P2::Float64 = 94232.25;

	aSound1::Float64 = sqrt(gamma*P1/rho1);
	aSound2::Float64 = sqrt(gamma*P2/rho2);

	a1::Float64 = 0.05;
	a2::Float64 = 0.05;

	lambda::Float64 = 30.0;
	b::Float64 = 10.0;
	phi1::Float64 = 0.0;
	phi2::Float64 = pi*0.5;
	Uc::Float64 = (U1*aSound1 + U2*aSound2)/(aSound1+aSound2);
	display(Uc)
	Scale::Float64 = 0.01*Uc; 
	display(Scale)
	
	## period 
	T::Float64 = lambda/Uc;

	yhat = y .- (ymax-ymin)*0.5;

	#t::Float64 = rand(Float64); 
	
	t::Float64 = 0.0+2000.0*7.5e-5;

	for i = 1:N
		Uy[i] = Scale*(a1*cos(2*pi*1.0*t/T + phi1)*exp(-yhat[i]*yhat[i]/b) + a2*cos(2*pi*2.0*t/T + phi2)*exp(-yhat[i]*yhat[i]/b));
	end

	UyMean  = mean(Uy);
	display(UyMean);

	figure(1)
	clf();
	plot(Uy,y,"--r")
	xlabel("Uy")
	ylabel("y")
	xlim(UyMean*0.9, UyMean*1.1);
	

end


testUyPerturbations();



