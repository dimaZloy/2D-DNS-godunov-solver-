
		using PyPlot;

		gamma  = 1.4;
		R = 287.0;
		
		Tf = 545;
		Pf = 94232.25;
		af = sqrt(gamma*R*Tf);
		Uf = 973.0;
		Mf = Uf/af;
		rhof = Pf/R/Tf;

		display("Fuel stream: ");
		println("Mf: \t", Mf)
		println("rhof: \t", rhof)
		
		

		To = 1475;
		Po = 94232.25;
		ao = sqrt(gamma*R*To);
		Uo = 1475.0;
		Mo = Uo/ao;
		Mo2 = Mo*Mo;
		rhoo = Po/R/To;

		display("Oxy stream: ");
		println("Mo: \t", Mo)
		println("rhoo: \t", rhoo)
		
		
		
		s  =  33.5995465*pi/180.0; ## flow angle 33 deg C;
		sin2 = sin(s)*sin(s)
		num = (gamma+1.0)*Mo2;
		det = 2.0*(Mo2*sin2-1.0)
		
		cot_a = tan(s)*( num/det - 1.0)
		a = acot(cot_a)*180.0/pi;
		
	
		display("wedge angle : ");
		display(a)
		
		Pb1 = Po* (2.0*gamma*Mo2*sin2-(gamma-1.0)) / (gamma+1.0)
		
		tb_n1 = 2.0*gamma*Mo*Mo*sin(s)*sin(s)-(gamma-1.0) 
		tb_n2 = (gamma-1.0)*Mo*Mo*sin(s)*sin(s) + 2.0
		tb_d1 = (gamma+1.0)*(gamma+1.0)*Mo*Mo*sin(s)*sin(s)
		
	
		Tb1 = To *(tb_n1*tb_n2)/(tb_d1 )


		# Tb = 1582.6;
		# Pb = 129951.0;
		# ab = sqrt(gamma*R*Tb);
		# Ubx = 1526.3;
		# Uby = 165.7;
		# Ub = sqrt(Ubx*Ubx + Uby*Uby)
		
		# Mb = Ub/ab;
		# rhob = Pb/R/Tb;
		# println("Tb:\t", Tb, "\t", Tb1)
		# println("Pb:\t", Pb, "\t", Pb1)
		# println("rhob:\t", rhob, "\t", rhob1)
		# println("ab:\t", ab, "\t", ab1)
		
		rhob_n1 = (gamma+1.0)*Mo2*sin2
		rhob_d1  = (gamma-1.0)*Mo2*sin2 + 2.0
		rhob1 = rhoo *  rhob_n1/rhob_d1 
		ab1 = sqrt(gamma*Pb1/rhob1)
		

		println("Bottom stream: ");

		
		println("Tb/To:\t", Tb1/To)
		println("Pb/Po:\t", Pb1/Po)
		println("rhob/rhoo:\t", rhob1/rhoo)
		println("ab/ao:\t", ab1/ao)
	

		delta = 1.44e-4;
		
		dyWedge =  -60.0*delta + 300.0*delta*tan(a*pi/180.0)
		println("wedge dy:\t", dyWedge)
		
		ymin = -60.0*delta;
		ymax =  60.0*delta;
		
		N = 100;
		dy = (ymax-ymin)/N;
		
		Ux = zeros(Float64,N);
		Uy = zeros(Float64,N);
		
		U1x = 1634.0;
		U1y = 0.0;
		U2x = 1526.0;
		U2y = 156.7;
		
		y = zeros(Float64,N);
		y[1] = ymin;
		
		for i = 2:N
			y[i] = y[i-1] + dy;
		end
		
		for i = 1:N
		
			Ux[i] = tanh(2.0*y[i]/delta)*(U1x-U2x)*0.5 + (U1x+U2x)*0.5;
			Uy[i] = tanh(2.0*y[i]/delta)*(U1y-U2y)*0.5 + (U1y+U2y)*0.5;
		
		end
		
		figure(199)
		clf()
		
		subplot(2,1,1)
		plot(Ux, y,"--k")
		
		subplot(2,1,2)
		plot(Uy, y, "--k")
		
