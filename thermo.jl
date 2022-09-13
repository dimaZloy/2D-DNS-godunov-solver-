
println("set thermo-physical properties ...");

@everywhere struct THERMOPHYSICS
	RGAS::Float64;
	Gamma::Float64;
	Cp::Float64;
	mu::Float64;
	kGas::Float64;		
end


@inline function calcAirViscosityPowerLawFromT(T::Float64)::Float64
	return 1.716e-5*(T/273.11)^(2/3);
end

@inline function calcAirCpLow(T::Float64)::Float64

	# a1::Float64 = 2898903.0;
	# a2::Float64 = -56496.26;
	# a3::Float64 = 1437.799;
	# a4::Float64 = -1.653609;
	# a5::Float64 = 0.003062254;
	# a6::Float64 = -2.279138e-6;
	# a7::Float64 = 6.272365e-10;

	return 2898903.0*T^(-2) + -56496.26*T^(-1) + 1437.799 + -1.653609*T + 0.003062254*T^(2) + -2.279138e-6*T^(3) + 6.272365e-10*T^(4);
end

@inline function calcAirCpHigh(T::Float64)::Float64

	# b1::Float64 = 6.932494e+7;
	# b2::Float64 = -361053.2;
	# b3::Float64 = 1476.665;
	# b4::Float64 = -0.06138349;
	# b5::Float64 = 2.027963e-5;
	# b6::Float64 = -3.075525e-9;
	# b7::Float64 = 1.888054e-13;

	return 6.932494e+7*T^(-2) + -361053.2*T^(-1) + 1476.665 + -0.06138349*T + 2.027963e-5*T^(2) + -3.075525e-9*T^(3) + 1.888054e-13*T^(4);
end

@inline function calcAirHeatCapacityFromT(T::Float64)::Float64

	if T < 200.0
		return calcAirCpLow(200.0)
	elseif T> 200.0 && T <1000.0
		return calcAirCpLow(T)
	elseif T >= 1000.0 && T < 6000.0	
		return calcAirCpHigh(T)
	else
		return calcAirCpHigh(6000.0)
	end
end


@inline function calcAirConductivityFromT(T::Float64, thermo::THERMOPHYSICS)::Float64

	return 15.0/4.0*8.314462/28.9647*calcAirViscosityPowerLawFromT(T)*(4.0/15.0*calcAirHeatCapacityFromT(T)*28.9647/8.314462+1.0/3.0);
end




#@everywhere thermo 
#@everywhere const thermoX = $thermo;




