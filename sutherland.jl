
@everywhere @inline function calcSutherlandViscosityOF(T::Float64)::Float64

	return 1.4792e-6*sqrt(T)/(1.0 + 116.0/T);
	
end