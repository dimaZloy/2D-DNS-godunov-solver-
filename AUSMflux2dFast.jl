

# AUSM+  flux splitting
# based on paperby Meng-Sing Liou "A Sequel to AUSM: AUSM+"
# JOURNAL OF COMPUTATIONAL PHYSICS 129, 364-382 (1996)
# there is a mistake in the article (I guess based on tests)
# in the equation 19b fro Mbetta - instead of 0.5 should use 0.25 !!!!!!!!!!!!!!!!!!!!
# find in this article:
# Azevedo, Korzenowski 
# An assessment of unstructured grid FV schemes for cold gas hypersonic flow calculations. 
# Journal of Aerospace Technology and Management, V1,n2,2009



#function get_gas_epsilon(p::Float64, rho::Float64, gamma::Float64)::Float64
# return p/rho/(gamma-1.0); 
#end

@inline  function P_m(M::Float64,AUSM_ALFA::Float64)::Float64	
	return (abs(M)>=1.0) ? 0.5*(1.0-sign(M)) : Palfa_m(M,AUSM_ALFA)
end

@inline  function P_p(M::Float64,AUSM_ALFA::Float64)::Float64
	return (abs(M)>=1.0) ? 0.5*(1.0+sign(M)) : Palfa_p(M,AUSM_ALFA)
end

@inline  function Palfa_m(M::Float64,AUSM_ALFA::Float64)::Float64
	return  0.25*(M-1.0)*(M-1.0)*(2.0+M)-AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

@inline  function Palfa_p(M::Float64,AUSM_ALFA::Float64)::Float64
	return  0.25*(M+1.0)*(M+1.0)*(2.0-M)+AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

@inline  function Mbetta_p(M::Float64,AUSM_BETTA::Float64)::Float64
	return  0.25*(M+1.0)*(M+1.0)+AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

@inline  function Mbetta_m(M::Float64,AUSM_BETTA::Float64)::Float64
	return -0.25*(M-1.0)*(M-1.0)-AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

@inline  function M_p(M::Float64,AUSM_BETTA::Float64)::Float64
	return (abs(M)>=1.0) ? 0.5*(M+abs(M)) : Mbetta_p(M,AUSM_BETTA)
end

@inline  function M_m(M::Float64,AUSM_BETTA::Float64)::Float64
	return (abs(M)>=1.0) ? 0.5*(M-abs(M)) : Mbetta_m(M,AUSM_BETTA)
end


#@inline @everywhere function compute_1D_ARBITRARY_INVISCID_AUSM_PLUS_FLUX_from_UPHYS(
#		uLeft::Array{Float64,2}, uRight::Array{Float64,2}, nx::Float64,  ny::Float64, side::Float64, gamma::Float64)
#	return compute_1D_ARBITRARY_INVISCID_AUSM_PLUS_FLUX_from_UPHYS(uLeft[1],uLeft[2],uLeft[3],uLeft[4], uRight[1], uRight[2],uRight[3],uRight[4], nx,ny,side, gamma); 
#end

# @inline  function AUSMplusFlux2d(
	# uLeft::Array{Float64,1}, uRight::Array{Float64,1}, nx::Float64,  
	# ny::Float64, side::Float64, gamma::Float64)::Array{Float64,1}
	
	# return compute_1D_ARBITRARY_INVISCID_AUSM_PLUS_FLUX_from_UPHYS(uLeft[1],uLeft[2],uLeft[3],uLeft[4], uRight[1], uRight[2],uRight[3],uRight[4], nx,ny,side, gamma); 
# end

# @inline  function AUSMplusFlux2df(
	# uLeft::Array{Float64,1}, uRight::Array{Float64,1}, nx::Float64,  
	# ny::Float64, side::Float64, gamma::Float64, flux::Array{Float64,1})
	
	# flux = compute_1D_ARBITRARY_INVISCID_AUSM_PLUS_FLUX_from_UPHYS(uLeft[1],uLeft[2],uLeft[3],uLeft[4], uRight[1], uRight[2],uRight[3],uRight[4], nx,ny,side, gamma); 
# end


# @inline  function AUSMplusFlux2dFast(
	# uLeft::Array{Float64,1}, uRight::Array{Float64,1}, nx::Float64,  
	# ny::Float64, side::Float64, gamma::Float64, flux::Array{Float64,1})
	
	# ##flux = compute_1D_ARBITRARY_INVISCID_AUSM_PLUS_FLUX_from_UPHYS(uLeft[1],uLeft[2],uLeft[3],uLeft[4], uRight[1], uRight[2],uRight[3],uRight[4], nx,ny,side, gamma); 
	
	 # computeAUSMfluxFast(uLeft[1],uLeft[2],uLeft[3],uLeft[4], uRight[1], uRight[2],uRight[3],uRight[4], nx, ny, side, gamma, flux);
	
# end


@inline function AUSMplusFlux2dFast(
				rhoL::Float64,	_UL::Float64, 	_VL::Float64, 	PL::Float64,
				rhoR::Float64,	_UR::Float64, 	_VR::Float64,	PR::Float64, 
				nx::Float64,  ny::Float64, side::Float64, 
				gamma::Float64, flux::Array{Float64,1})


	# VL_tilda::Float64   = _UL*nx + _VL*ny;
	# VR_tilda::Float64   = _UR*nx + _VR*ny;


	# TLeft::Float64      = _UL*ny - _VL*nx;
	# TRight::Float64     = _UR*ny - _VR*nx;

	# UMAG_L2::Float64 = (_UL*_UL + _VL*_VL);
	# UMAG_R2::Float64 = (_UR*_UR + _VR*_VR);

	# AUSM_BETTA::Float64 = 1.0/8.0;
	# AUSM_ALFA::Float64  = 3.0/16.0; 


	#htL =  get_gas_epsilon(PL,rhoL,gamma)+ 0.5*(UMAG_L2) +  PL/rhoL; 
	#htR =  get_gas_epsilon(PR,rhoR,gamma)+ 0.5*(UMAG_R2) +  PR/rhoR;

	# htL::Float64 =  PL/rhoL/(gamma-1.0) + 0.5*(UMAG_L2) +  PL/rhoL; 
	# htR::Float64 =  PR/rhoR/(gamma-1.0) + 0.5*(UMAG_R2) +  PR/rhoR;


	htL::Float64 =  PL/rhoL/(gamma-1.0) + 0.5*(_UL*_UL + _VL*_VL) +  PL/rhoL; 
	htR::Float64 =  PR/rhoR/(gamma-1.0) + 0.5*(_UR*_UR + _VR*_VR) +  PR/rhoR;
	
	##htN::Float64  = 0.5*(htL + htR - 0.5*(TLeft*TLeft + TRight*TRight));
	
	htN::Float64  = 0.5*(htL + htR - 0.5*((_UL*ny - _VL*nx)*(_UL*ny - _VL*nx) + (_UR*ny - _VR*nx)*(_UR*ny - _VR*nx)));

	# aL_dot::Float64 = sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htL);
	# aL_tilda::Float64 = aL_dot*min(1.0,aL_dot/abs(VL_tilda));
	# aR_dot::Float64 = sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htR);
	# aR_tilda::Float64 = aR_dot*min(1.0,aR_dot/abs(VR_tilda));
	# a12::Float64 = min(aL_tilda,aR_tilda);
	
	
	
	aL_tilda::Float64 = sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htL) * min(1.0, (sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htL))/abs( _UL*nx + _VL*ny ));
	aR_tilda::Float64 = sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htR) * min(1.0, (sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htR))/abs( _UR*nx + _VR*ny ));
	a12::Float64 = min(aL_tilda,aR_tilda);
	
	
	# MLeft::Float64  = VL_tilda/a12;
	# MRight::Float64 = VR_tilda/a12;
	
	#m_dot12::Float64 = M_p(MLeft,AUSM_BETTA)+M_m(MRight,AUSM_BETTA);
	#p12::Float64 = P_p(MLeft,AUSM_ALFA)*PL + P_m(MRight,AUSM_ALFA)*PR;
	
	# m_dot12::Float64 = M_p(MLeft,1.0/8.0)   + M_m(MRight,1.0/8.0);
	# p12::Float64     = P_p(MLeft,3.0/16.0)*PL + P_m(MRight,3.0/16.0)*PR;

	m_dot12::Float64 = M_p( (_UL*nx + _VL*ny)/a12,1.0/8.0)     + M_m( (_UR*nx + _VR*ny)/a12,1.0/8.0);
	p12::Float64     = P_p( (_UL*nx + _VL*ny)/a12,3.0/16.0)*PL  + P_m( (_UR*nx + _VR*ny)/a12,3.0/16.0)*PR;


	# m_dot12_p::Float64 = 0.5*(m_dot12+abs(m_dot12));
	# m_dot12_m::Float64 = 0.5*(m_dot12-abs(m_dot12));	
	
	
	flux[1] = -( a12*( 0.5*(m_dot12+abs(m_dot12))*rhoL      + 0.5*(m_dot12-abs(m_dot12))*rhoR    ) + 0.0    )*side;
	flux[2] = -( a12*( 0.5*(m_dot12+abs(m_dot12))*rhoL*_UL  + 0.5*(m_dot12-abs(m_dot12))*rhoR*_UR) + p12*nx )*side;
	flux[3] = -( a12*( 0.5*(m_dot12+abs(m_dot12))*rhoL*_VL  + 0.5*(m_dot12-abs(m_dot12))*rhoR*_VR) + p12*ny )*side;
	flux[4] = -( a12*( 0.5*(m_dot12+abs(m_dot12))*rhoL*htL  + 0.5*(m_dot12-abs(m_dot12))*rhoR*htR) + 0.0    )*side;
	
	# p = zeros(Float64,4);
	# F_LEFT = zeros(Float64,4);
	# F_RIGHT = zeros(Float64,4);
	##flux = zeros(Float64,4);
	
	# flux[1] = -( a12*(m_dot12_p*rhoL      + m_dot12_m*rhoR    ) + 0.0    )*side;
	# flux[2] = -( a12*(m_dot12_p*rhoL*_UL  + m_dot12_m*rhoR*_UR) + p12*nx )*side;
	# flux[3] = -( a12*(m_dot12_p*rhoL*_VL  + m_dot12_m*rhoR*_VR) + p12*ny )*side;
	# flux[4] = -( a12*(m_dot12_p*rhoL*htL  + m_dot12_m*rhoR*htR) + 0.0    )*side;
	
	
	##return flux; 
	
end








