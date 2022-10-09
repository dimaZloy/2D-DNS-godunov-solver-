


# AUSM+  flux splitting
# based on paperby Meng-Sing Liou "A Sequel to AUSM: AUSM+"
# JOURNAL OF COMPUTATIONAL PHYSICS 129, 364-382 (1996)
# there is a mistake in the article (I guess based on tests)
# in the equation 19b fro Mbetta - instead of 0.5 should use 0.25 !!!!!!!!!!!!!!!!!!!!
# find in this article:
# Azevedo, Korzenowski 
# An assessment of unstructured grid FV schemes for cold gas hypersonic flow calculations. 
# Journal of Aerospace Technology and Management, V1,n2,2009


# rewritten for CUDA

#function get_gas_epsilon(p::Float32, rho::Float32, gamma::Float32)::Float32
# return p/rho/(gamma-1.0); 
#end


function cuP_m(M::Float32,AUSM_ALFA::Float32)::Float32	
	return (CUDA.abs(M)>=1.0) ? 0.5*(1.0-CUDA.sign(M)) : cuPalfa_m(M,AUSM_ALFA)
end

function cuP_p(M::Float32,AUSM_ALFA::Float32)::Float32
	return (CUDA.abs(M)>=1.0) ? 0.5*(1.0+CUDA.sign(M)) : cuPalfa_p(M,AUSM_ALFA)
end

function cuPalfa_m(M::Float32,AUSM_ALFA::Float32)::Float32
	return  0.25*(M-1.0)*(M-1.0)*(2.0+M)-AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

function cuPalfa_p(M::Float32,AUSM_ALFA::Float32)::Float32
	return  0.25*(M+1.0)*(M+1.0)*(2.0-M)+AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

function cuMbetta_p(M::Float32,AUSM_BETTA::Float32)::Float32
	return  0.25*(M+1.0)*(M+1.0)+AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

function cuMbetta_m(M::Float32,AUSM_BETTA::Float32)::Float32
	return -0.25*(M-1.0)*(M-1.0)-AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

function cuM_p(M::Float32,AUSM_BETTA::Float32)::Float32
	return (CUDA.abs(M)>=1.0) ? 0.5*(M+CUDA.abs(M)) : cuMbetta_p(M,AUSM_BETTA)
end

function cuM_m(M::Float32,AUSM_BETTA::Float32)::Float32
	return (CUDA.abs(M)>=1.0) ? 0.5*(M-CUDA.abs(M)) : cuMbetta_m(M,AUSM_BETTA)
end



function kernel_AUSM2d(rhoL, _UL, _VL, PL, rhoR, _UR, _VR, PR, nx, ny, side, flux1, flux2, flux3, flux4, gammaV,slopeIndex)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;
    j = length(rhoL)*slopeIndex;

    if i <= length(rhoL) 

          # htL::Float32 =  PL[i]/rhoL[i]/(gammaV[i]-1.0) + 0.5*(_UL[i]*_UL[i] + _VL[i]*_VL[i]) +  PL[i]/rhoL[i]; 
          # htR::Float32 =  PR[i]/rhoR[i]/(gammaV[i]-1.0) + 0.5*(_UR[i]*_UR[i] + _VR[i]*_VR[i]) +  PR[i]/rhoR[i];

          # aL_til::Float32 = CUDA.sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htL) * min(1.0, (sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htL))/CUDA.abs( _UL[i]*nx[i] + _VL[i]*ny[i] ));
          # aR_til::Float32 = CUDA.sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htR) * min(1.0, (sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htR))/CUDA.abs( _UR[i]*nx[i] + _VR[i]*ny[i] ));
          # a12::Float32 = CUDA.min(aL_til,aR_til);
             
          # m_dot12::Float32 = cuM_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12,1.0/8.0)         + cuM_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12,1.0/8.0);
          # p12::Float32     = cuP_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12,3.0/16.0)*PL[i]  + cuP_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12,3.0/16.0)*PR[i];
     
          # flux1[i] = flux1[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]         + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i])        + 0.0       )*side[i];
          # flux2[i] = flux2[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_UL[i]  + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_UR[i]) + p12*nx[i] )*side[i];
          # flux3[i] = flux3[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_VL[i]  + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_VR[i]) + p12*ny[i] )*side[i];
          # flux4[i] = flux4[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*htL     + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*htR)    + 0.0       )*side[i];

          htL::Float32 =  PL[i]/rhoL[i]/(gammaV-Float32(1.0)) + Float32(0.5)*(_UL[i]*_UL[i] + _VL[i]*_VL[i]) +  PL[i]/rhoL[i]; 
          htR::Float32 =  PR[i]/rhoR[i]/(gammaV-Float32(1.0)) + Float32(0.5)*(_UR[i]*_UR[i] + _VR[i]*_VR[i]) +  PR[i]/rhoR[i];

          aL_til::Float32 = CUDA.sqrt(Float32(2.0)*( gammaV-Float32(1.0))/(gammaV+Float32(1.0))*htL) * CUDA.min(Float32(1.0), (CUDA.sqrt(Float32(2.0)*( gammaV-Float32(1.0))/(gammaV+Float32(1.0))*htL))/CUDA.abs( _UL[i]*nx[i+j] + _VL[i]*ny[i+j] ));
          aR_til::Float32 = CUDA.sqrt(Float32(2.0)*( gammaV-Float32(1.0))/(gammaV+Float32(1.0))*htR) * CUDA.min(Float32(1.0), (CUDA.sqrt(Float32(2.0)*( gammaV-Float32(1.0))/(gammaV+Float32(1.0))*htR))/CUDA.abs( _UR[i]*nx[i+j] + _VR[i]*ny[i+j] ));
          a12::Float32 = CUDA.min(aL_til,aR_til);
             
          m_dot12::Float32 = cuM_p( (_UL[i]*nx[i+j] + _VL[i]*ny[i+j])/a12,Float32(1.0/8.0))         + cuM_m( (_UR[i]*nx[i+j] + _VR[i]*ny[i+j])/a12,Float32(1.0/8.0));
          p12::Float32     = cuP_p( (_UL[i]*nx[i+j] + _VL[i]*ny[i+j])/a12,Float32(3.0/16.0))*PL[i]  + cuP_m( (_UR[i]*nx[i+j] + _VR[i]*ny[i+j])/a12,Float32(3.0/16.0))*PR[i];
     
          flux1[i] = flux1[i]  -( a12*( Float32(0.5)*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]         + Float32(0.5)*(m_dot12-CUDA.abs(m_dot12))*rhoR[i])                      )*side[i+j];
          flux2[i] = flux2[i]  -( a12*( Float32(0.5)*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_UL[i]  + Float32(0.5)*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_UR[i]) + p12*nx[i+j] )*side[i+j];
          flux3[i] = flux3[i]  -( a12*( Float32(0.5)*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_VL[i]  + Float32(0.5)*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_VR[i]) + p12*ny[i+j] )*side[i+j];
          flux4[i] = flux4[i]  -( a12*( Float32(0.5)*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*htL     + Float32(0.5)*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*htR)                  )*side[i+j];


    end    


    return
end




function kernel_AUSM2dd(rhoL, _UL, _VL, PL, rhoR, _UR, _VR, PR, nx, ny, side, flux1,flux2,flux3,flux4,gammaV, numNeib)

    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;

    if (i <= length(rhoL)  && numNeib[i] == 4)

        htL::Float32 =  PL[i]/rhoL[i]/(gammaV[i]-1.0) + 0.5*(_UL[i]*_UL[i] + _VL[i]*_VL[i]) +  PL[i]/rhoL[i]; 
        htR::Float32 =  PR[i]/rhoR[i]/(gammaV[i]-1.0) + 0.5*(_UR[i]*_UR[i] + _VR[i]*_VR[i]) +  PR[i]/rhoR[i];
   
      #  htN::Float32  = 0.5*(htL + htR - 0.5*((_UL*ny[i] - _VL*nx[i])*(_UL*ny[i] - _VL*nx[i]) + (_UR*ny[i] - _VR*nx[i])*(_UR*ny[i] - _VR*nx[i])));

        aL_til::Float32 = CUDA.sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htL) * min(1.0, (sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htL))/CUDA.abs( _UL[i]*nx[i] + _VL[i]*ny[i] ));
        aR_til::Float32 = CUDA.sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htR) * min(1.0, (sqrt(2.0*( gammaV[i]-1.0)/(gammaV[i]+1.0)*htR))/CUDA.abs( _UR[i]*nx[i] + _VR[i]*ny[i] ));
        a12::Float32 = CUDA.min(aL_til,aR_til);
           
        m_dot12::Float32 = cuM_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12,1.0/8.0)         + cuM_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12,1.0/8.0);
        p12::Float32     = cuP_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12,3.0/16.0)*PL[i]  + cuP_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12,3.0/16.0)*PR[i];
   
        flux1[i] = flux1[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]         + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i])        + 0.0       )*side[i];
        flux2[i] = flux2[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_UL[i]  + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_UR[i]) + p12*nx[i] )*side[i];
        flux3[i] = flux3[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*_VL[i]  + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*_VR[i]) + p12*ny[i] )*side[i];
        flux4[i] = flux4[i]  -( a12*( 0.5*(m_dot12+CUDA.abs(m_dot12))*rhoL[i]*htL     + 0.5*(m_dot12-CUDA.abs(m_dot12))*rhoR[i]*htR)    + 0.0       )*side[i];

    
    end    


    return
end


