function kernel_testRoe(rhoLeft, ULeft, VLeft, PLeft, rhoRight, URight, VRight, PRight, nx, ny, side, flux1,flux2,flux3,flux4,gammaV)

		#primL::Array{Float64,1},primR::Array{Float64,1},
		#nx::Float64,ny::Float64,side::Float64, gamma::Float64)::Array{Float64,1}

# [1] P. L. Roe, Approximate Riemann Solvers, Parameter Vectors and
# Difference Schemes, Journal of Computational Physics, 43, pp. 357-372.
#
# [2] H. Nishikawa and K. Kitamura, Very Simple, Carbuncle-Free,
# Boundary-Layer Resolving, Rotated-Hybrid Riemann Solvers,
# Journal of Computational Physics, 227, pp. 2560-2581, 2008.

# -------------------------------------------------------------------------
#  Input:   primL(1:4) =  left state (rhoL, uL, vL, pL)
#           primR(1:4) = right state (rhoR, uR, vR, pR)
#               njk(2) = Face normal (L -> R). Must be a unit vector.
#
# Output:    flux(1:4) = numerical flux
#                  wsn = half the max wave speed (to be used for time step calculations)
# -------------------------------------------------------------------------



#%Tangent vector (Do you like it? Actually, Roe flux can be implemented without any tangent vector. See "I do like CFD, VOL.1" for details.)

i = (blockIdx().x - 1) * blockDim().x + threadIdx().x;

if i <= length(rhoLeft) 

  #gamma = gammaV[i]; 

  mx::Float64 = -ny[i];
  my::Float64 =  nx[i];


#  Left state
    rhoL::Float64 = rhoLeft[i];
      uL::Float64 = ULeft[i];
      vL::Float64 = VLeft[i];
      pL::Float64 = PLeft[i];

     unL::Float64 = uL*nx[i]+vL*ny[i];
     umL::Float64 = uL*mx+vL*my;
      aL::Float64 = CUDA.sqrt(gammaV[i]*pL/rhoL);
      HL::Float64 = aL*aL/(gammaV[i]-1.0) + 0.5*(uL*uL+vL*vL);

#  Right state
    rhoR::Float64 = rhoRight[i];
      uR::Float64 = URight[i];
      vR::Float64 = VRight[i];
      pR::Float64 = PRight[i];

     unR::Float64 = uR*nx[i]+vR*ny[i];
     umR::Float64 = uR*mx+vR*my;
      
      aR::Float64 = CUDA.sqrt(gammaV[i]*pR/rhoR);
      HR::Float64 = aR*aR/(gammaV[i]-1.0) + 0.5*(uR*uR+vR*vR);

# First compute the Roe Averages
    RT::Float64 = CUDA.sqrt(rhoR/rhoL);
   rho::Float64 = RT*rhoL;
     u::Float64 = (uL+RT*uR)/(1.0+RT);
     v::Float64 = (vL+RT*vR)/(1.0+RT);
     H::Float64 = (HL+RT* HR)/(1.0+RT);
     a::Float64 = CUDA.sqrt( (gammaV[i]-1.0)*(H-0.5*(u*u+v*v)) );
    un::Float64 = u*nx[i]+v*ny[i];
    um::Float64 = u*mx+v*my;

# Wave Strengths
   drho::Float64 = rhoR - rhoL; 
     dp::Float64 =   pR - pL;
    dun::Float64 =  unR - unL;
    dum::Float64 =  umR - umL;

  #LdU = zeros(Float64,4);
		
  LdU1::Float64 = (dp - rho*a*dun )/(2.0*a*a);
  LdU2::Float64 = rho*dum;
  LdU3::Float64 =  drho - dp/(a*a);
  LdU4::Float64 = (dp + rho*a*dun )/(2.0*a*a);

# Wave Speed

  #ws = zeros(Float64,4);
  
  ws1::Float64 = CUDA.abs(un-a);
  ws2::Float64 = CUDA.abs(un);
  ws3::Float64 = CUDA.abs(un);
  ws4::Float64 = CUDA.abs(un+a);

# Harten's Entropy Fix JCP(1983), 49, pp357-393: 
# only for the nonlinear fields.
  #dws = zeros(Float64,4);
  dws1 = 1.0/5.0; 
  dws2 = 0.0;
  dws3 = 0.0;
  
  if ws1 < dws1
		ws1 = 0.5*( ws1*ws1/dws1 + dws1 );
  end
  
  dws4=1.0/5.0; 
  if ws4 < dws4
		ws4 = 0.5*( ws4*ws4/dws4 + dws4 ); 
  end

#Right Eigenvectors
#Rv = zeros(Float64,4,4);	
#   Rv[1,1] = 1.0;
#   Rv[2,1] = u - a*nx[i];
#   Rv[3,1] = v - a*ny[i];
#   Rv[4,1] = H - un*a;

#   Rv[1,2] = 0.0;
#   Rv[2,2] = mx;
#   Rv[3,2] = my;
#   Rv[4,2] = um;

#   Rv[1,3] = 1.0;
#   Rv[2,3] = u;
#   Rv[3,3] = v;
#   Rv[4,3] = 0.5*(u*u + v*v);

#   Rv[1,4] = 1.0;
#   Rv[2,4] = u + a*nx[i];
#   Rv[3,4] = v + a*ny[i];
#   Rv[4,4] = H + un*a;


  Rv11::Float64 = 1.0;
  Rv21::Float64 = u - a*nx[i];
  Rv31::Float64 = v - a*ny[i];
  Rv41::Float64 = H - un*a;

  Rv12::Float64 = 0.0;
  Rv22::Float64 = mx;
  Rv32::Float64 = my;
  Rv42::Float64 = um;

  Rv13::Float64 = 1.0;
  Rv23::Float64 = u;
  Rv33::Float64 = v;
  Rv43::Float64 = 0.5*(u*u + v*v);

  Rv14::Float64 = 1.0;
  Rv24::Float64 = u + a*nx[i];
  Rv34::Float64 = v + a*ny[i];
  Rv44::Float64 = H + un*a;

#Dissipation Term
  diss1::Float64 = 0.0;
  diss2::Float64 = 0.0;
  diss3::Float64 = 0.0;
  diss4::Float64 = 0.0;

  #for i=1:4;
	#for j=1:4;
		diss1 = diss1 + ws1*LdU1*Rv11;
    diss1 = diss1 + ws2*LdU2*Rv12;
    diss1 = diss1 + ws3*LdU3*Rv13;
    diss1 = diss1 + ws4*LdU4*Rv14;    

    diss2 = diss2 + ws1*LdU1*Rv21;
    diss2 = diss2 + ws2*LdU2*Rv22;
    diss2 = diss2 + ws3*LdU3*Rv23;
    diss2 = diss2 + ws4*LdU4*Rv24;    

    diss3 = diss3 + ws1*LdU1*Rv31;
    diss3 = diss3 + ws2*LdU2*Rv32;
    diss3 = diss3 + ws3*LdU3*Rv33;
    diss3 = diss3 + ws4*LdU4*Rv34;    

    diss4 = diss4 + ws1*LdU1*Rv41;
    diss4 = diss4 + ws2*LdU2*Rv42;
    diss4 = diss4 + ws3*LdU3*Rv43;
    diss4 = diss4 + ws4*LdU4*Rv44;    
  #  end 
  #end

  
#Compute the flux.
  #fL = zeros(Float64,4)
  fL1::Float64 = rhoL*unL;
  fL2::Float64 = rhoL*unL * uL + pL*nx[i];
  fL3::Float64 = rhoL*unL * vL + pL*ny[i];
  fL4::Float64 = rhoL*unL * HL;

  #fR = zeros(Float64,4)
  fR1::Float64 = rhoR*unR;
  fR2::Float64 = rhoR*unR * uR + pR*nx[i];
  fR3::Float64 = rhoR*unR * vR + pR*ny[i];
  fR4::Float64 = rhoR*unR * HR;

  # flux[1] = 0.5 * (fL[1] + fR[1] - diss[1]);
  # flux[2] = 0.5 * (fL[2] + fR[2] - diss[2]);
  # flux[3] = 0.5 * (fL[3] + fR[3] - diss[3]);
  # flux[4] = 0.5 * (fL[4] + fR[4] - diss[4]);
  #wsn = 0.5*(abs(un) + a);  #Normal max wave speed times half
  
  flux1[i] = flux1[i] -0.5 * (fL1 + fR1 - diss1) * side[i];
  flux2[i] = flux2[i] -0.5 * (fL2 + fR2 - diss2) * side[i];
  flux3[i] = flux3[i] -0.5 * (fL3 + fR3 - diss3) * side[i];
  flux4[i] = flux4[i] -0.5 * (fL4 + fR4 - diss4) * side[i];
  
  end

  return
  
end