
# @inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
#       cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64 )::Array{Float64,1}

# 		bnd_cell = zeros(Float64,4);

# 		if (neib_cell == -1) #inlet

#             bnd_cell[1] = 1.1766;
#             bnd_cell[2] = 1232.6445;
#             bnd_cell[3] = 0.0;
#             bnd_cell[4] = 101325.0;

# 		elseif (neib_cell == -3) #walls


# 			bnd_cell = updateVelocityFromCurvWall(i,k,cur_cell,nx,ny);

# 		elseif (neib_cell == -2) # outlet

# 			bnd_cell = cur_cell;	
					
# 		end	

# 	return bnd_cell; 
# end


@inline function turbulentInlet(patchValueOld::Float64, refValue::Float64,  fluctScale::Float64)::Float64

	alpha::Float64 = 0.1;
	cRMS::Float64 = sqrt(12.0*(2.0*alpha-alpha*alpha))/alpha;
	Z1::Float64 = cRMS*rand(Float64)*0.5*fluctScale*abs(refValue);
	Z2::Float64 = cRMS*rand(Float64)*-0.5*fluctScale*abs(refValue);
	return (1.0-alpha)*patchValueOld + alpha*(refValue + 0.5*(Z1+Z2));

end


@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64, bnd_cell::Array{Float64,1} )

	
	  if (neib_cell == -1) #inlet

	  delta::Float64 = 1.44e-4;
			
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
	  T::Float64 = lambda/Uc;

	  yhat::Float64 = y - 60.0*delta*0.5;
	  
	  ##scale::Float64 = 1.0;
	  ##scale::Float64 = 0.01*Uc;

	  inletUx::Float64 = tanh(2.0*y/delta)*(U1-U2)*0.5 + (U1 + U2)*0.5; 
	  inletUy::Float64 = 500.0*(a1*cos(2*pi*1.0*t/T + phi1)*exp(-yhat*yhat/b) + a2*cos(2*pi*2.0*t/T + phi2)*exp(-yhat*yhat/b) ); 
	  
	  bnd_cell[1] = tanh(2.0*y/delta)*(rho1 - rho2)*0.5 + (rho1 + rho2)*0.5;
	  #bnd_cell[2] = turbulentInlet(cur_cell[2], inletUx,0.05);
	  bnd_cell[2] = inletUx;
	  bnd_cell[3] = turbulentInlet(cur_cell[3], inletUy,0.025);  			
	  bnd_cell[4] = tanh(2.0*y/delta)*(P1-P2)*0.5 + (P1+P2)*0.5;			 



  elseif (neib_cell == -4) #top wall

	  
	#  updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);
	  
	    bnd_cell[1] = 0.2861057;
	    bnd_cell[2] = 1526.3;
	    bnd_cell[3] = 165.7;
	    bnd_cell[4] = 129951.0;

  elseif (neib_cell == -3) # outlet

	  ##bnd_cell = cur_cell;	
	  bnd_cell[1] =cur_cell[1];
	  bnd_cell[2] =cur_cell[2];
	  bnd_cell[3] =cur_cell[3];
	  bnd_cell[4] =cur_cell[4];
  
  elseif (neib_cell == -2) #bottom
  

	#  updateVelocityFromCurvWall(i,k,cur_cell,nx,ny, bnd_cell);

	   bnd_cell[1] = cur_cell[1];
	   bnd_cell[2] = cur_cell[2];
	   bnd_cell[3] =-cur_cell[3];
	   bnd_cell[4] = cur_cell[4];
  

			  
  end	

	##return bnd_cell; 
end

