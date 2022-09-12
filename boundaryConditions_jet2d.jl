
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


@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64, bnd_cell::Array{Float64,1} )

		##bnd_cell = zeros(Float64,4);

		if (neib_cell == -2) #inlet free-streem


             #bnd_cell[1] = 1.1766766855256956;
             #bnd_cell[2] = 19.964645104094163;
             #bnd_cell[3] = 0.0;
             #bnd_cell[4] = 101325.0;


			 # subsonic inflow boundary condition

			 Minf::Float64 = 0.0575;
			 Tinf = 300.0;
			 Pinf = 101325.0;

			 Pt::Float64 = Pinf/(1.0 + 0.5*(gamma-1.0)*Minf*Minf)^(-gamma/(gamma-1.0));
			 Tt::Float64 = Tinf/(1.0 + 0.5*(gamma-1.0)*Minf*Minf)^(-1.0);

			 Ui::Float64 = sqrt(cur_cell[2]*cur_cell[2] + cur_cell[3]*cur_cell[3]);
			 Ht::Float64 = cur_cell[4]/cur_cell[1]*gamma/(gamma-1.0) + 0.5*Ui*Ui; 
			 ci::Float64 = sqrt(gamma*cur_cell[4]/cur_cell[1]); 
			 Rp::Float64 = -Ui - 2.0*ci/(gamma-1.0); 	
			 a::Float64 = 1.0 + 2.0/(gamma-1.0);
			 b::Float64 = 2.0*Rp;
			 c::Float64 = 0.5*(gamma-1.0)*(Rp*Rp - 2*Ht);
			 cbp::Float64 =  0.5/a * (-b + sqrt(b*b-4.0*a*c) );
			 cbm::Float64 =  0.5/a * (-b - sqrt(b*b-4.0*a*c) );

			 cb::Float64 = max(cbp, cbm);
			 U::Float64 = 2*cb/(gamma-1.0) + Rp;
			 Mb::Float64 = U/cb;
			 Pb::Float64 = Pt/(1.0 + 0.5*(gamma-1.0)*Mb*Mb)^(-gamma/(gamma-1.0));
			 Tb::Float64 = Tt/(1.0 + 0.5*(gamma-1.0)*Mb*Mb)^(-1.0);
			 R::Float64 = 8.31432e+3/28.966;

			 bnd_cell[1] = Pb/R/Tb;
             bnd_cell[2] = U;
             bnd_cell[3] = 0.0;
             bnd_cell[4] = Pb;


			

		elseif (neib_cell == -1) #inlet jet

			#updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);

			bnd_cell[1] = 3.3287824234189656;
            bnd_cell[2] = 583.4956808907839;
            bnd_cell[3] = 0.0;
            bnd_cell[4] = 793578.544909857;

		elseif (neib_cell == -3) # outlet 

			bnd_cell[1] = cur_cell[1];	
			bnd_cell[2] = cur_cell[2];	
			bnd_cell[3] = cur_cell[3];	
			bnd_cell[4] = cur_cell[4];	
			

			# pressure outflow boundary condition 
			#  Mi::Float64 = sqrt(cur_cell[2]*cur_cell[2] + cur_cell[3]*cur_cell[3])/sqrt(gamma*cur_cell[4]/cur_cell[1]);
			#  pb::Float64 = 0.0;
			#  Mi <1.0 ? pb = 101325.0 : pb =  cur_cell[4];
			#  bnd_cell[1] = pb/cur_cell[4]*cur_cell[1];
            #  bnd_cell[2] = cur_cell[2];
            #  bnd_cell[3] = cur_cell[3];
            #  bnd_cell[4] = pb;
		

		elseif (neib_cell == -4) # walls

			updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);
					
		end	
					


	##return bnd_cell; 
end

