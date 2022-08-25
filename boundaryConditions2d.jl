
@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64 )::Array{Float64,1}

		bnd_cell = zeros(Float64,4);

		if (neib_cell == -1) #inlet

            bnd_cell[1] = 1.1766;
            bnd_cell[2] = 1232.6445;
            bnd_cell[3] = 0.0;
            bnd_cell[4] = 101325.0;

		elseif (neib_cell == -3) #walls


			bnd_cell = updateVelocityFromCurvWall(i,k,cur_cell,nx,ny);

		elseif (neib_cell == -2) # outlet

			bnd_cell = cur_cell;	
					
		end	

	return bnd_cell; 
end


@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64, bnd_cell::Array{Float64,1} )

		##bnd_cell = zeros(Float64,4);

		if (neib_cell == -1) #inlet

            bnd_cell[1] = 1.1766;
            bnd_cell[2] = 1232.6445;
            bnd_cell[3] = 0.0;
            bnd_cell[4] = 101325.0;

		elseif (neib_cell == -3) #walls


			##bnd_cell = updateVelocityFromCurvWall(i,k,cur_cell,nx,ny);
			updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);

		elseif (neib_cell == -2) # outlet

			bnd_cell[1] = cur_cell[1];	
			bnd_cell[2] = cur_cell[2];	
			bnd_cell[3] = cur_cell[3];	
			bnd_cell[4] = cur_cell[4];	
					
		end	

	##return bnd_cell; 
end

