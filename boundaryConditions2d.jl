


@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64, bnd_cell::Array{Float64,1} )


	if (neib_cell == -1) #inlet

            bnd_cell[1] = 0.6380;
            bnd_cell[2] = 712.145;
            bnd_cell[3] = 0.0;
            bnd_cell[4] = 50000.0;

		
		elseif (neib_cell == -2) #outlet 

			bnd_cell[1] = cur_cell[1];	
			bnd_cell[2] = cur_cell[2];	
			bnd_cell[3] = cur_cell[3];	
			bnd_cell[4] = cur_cell[4];	
			

		elseif (neib_cell == -3) #wall

	 	   	#updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);
			
			bnd_cell[1] = cur_cell[1];
            bnd_cell[2] = 0.0; ##-cur_cell[2];
            bnd_cell[3] = 0.0; ##-cur_cell[3];
            bnd_cell[4] = cur_cell[4];

		elseif (neib_cell == -4) #slip-wall

	 	   	updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);
			
					
		end	
		
		
		## laminar BL ans incidend shock wave interaction
		
		# if (neib_cell == -1) #top 

            # bnd_cell[1] = 1.7;
            # bnd_cell[2] = 263.72;
            # bnd_cell[3] = -51.62;
            # bnd_cell[4] = 15282.0;

		
		# elseif (neib_cell == -2) #right 

			# bnd_cell[1] = cur_cell[1];	
			# bnd_cell[2] = cur_cell[2];	
			# bnd_cell[3] = cur_cell[3];	
			# bnd_cell[4] = cur_cell[4];	
			

		# elseif (neib_cell == -3) #bottom 

	 	   	# #updateVelocityFromCurvWall(i,k,cur_cell,nx,ny,bnd_cell);
			
			 # bnd_cell[1] = cur_cell[1];
             # bnd_cell[2] = 0.0; ##-cur_cell[2];
             # bnd_cell[3] = 0.0; ##-cur_cell[3];
             # bnd_cell[4] = cur_cell[4];

        
		# elseif (neib_cell == -4) # left boundary 

            # bnd_cell[1] = 1.0;
            # bnd_cell[2] = 290.0;
            # bnd_cell[3] = 0.0;
            # bnd_cell[4] = 7143;

					
		# end	

end




@inline function updateVelocityFromCurvWall(i::Int32, k::Int32, U::Array{Float64,1}, nx::Float64, ny::Float64)

# High-Order Accurate Implementation of Solid Wall Boundary Conditions in Curved Geometries, 
# Lilia Krivodonova and Marsha Berger, Courant Institute of Mathematical Sciences, New York, NY 10012

# a = U[1]*(ny*ny - nx*nx) - 2.0*nx*ny*U[2];
# b = U[2]*(nx*nx - ny*ny) - 2.0*nx*ny*U[1];


	Un = deepcopy(U); 

        Un[2] = U[2]*(ny*ny - nx*nx) - 2.0*nx*ny*U[3];
        Un[3] = U[3]*(nx*nx - ny*ny) - 2.0*nx*ny*U[2];


	return Un;	
end


@inline function updateVelocityFromCurvWall(i::Int32, k::Int32, U::Array{Float64,1}, nx::Float64, ny::Float64, Un::Array{Float64,1})

# High-Order Accurate Implementation of Solid Wall Boundary Conditions in Curved Geometries, 
# Lilia Krivodonova and Marsha Berger, Courant Institute of Mathematical Sciences, New York, NY 10012

# a = U[1]*(ny*ny - nx*nx) - 2.0*nx*ny*U[2];
# b = U[2]*(nx*nx - ny*ny) - 2.0*nx*ny*U[1];


	##Un = deepcopy(U); 

	Un[1] = U[1];
    Un[2] = U[2]*(ny*ny - nx*nx) - 2.0*nx*ny*U[3];
    Un[3] = U[3]*(nx*nx - ny*ny) - 2.0*nx*ny*U[2];
	Un[4] = U[4];

	##return Un;	
end
