
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
