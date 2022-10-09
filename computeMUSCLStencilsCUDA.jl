

function computeMUSCLStencilsCUDA(
	cellsThreadsX::Array{Int32,2}, 
	testMesh::mesh2d_Int32, testFields::fields2d, thermo::THERMOPHYSICS, 
	uLeft1::Array{Float64,2}, uLeft2::Array{Float64,2}, uLeft3::Array{Float64,2}, uLeft4::Array{Float64,2},
    uRight1::Array{Float64,2}, uRight2::Array{Float64,2}, uRight3::Array{Float64,2}, uRight4::Array{Float64,2}, 
    numNeibs::Vector{Int32}, flowTime::Float64)

	
	Threads.@threads for p in 1:Threads.nthreads()	
	
		for i = cellsThreadsX[p,1]: cellsThreadsX[p,2]
		
			computeInterfaceSlopeCUDA(i, Int32(1), testMesh, testFields, thermo, uLeft1, uRight1, flowTime) ;
			computeInterfaceSlopeCUDA(i, Int32(2), testMesh, testFields, thermo, uLeft2, uRight2, flowTime) ;
			computeInterfaceSlopeCUDA(i, Int32(3), testMesh, testFields, thermo, uLeft3, uRight3, flowTime) ;

			if (numNeibs[i] == 4)
				computeInterfaceSlopeCUDA(i, Int32(4), testMesh, testFields, thermo, uLeft4, uRight4, flowTime) ;
			end
		
	
		end # i - loop for all cells

	end

end
