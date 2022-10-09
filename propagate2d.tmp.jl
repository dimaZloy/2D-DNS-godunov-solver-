
function calcOneStageCUDA(
	betta::Float64, dtX::Float64, flowTime::Float64, 
	testMeshX::mesh2d_Int32, testfields2dX::fields2d, thermoX::THERMOPHYSICS, cellsThreadsX::Array{Int32,2},
	UconsCellsOldX::Array{Float64,2}, UconsDiffTermX::Array{Float64,2}, UconsCellsNewX::Array{Float64,2},
	uLeft::Array{Float64,2},  
	uRight1::Array{Float64,2}, uRight2::Array{Float64,2}, uRight3::Array{Float64,2}, uRight4::Array{Float64,2},
	iFluxV1::Array{Float64,1}, iFluxV2::Array{Float64,1}, iFluxV3::Array{Float64,1}, iFluxV4::Array{Float64,1},
	curLeftV::CuVector{Float32, Mem.DeviceBuffer},  
	cuULeftV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuVLeftV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuPLeftV::CuVector{Float32, Mem.DeviceBuffer}, 
	curRightV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuURightV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuVRightV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuPRightV::CuVector{Float32, Mem.DeviceBuffer}, 
	cuNxV1234::CuVector{Float32, Mem.DeviceBuffer}, 
	cuNyV1234::CuVector{Float32, Mem.DeviceBuffer},
	cuSideV1234::CuVector{Float32, Mem.DeviceBuffer}, 
	cuFluxV1::CuVector{Float32, Mem.DeviceBuffer}, cuFluxV2::CuVector{Float32, Mem.DeviceBuffer}, cuFluxV3::CuVector{Float32, Mem.DeviceBuffer}, cuFluxV4::CuVector{Float32, Mem.DeviceBuffer},
	cuNeibs::CuVector{Int32, Mem.DeviceBuffer})

	#display("Compute cuda stencils ... ")
	 Threads.@threads for p in 1:Threads.nthreads()

	# 	#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);		
		 computeStencilsCUDA(cellsThreadsX[p,1], cellsThreadsX[p,2],  betta, dtX, flowTime,  testMeshX, testfields2dX, thermoX, 
				 uLeft, uRight1, uRight2, uRight3, uRight4);

					
	 end

	#display("done")


	#display("Compute cuda fluxes ... ")

	# TO DO> update to use more threads then 1024 ! 

	num_blocks = cld(testMeshX.nCells,256);

	cuGamma = cudaconvert(Float32(thermoX.Gamma)); 

	CUDA.@sync begin

		fill!(cuFluxV1,0.0);
		fill!(cuFluxV2,0.0);
		fill!(cuFluxV3,0.0);
		fill!(cuFluxV4,0.0);

		copyto!(curLeftV,uLeft[1,:]);
		copyto!(cuULeftV,uLeft[2,:]);
		copyto!(cuVLeftV,uLeft[3,:]);
		copyto!(cuPLeftV,uLeft[4,:]);

		copyto!(curRightV,uRight1[1,:]);
		copyto!(cuURightV,uRight1[2,:]);
		copyto!(cuVRightV,uRight1[3,:]);
		copyto!(cuPRightV,uRight1[4,:]);


	 @cuda blocks = num_blocks threads = 256 kernel_AUSM2d(
			curRightV, cuURightV, cuVRightV, cuPRightV,
			curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
			 cuNxV1234, cuNyV1234, cuSideV1234, cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4, cuGamma, cudaconvert(0));
 
		 copyto!(curRightV,uRight2[1,:]);
		 copyto!(cuURightV,uRight2[2,:]);
		 copyto!(cuVRightV,uRight2[3,:]);
		 copyto!(cuPRightV,uRight2[4,:]);
		 
			
	 @cuda blocks = num_blocks threads = 256 kernel_AUSM2d(
			curRightV, cuURightV, cuVRightV, cuPRightV, 
			curLeftV, cuULeftV, cuVLeftV, cuPLeftV,  
			 cuNxV1234, cuNyV1234, cuSideV1234, cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4,cuGamma, cudaconvert(1) );
 

		 copyto!(curRightV,uRight3[1,:]);
		 copyto!(cuURightV,uRight3[2,:]);
		 copyto!(cuVRightV,uRight3[3,:]);
		 copyto!(cuPRightV,uRight3[4,:]);
 

	 @cuda blocks = num_blocks threads = 256 kernel_AUSM2d(
			curRightV, cuURightV, cuVRightV, cuPRightV, 
			curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
			 cuNxV1234, cuNyV1234, cuSideV1234, cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4, cuGamma, cudaconvert(2) );
			
		#  copyto!(curRightV,uRight4[1,:]);
		 #  copyto!(cuURightV,uRight4[2,:]);
		 #  copyto!(cuVRightV,uRight4[3,:]);
		 #  copyto!(cuPRightV,uRight4[4,:]);

	#   @cuda blocks = 256 threads = num_threads kernel_AUSM2dd(
	#	curRightV, cuURightV, cuVRightV, cuPRightV,
	#	curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
	#  	 cuNxV3, cuNyV3, cuSideV3,cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4,cuGammaV, cuNeibs );


	
		copyto!(iFluxV1,cuFluxV1);
		copyto!(iFluxV2,cuFluxV2);
		copyto!(iFluxV3,cuFluxV3);
		copyto!(iFluxV4,cuFluxV4);

	end
	  
		
	#display("done ... ")
	#display("update solution ... ")

	 Threads.@threads for p in 1:Threads.nthreads()

	# 	#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);	
		
		 for i = cellsThreadsX[p,1]:cellsThreadsX[p,2]
		
			 Rarea::Float32 = 1.0/testMeshX.cell_areas[i];
			  UconsCellsNewX[i,1] = ( UconsCellsOldX[i,1] - iFluxV1[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,1] );
			  UconsCellsNewX[i,2] = ( UconsCellsOldX[i,2] - iFluxV2[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,2] );
			  UconsCellsNewX[i,3] = ( UconsCellsOldX[i,3] - iFluxV3[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,3] );
			  UconsCellsNewX[i,4] = ( UconsCellsOldX[i,4] - iFluxV4[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,4] );
			
		 end
					
	 end
	#display("done")

end
