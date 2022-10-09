


function calcOneStageMUSCL(
	betta::Float64, dtX::Float64, flowTime::Float64, 
	testMesh::mesh2d_Int32, testfields2dX::fields2d, thermo::THERMOPHYSICS, cellsThreadsX::Array{Int32,2},
	UconsCellsOldX::Array{Float64,2}, UconsDiffTermX::Array{Float64,2}, UconsCellsNewX::Array{Float64,2},
	ULeft1::Array{Float64,2}, ULeft2::Array{Float64,2}, ULeft3::Array{Float64,2}, ULeft4::Array{Float64,2},     
	URight1::Array{Float64,2}, URight2::Array{Float64,2}, URight3::Array{Float64,2}, URight4::Array{Float64,2}, FLUXES::Array{Float64,2})



    Threads.@threads for p in 1:Threads.nthreads()
		
         edge_flux1 = zeros(Float64,4);
         edge_flux2 = zeros(Float64,4);
         edge_flux3 = zeros(Float64,4);
         edge_flux4 = zeros(Float64,4);

        for C = cellsThreadsX[p,1]:cellsThreadsX[p,2]

             edge_flux1[1] = 0.0;
		     edge_flux1[2] = 0.0;
             edge_flux1[3] = 0.0;
             edge_flux1[4] = 0.0;
            
             edge_flux2[1] = 0.0;
             edge_flux2[2] = 0.0;
             edge_flux2[3] = 0.0;
             edge_flux2[4] = 0.0;
            
             edge_flux3[1] = 0.0;
             edge_flux3[2] = 0.0;
             edge_flux3[3] = 0.0;
             edge_flux3[4] = 0.0;

             edge_flux4[1] = 0.0;
             edge_flux4[2] = 0.0;
             edge_flux4[3] = 0.0;
             edge_flux4[4] = 0.0;

             FLUXES[C,1] = FLUXES[C,2] = FLUXES[C,3] = FLUXES[C,4] = 0.0;

            AUSMplusFlux2dFast(URight1[1,C],URight1[2,C],URight1[3,C],URight1[4,C],ULeft1[1,C],ULeft1[2,C],ULeft1[3,C],ULeft1[4,C],
             testMesh.cell_edges_Nx[C,1],testMesh.cell_edges_Ny[C,1], testMesh.cell_edges_length[C,1],thermo.Gamma, edge_flux1);

            AUSMplusFlux2dFast(URight2[1,C],URight2[2,C],URight2[3,C],URight2[4,C],ULeft2[1,C],ULeft2[2,C],ULeft2[3,C],ULeft2[4,C], 
             testMesh.cell_edges_Nx[C,2],testMesh.cell_edges_Ny[C,2], testMesh.cell_edges_length[C,2],thermo.Gamma, edge_flux2);

            AUSMplusFlux2dFast(URight3[1,C],URight3[2,C],URight3[3,C],URight3[4,C],ULeft3[1,C],ULeft3[2,C],ULeft3[3,C],ULeft3[4,C], 
                testMesh.cell_edges_Nx[C,3],testMesh.cell_edges_Ny[C,3],testMesh.cell_edges_length[C,3],thermo.Gamma, edge_flux3);

            
             FLUXES[C,1] = edge_flux1[1] + edge_flux2[1] + edge_flux3[1];
			 FLUXES[C,2] = edge_flux1[2] + edge_flux2[2] + edge_flux3[2];
			 FLUXES[C,3] = edge_flux1[3] + edge_flux2[3] + edge_flux3[3];
			 FLUXES[C,4] = edge_flux1[4] + edge_flux2[4] + edge_flux3[4];

            # FLUXES[C,1] = edge_flux1[1,C] + edge_flux2[1,C] + edge_flux3[1,C];
			# FLUXES[C,2] = edge_flux1[2,C] + edge_flux2[2,C] + edge_flux3[2,C];
			# FLUXES[C,3] = edge_flux1[3,C] + edge_flux2[3,C] + edge_flux3[3,C];
			# FLUXES[C,4] = edge_flux1[4,C] + edge_flux2[4,C] + edge_flux3[4,C];


            if (testMesh.mesh_connectivity[C,3] == 4)
                AUSMplusFlux2dFast(URight4[1,C],URight4[2,C],URight4[3,C],URight4[4,C],ULeft4[1,C],ULeft4[2,C],ULeft4[3,C],ULeft4[4,C], 
                    testMesh.cell_edges_Nx[C,4],testMesh.cell_edges_Ny[C,4], testMesh.cell_edges_length[C,4],thermo.Gamma, edge_flux4);

                     FLUXES[C,1] = edge_flux1[1] + edge_flux2[1] + edge_flux3[1] + edge_flux4[1];
                     FLUXES[C,2] = edge_flux1[2] + edge_flux2[2] + edge_flux3[2] + edge_flux4[2];
                     FLUXES[C,3] = edge_flux1[3] + edge_flux2[3] + edge_flux3[3] + edge_flux4[3];
                     FLUXES[C,4] = edge_flux1[4] + edge_flux2[4] + edge_flux3[4] + edge_flux4[4];

                    # FLUXES[C,1] = edge_flux1[1,C] + edge_flux2[1,C] + edge_flux3[1,C] + edge_flux4[1,C];
                    # FLUXES[C,2] = edge_flux1[2,C] + edge_flux2[2,C] + edge_flux3[2,C] + edge_flux4[2,C];
                    # FLUXES[C,3] = edge_flux1[3,C] + edge_flux2[3,C] + edge_flux3[3,C] + edge_flux4[3,C];
                    # FLUXES[C,4] = edge_flux1[4,C] + edge_flux2[4,C] + edge_flux3[4,C] + edge_flux4[4,C];
                
            end


        end

    end    

	  
		
	#display("done ... ")
	#display("update solution ... ")

	 Threads.@threads for p in 1:Threads.nthreads()
		
		 for i = cellsThreadsX[p,1]:cellsThreadsX[p,2]
		
			 Rarea::Float64 = 1.0/testMesh.cell_areas[i];
			  UconsCellsNewX[i,1] = ( UconsCellsOldX[i,1] - FLUXES[i,1]*betta*dtX*Rarea ); #+ betta*dtX*UconsDiffTermX[i,1] );
			  UconsCellsNewX[i,2] = ( UconsCellsOldX[i,2] - FLUXES[i,2]*betta*dtX*Rarea ); #+ betta*dtX*UconsDiffTermX[i,2] );
			  UconsCellsNewX[i,3] = ( UconsCellsOldX[i,3] - FLUXES[i,3]*betta*dtX*Rarea ); #+ betta*dtX*UconsDiffTermX[i,3] );
			  UconsCellsNewX[i,4] = ( UconsCellsOldX[i,4] - FLUXES[i,4]*betta*dtX*Rarea ); #+ betta*dtX*UconsDiffTermX[i,4] );
			
		 end
					
	 end
	#display("done")

end


## DO NOT USE  CuVector{Float64, Mem.UnifiedBuffer} 
## its two times slowly than CuVector{Float64, Mem.DeviceBuffer} 

#= function calcOneStageCUDA(
		betta::Float64, dtX::Float64, flowTime::Float64, 
		testMeshX::mesh2d_Int32, testfields2dX::fields2d, thermoX::THERMOPHYSICS, cellsThreadsX::Array{Int32,2},
		UconsCellsOldX::Array{Float64,2}, UconsDiffTermX::Array{Float64,2}, UconsCellsNewX::Array{Float64,2},
		uLeft::Array{Float64,2},  
		uRight1::Array{Float64,2}, uRight2::Array{Float64,2}, uRight3::Array{Float64,2}, uRight4::Array{Float64,2},
		iFluxV1::Array{Float64,1}, iFluxV2::Array{Float64,1}, iFluxV3::Array{Float64,1}, iFluxV4::Array{Float64,1},
		curLeftV::CuVector{Float64, Mem.UnifiedBuffer},  
		cuULeftV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuVLeftV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuPLeftV::CuVector{Float64, Mem.UnifiedBuffer}, 
		curRightV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuURightV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuVRightV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuPRightV::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNxV1::CuVector{Float64, Mem.UnifiedBuffer},  
		cuNxV2::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNxV3::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNxV4::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNyV1::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNyV2::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNyV3::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuNyV4::CuVector{Float64, Mem.UnifiedBuffer}, 
		cuSideV1::CuVector{Float64, Mem.UnifiedBuffer},
		cuSideV2::CuVector{Float64, Mem.UnifiedBuffer},
		cuSideV3::CuVector{Float64, Mem.UnifiedBuffer},
		cuSideV4::CuVector{Float64, Mem.UnifiedBuffer},
		cuFluxV1::CuVector{Float64, Mem.UnifiedBuffer},
		cuFluxV2::CuVector{Float64, Mem.UnifiedBuffer},
		cuFluxV3::CuVector{Float64, Mem.UnifiedBuffer},
		cuFluxV4::CuVector{Float64, Mem.UnifiedBuffer},
		cuGammaV::CuVector{Float64, Mem.UnifiedBuffer},
		cuNeibs::CuVector{Int32, Mem.UnifiedBuffer})

		#display("Compute cuda stencils ... ")
		 Threads.@threads for p in 1:Threads.nthreads()
	
		# 	#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);		
		 	computeStencilsCUDA(cellsThreadsX[p,1], cellsThreadsX[p,2],  betta, dtX, flowTime,  testMeshX, testfields2dX, thermoX, 
		 			uLeft, uRight1, uRight2, uRight3, uRight4);

						
		 end

		#display("done")


		#display("Compute cuda fluxes ... ")

		# TO DO> update to use more threads then 1024 ! 

		num_blocks = cld(testMeshX.nCells,512);



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


		 @cuda blocks = num_blocks threads = 512 kernel_AUSM2d(
				curRightV, cuURightV, cuVRightV, cuPRightV,
				curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
		 		cuNxV1, cuNyV1, cuSideV1, cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4, cuGammaV);
	 
			 copyto!(curRightV,uRight2[1,:]);
			 copyto!(cuURightV,uRight2[2,:]);
			 copyto!(cuVRightV,uRight2[3,:]);
			 copyto!(cuPRightV,uRight2[4,:]);
	 		
				
		 @cuda blocks = num_blocks threads = 512 kernel_AUSM2d(
				curRightV, cuURightV, cuVRightV, cuPRightV, 
				curLeftV, cuULeftV, cuVLeftV, cuPLeftV,  
		 		cuNxV2, cuNyV2, cuSideV2, cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4,cuGammaV );
	 
	
			 copyto!(curRightV,uRight3[1,:]);
			 copyto!(cuURightV,uRight3[2,:]);
			 copyto!(cuVRightV,uRight3[3,:]);
			 copyto!(cuPRightV,uRight3[4,:]);
	 

		 @cuda blocks = num_blocks threads = 512 kernel_AUSM2d(
				curRightV, cuURightV, cuVRightV, cuPRightV, 
				curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
		 		cuNxV3, cuNyV3, cuSideV3,cuFluxV1, cuFluxV2, cuFluxV3,cuFluxV4,cuGammaV );
				
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
			
		 		Rarea::Float64 = 1.0/testMeshX.cell_areas[i];
  		 		 UconsCellsNewX[i,1] = ( UconsCellsOldX[i,1] - iFluxV1[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,1] );
		 		 UconsCellsNewX[i,2] = ( UconsCellsOldX[i,2] - iFluxV2[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,2] );
		 		 UconsCellsNewX[i,3] = ( UconsCellsOldX[i,3] - iFluxV3[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,3] );
		 		 UconsCellsNewX[i,4] = ( UconsCellsOldX[i,4] - iFluxV4[i]*betta*dtX*Rarea + betta*dtX*UconsDiffTermX[i,4] );
				
		 	end
						
		 end
		#display("done")

end
 =#



# @everywhere function doExplicitRK3TVD(
		# betta::Float64, dtX::Float64,
		# testMeshDistrX::mesh2d_shared, testfields2dX::fields2d_shared, thermoX::THERMOPHYSICS, cellsThreadsX::SharedArray{Int64,2},
		# UconsCellsOldX::SharedArray{Float64,2}, iFLUXX::SharedArray{Float64,2}, UconsDiffTerm::SharedArray{Float64,2},  
		# UconsCellsNew1X::SharedArray{Float64,2}, UconsCellsNew2X::SharedArray{Float64,2}, UconsCellsNew3X::SharedArray{Float64,2}, UconsCellsNewX::SharedArray{Float64,2})


			
			# ## based on
			# ## JOURNAL OF COMPUTATIONAL PHYSICS 83, 32-78 (1989)
			# ## Efficient Implementation   of Essentially Non-oscillatory   Shock-Capturing   Schemes, II
			# ## CHI- WANG SHU  AND   STANLEY OSHERS

	
			# # @sync @distributed for p in workers()	
	
				# # beginCell::Int64 = cellsThreadsX[p-1,1];
				# # endCell::Int64 = cellsThreadsX[p-1,2];
				# # #println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
										 
				# # SecondOrderUpwindM2(beginCell ,endCell, 1.0, dtX, testMeshDistrX, testfields2dX, thermoX, UconsCellsOldX, iFLUXX, UconsCellsNew1X);
					
			# # end
			# # @everywhere finalize(SecondOrderUpwindM2);				
			
			# calcOneStage(1.0, dtX, testMeshDistrX, testfields2dX , thermoX , cellsThreadsX, UconsCellsOldX, iFLUXX, UconsDiffTerm,  UconsCellsNew1X); 	
		

			# @sync @distributed for p in workers()	
				# a1::Float64  = 3.0/4.0;
				# a2::Float64  = 1.0/4.0;
				# Gamma::Float64 = thermoX.Gamma;
				
				# beginCell::Int64 = cellsThreadsX[p-1,1];
				# endCell::Int64 = cellsThreadsX[p-1,2];
				
				
				
				# for i = beginCell:endCell
					# UconsCellsNew2X[i,1] = UconsCellsOldX[i,1].*a1 .+ UconsCellsNew1X[i,1].*a2;
					# UconsCellsNew2X[i,2] = UconsCellsOldX[i,2].*a1 .+ UconsCellsNew1X[i,2].*a2;
					# UconsCellsNew2X[i,3] = UconsCellsOldX[i,3].*a1 .+ UconsCellsNew1X[i,3].*a2;
					# UconsCellsNew2X[i,4] = UconsCellsOldX[i,4].*a1 .+ UconsCellsNew1X[i,4].*a2;

					# testfields2dX.densityCells[i] = UconsCellsNew2X[i,1];
					# testfields2dX.UxCells[i] 	  = UconsCellsNew2X[i,2]/UconsCellsNew2X[i,1];		
					# testfields2dX.UyCells[i] 	  = UconsCellsNew2X[i,3]/UconsCellsNew2X[i,1];
					# testfields2dX.pressureCells[i] = (Gamma-1.0)*( UconsCellsNew2X[i,4] - 0.5*( UconsCellsNew2X[i,2]*UconsCellsNew2X[i,2] + UconsCellsNew2X[i,3]*UconsCellsNew2X[i,3] )/UconsCellsNew2X[i,1] );

					# testfields2dX.aSoundCells[i] = sqrt( Gamma * testfields2dX.pressureCells[i]/testfields2dX.densityCells[i] );
					# testfields2dX.VMAXCells[i]  = sqrt( testfields2dX.UxCells[i]*testfields2dX.UxCells[i] + testfields2dX.UyCells[i]*testfields2dX.UyCells[i] ) + testfields2dX.aSoundCells[i];
					
				# end
				
			
			# end		
			
		
			
			# # @sync @distributed for p in workers()	
	
				# # beginCell::Int64 = cellsThreadsX[p-1,1];
				# # endCell::Int64 = cellsThreadsX[p-1,2];
				# # #println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
										 
				# # SecondOrderUpwindM2(beginCell ,endCell, 1.0/4.0, dtX, testMeshDistrX, testfields2dX, thermoX, UconsCellsNew2X, iFLUXX, UconsCellsNew3X);
					
			# # end
			# # @everywhere finalize(SecondOrderUpwindM2);			


			# calcOneStage(1.0/4.0, dtX, testMeshDistrX, testfields2dX , thermoX , cellsThreadsX, UconsCellsNew2X, iFLUXX, UconsDiffTerm, UconsCellsNew3X); 				
			
			
			# @sync @distributed for p in workers()	
				# b1::Float64  = 1.0/3.0;
				# b2::Float64  = 2.0/3.0;
				# Gamma::Float64 = thermoX.Gamma;
				
				# beginCell::Int64 = cellsThreadsX[p-1,1];
				# endCell::Int64 = cellsThreadsX[p-1,2];
				
				# for i = beginCell:endCell
				
					# UconsCellsNew2X[i,1] = UconsCellsOldX[i,1].*b1 .+ UconsCellsNew3X[i,1].*b2;
					# UconsCellsNew2X[i,2] = UconsCellsOldX[i,2].*b1 .+ UconsCellsNew3X[i,2].*b2;
					# UconsCellsNew2X[i,3] = UconsCellsOldX[i,3].*b1 .+ UconsCellsNew3X[i,3].*b2;
					# UconsCellsNew2X[i,4] = UconsCellsOldX[i,4].*b1 .+ UconsCellsNew3X[i,4].*b2;

					# testfields2dX.densityCells[i] = UconsCellsNew2X[i,1];
					# testfields2dX.UxCells[i] 	  = UconsCellsNew2X[i,2]/UconsCellsNew2X[i,1];		
					# testfields2dX.UyCells[i] 	  = UconsCellsNew2X[i,3]/UconsCellsNew2X[i,1];
					# testfields2dX.pressureCells[i] = (Gamma-1.0)*( UconsCellsNew2X[i,4] - 0.5*( UconsCellsNew2X[i,2]*UconsCellsNew2X[i,2] + UconsCellsNew2X[i,3]*UconsCellsNew2X[i,3] )/UconsCellsNew2X[i,1] );

					# testfields2dX.aSoundCells[i] = sqrt( Gamma * testfields2dX.pressureCells[i]/testfields2dX.densityCells[i] );
					# testfields2dX.VMAXCells[i]  = sqrt( testfields2dX.UxCells[i]*testfields2dX.UxCells[i] + testfields2dX.UyCells[i]*testfields2dX.UyCells[i] ) + testfields2dX.aSoundCells[i];
					
				# end
			
				
			
			# end		
			

			# # @sync @distributed for p in workers()	
	
				# # beginCell::Int64 = cellsThreadsX[p-1,1];
				# # endCell::Int64 = cellsThreadsX[p-1,2];
				# # #println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
										 
				# # SecondOrderUpwindM2(beginCell ,endCell, 2.0/3.0, dtX, testMeshDistrX, testfields2dX, thermoX, UconsCellsNew2X, iFLUXX, UconsCellsNewX);
					
			# # end
			# # @everywhere finalize(SecondOrderUpwindM2);				
			
			
			# calcOneStage(2.0/3.0, dtX, testMeshDistrX, testfields2dX , thermoX , cellsThreadsX, UconsCellsNew2X, iFLUXX, UconsDiffTerm,  UconsCellsNewX); 	
			

# end