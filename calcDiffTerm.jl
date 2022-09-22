

function calcDiffTerm(cellsThreadsX::Array{Int32,2}, nodesThreadsX::Array{Int32,2}, testMeshDistrX::mesh2d_Int32, 
		testfields2dX::fields2d, viscousFields2dX::viscousFields2d, thermoX::THERMOPHYSICS, 
		UCNodes::Array{Float64,2}, UConsDiffCells::Array{Float64,2})


		Threads.@threads for p in 1:Threads.nthreads()
		
	
			#beginCell::Int32 = cellsThreadsX[p,1];
			#endCell::Int32 = cellsThreadsX[p,2];
			
			# calculate gradients in cells based on UCNodes:
			# viscousFields2dX.cdUdxCells[C] 
		  	# viscousFields2dX.cdUdyCells[C] 
		 	# viscousFields2dX.cdVdxCells[C] 
		  	# viscousFields2dX.cdVdyCells[C] 
			# viscousFields2dX.cdEdxCells[C] 
		  	# viscousFields2dX.cdEdyCells[C] 
			nodesGradientReconstructionUconsFastSA(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX, UCNodes, viscousFields2dX);	
			
		end
		
		## then we need to reconstruct gradients to nodes: 
		Threads.@threads for p in 1:Threads.nthreads()
		
	
			#beginCell::Int32 = nodesThreadsX[p,1];
			#endCell::Int32 = nodesThreadsX[p,2];
			# compute :
			# viscousFields2dX.cdUdxNodes[C] 
		  	# viscousFields2dX.cdUdyNodes[C] 
		 	# viscousFields2dX.cdVdxNodes[C] 
		  	# viscousFields2dX.cdVdyNodes[C] 
			# viscousFields2dX.cdEdxNodes[C] 
		  	# viscousFields2dX.cdEdyNodes[C] 
			
			cells2nodesSolutionReconstructionWithStencilsViscousGradients(nodesThreadsX[p,1],nodesThreadsX[p,2], testMeshDistrX, viscousFields2dX);	
			
		end
		

		
		Threads.@threads for p in 1:Threads.nthreads()
	
	
			#beginCell::Int32 = cellsThreadsX[p,1];
			#endCell::Int32 = cellsThreadsX[p,2];		
			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdUdxCells,viscousFields2dX.cdUdyCells, viscousFields2dX.laplasUCuCells);
			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdVdxCells,viscousFields2dX.cdVdyCells, viscousFields2dX.laplasUCvCells);
			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdEdxCells,viscousFields2dX.cdEdyCells, viscousFields2dX.laplasUCeCells);

			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdUdxNodes,viscousFields2dX.cdUdyNodes, viscousFields2dX.laplasUCuCells);
			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdVdxNodes,viscousFields2dX.cdVdyNodes, viscousFields2dX.laplasUCvCells);
			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdEdxNodes,viscousFields2dX.cdEdyNodes, viscousFields2dX.laplasUCeCells);

			
		end
		
	

		Threads.@threads for p in 1:Threads.nthreads()	
	
	
			#beginCell::Int32 = cellsThreadsX[p,1];
			#endCell::Int32 = cellsThreadsX[p,2];
			#for i = beginCell:endCell

			for i = cellsThreadsX[p,1]:cellsThreadsX[p,2]
				
					
				# if (viscousFields2dX.artViscosityCells[i] > avEpsilon)
					
					
					# UConsDiffCells[i,2] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCuCells[i];
					# UConsDiffCells[i,3] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCvCells[i];
					# UConsDiffCells[i,4] = viscousFields2dX.artViscosityCells[i]*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];

				# end

				T::Float64 = testfields2dX.pressureCells[i]/testfields2dX.densityCells[i]/thermoX.RGAS;
				mu::Float64 = calcAirViscosityPowerLawFromT(T);
				#Pr::Float64 = 3.0/4.0;
		
				
				UConsDiffCells[i,2] = mu*viscousFields2dX.laplasUCuCells[i];
				UConsDiffCells[i,3] = mu*viscousFields2dX.laplasUCvCells[i];
				#UConsDiffCells[i,4] = mu*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];
				UConsDiffCells[i,4] = mu*calcAirConductivityFromT(T,thermoX)*viscousFields2dX.laplasUCeCells[i];


			end
		
		
					
		end
		


end

