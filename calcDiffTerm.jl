

function calcDiffTerm(cellsThreadsX::Array{Int32,2}, testMeshDistrX::mesh2d_Int32, 
		testfields2dX::fields2d, viscousFields2dX::viscousFields2d, thermoX::THERMOPHYSICS, 
		UCNodes::Array{Float64,2}, UConsDiffCells::Array{Float64,2}, UConsDiffNodes::Array{Float64,2}, dampCells::Array{Float64,1} )


		Threads.@threads for p in 1:Threads.nthreads()
		#@sync @distributed for p in workers()	
	
			beginCell::Int32 = cellsThreadsX[p,1];
			endCell::Int32 = cellsThreadsX[p,2];
			
			#nodesGradientReconstructionUconsFastSA33(beginCell,endCell, testMeshDistrX, UCNodes, viscousFields2dX);	
			
			
			nodesGradientReconstructionUconsFastSA(beginCell,endCell, testMeshDistrX, UCNodes, viscousFields2dX);	
			
		end
		
		
		Threads.@threads for p in 1:Threads.nthreads()
		#@sync @distributed for p in workers()	
	
			beginCell::Int32 = cellsThreadsX[p,1];
			endCell::Int32 = cellsThreadsX[p,2];
			
			nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdUdxCells,viscousFields2dX.cdUdyCells, viscousFields2dX.laplasUCuCells);
			nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdVdxCells,viscousFields2dX.cdVdyCells, viscousFields2dX.laplasUCvCells);
			nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdEdxCells,viscousFields2dX.cdEdyCells, viscousFields2dX.laplasUCeCells);

			# nodesDivergenceReconstructionFastSA33(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdUdxCells,viscousFields2dX.cdUdyCells, viscousFields2dX.laplasUCuCells);
			# nodesDivergenceReconstructionFastSA33(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdVdxCells,viscousFields2dX.cdVdyCells, viscousFields2dX.laplasUCvCells);
			# nodesDivergenceReconstructionFastSA33(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdEdxCells,viscousFields2dX.cdEdyCells, viscousFields2dX.laplasUCeCells);

			
			
		end
		
		
		##display(viscousFields2dX.laplasUCeCells)

		Threads.@threads for p in 1:Threads.nthreads()	
		#@sync @distributed for p in workers()	
	
			beginCell::Int32 = cellsThreadsX[p,1];
			endCell::Int32 = cellsThreadsX[p,2];
			#avEpsilon::Float64 = 1e-5;
			#Pr::Float64 = 3.0/4.0;
			
			
			for i = beginCell:endCell
			
					
				# if (viscousFields2dX.artViscosityCells[i] > avEpsilon)
					
					
					# UConsDiffCells[i,2] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCuCells[i];
					# UConsDiffCells[i,3] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCvCells[i];
					# UConsDiffCells[i,4] = viscousFields2dX.artViscosityCells[i]*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];

				# end
				
				# UConsDiffCells[i,2] = avEpsilon*viscousFields2dX.laplasUCuCells[i];
				# UConsDiffCells[i,3] = avEpsilon*viscousFields2dX.laplasUCvCells[i];
				# UConsDiffCells[i,4] = avEpsilon*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];

				UConsDiffCells[i,2] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCuCells[i]*dampCells[i];
				UConsDiffCells[i,3] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCvCells[i]*dampCells[i];
				UConsDiffCells[i,4] = testfields2dX.kCells[i]*viscousFields2dX.laplasUCeCells[i]*dampCells[i];
				#UConsDiffCells[i,4] = viscousFields2dX.artViscosityCells[i]*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i]*dampCells[i];
				

			end
		
		
					
		end
		


end

