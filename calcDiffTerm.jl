
@inline function cells2nodesSolutionReconstructionWithStencilsVector(nodesThreads::Array{Int32,2}, testMesh::mesh2d_Int32, cell_solution::Array{Float64,1}, node_solution::Array{Float64,1} )

    Threads.@threads for p in 1:Threads.nthreads()
        
        for J = nodesThreads[p,1]:nodesThreads[p,2]
    
            
                det::Float64 = 0.0;
                for j = 1:testMesh.nNeibCells
                    neibCell::Int32 = testMesh.cell_clusters[J,j]; 
                    if (neibCell !=0)
                        wi::Float64 = testMesh.node_stencils[J,j];
                        node_solution[J] += cell_solution[neibCell]*wi;
                        det += wi;
                    end
                end
                if (det!=0)
                    node_solution[J] = node_solution[J]/det; 
                end
            

        end ## J
        
    end ## threads

end



function calcDiffTerm(cellsThreads::Array{Int32,2}, nodesThreads::Array{Int32,2}, testMesh::mesh2d_Int32, 
	testfields2d::fields2d, viscousFields2d::viscousFields2d, thermo::THERMOPHYSICS, 
	UCCells::Array{Float64,2}, UCNodes::Array{Float64,2}, UConsDiffCells::Array{Float64,2})


	
	UC2GradXApprox = zeros(Float64,testMesh.nCells);
	UC2GradYApprox = zeros(Float64,testMesh.nCells);

	UC3GradXApprox = zeros(Float64,testMesh.nCells);
	UC3GradYApprox = zeros(Float64,testMesh.nCells);

	UC4GradXApprox = zeros(Float64,testMesh.nCells);
	UC4GradYApprox = zeros(Float64,testMesh.nCells);

	dummy = zeros(Float64,testMesh.nCells);

	UC2ddUxGradXApprox = zeros(Float64,testMesh.nCells);
	UC2ddUyGradYApprox = zeros(Float64,testMesh.nCells);
	UC3ddUxGradXApprox = zeros(Float64,testMesh.nCells);
	UC3ddUyGradYApprox = zeros(Float64,testMesh.nCells);
	UC4ddUxGradXApprox = zeros(Float64,testMesh.nCells);
	UC4ddUyGradYApprox = zeros(Float64,testMesh.nCells);

	UC2GradXApproxNodes = zeros(Float64,testMesh.nNodes);
	UC2GradYApproxNodes = zeros(Float64,testMesh.nNodes);

	UC3GradXApproxNodes = zeros(Float64,testMesh.nNodes);
	UC3GradYApproxNodes = zeros(Float64,testMesh.nNodes);

	UC4GradXApproxNodes = zeros(Float64,testMesh.nNodes);
	UC4GradYApproxNodes = zeros(Float64,testMesh.nNodes);


	calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UCCells[:,2], UCNodes[:,2], UC2GradXApprox, UC2GradYApprox);
	calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UCCells[:,3], UCNodes[:,3], UC3GradXApprox, UC3GradYApprox);
	calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UCCells[:,4], UCNodes[:,4], UC4GradXApprox, UC4GradYApprox);

	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC2GradXApprox, UC2GradXApproxNodes);
	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC2GradYApprox, UC2GradYApproxNodes);

	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC3GradXApprox, UC3GradXApproxNodes);
	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC3GradYApprox, UC3GradYApproxNodes);

	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC4GradXApprox, UC4GradXApproxNodes);
	cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UC4GradYApprox, UC4GradYApproxNodes);


    # calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC2GradXApprox, UC2GradXApproxNodes, UC2ddUxGradXApprox, UC2ddUxGradYApprox);
    # calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC2GradYApprox, UC2GradYApproxNodes, UC2ddUyGradXApprox, UC2ddUyGradYApprox);

	# calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC3GradXApprox, UC3GradXApproxNodes, UC3ddUxGradXApprox, UC3ddUxGradYApprox);
    # calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC3GradYApprox, UC3GradYApproxNodes, UC3ddUyGradXApprox, UC3ddUyGradYApprox);

	# calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC4GradXApprox, UC4GradXApproxNodes, UC4ddUxGradXApprox, UC4ddUxGradYApprox);
    # calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC4GradYApprox, UC4GradYApproxNodes, UC4ddUyGradXApprox, UC4ddUyGradYApprox);

    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC2GradXApprox, UC2GradXApproxNodes, UC2ddUxGradXApprox, dummy);
    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC2GradYApprox, UC2GradYApproxNodes, dummy, UC2ddUyGradYApprox);

	calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC3GradXApprox, UC3GradXApproxNodes, UC3ddUxGradXApprox, dummy);
    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC3GradYApprox, UC3GradYApproxNodes, dummy, UC3ddUyGradYApprox);

	calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC4GradXApprox, UC4GradXApproxNodes, UC4ddUxGradXApprox, dummy);
    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UC4GradYApprox, UC4GradYApproxNodes, dummy, UC4ddUyGradYApprox);


	Threads.@threads for p in 1:Threads.nthreads()	
	
	
		for i = cellsThreads[p,1]:cellsThreads[p,2]

			#ULapApprox[i] = (ddUxGradXApprox[i] + ddUyGradYApprox[i] );

			viscousFields2d.laplasUCuCells[i] = UC2ddUxGradXApprox[i]  + UC2ddUyGradYApprox[i] ;
			viscousFields2d.laplasUCvCells[i] = UC3ddUxGradXApprox[i]  + UC3ddUyGradYApprox[i] ;
			viscousFields2d.laplasUCeCells[i] = UC4ddUxGradXApprox[i]  + UC4ddUyGradYApprox[i] ;
					
			T::Float64 = testfields2d.pressureCells[i]/testfields2d.densityCells[i]/thermo.RGAS;
			mu::Float64 = 1e-2; 			
			UConsDiffCells[i,2] = mu*viscousFields2d.laplasUCuCells[i];
			UConsDiffCells[i,3] = mu*viscousFields2d.laplasUCvCells[i];
			UConsDiffCells[i,4] = mu*calcAirConductivityFromT(T,thermo)*viscousFields2d.laplasUCeCells[i];
			#UConsDiffCells[i,4] = mu*viscousFields2d.laplasUCeCells[i];


		end
				
	end
	
end


# function calcDiffTerm(cellsThreadsX::Array{Int32,2}, nodesThreadsX::Array{Int32,2}, testMeshDistrX::mesh2d_Int32, 
# 		testfields2dX::fields2d, viscousFields2dX::viscousFields2d, thermoX::THERMOPHYSICS, 
# 		UCNodes::Array{Float64,2}, UConsDiffCells::Array{Float64,2})


# 		Threads.@threads for p in 1:Threads.nthreads()
		
	
# 			#beginCell::Int32 = cellsThreadsX[p,1];
# 			#endCell::Int32 = cellsThreadsX[p,2];
			
# 			# calculate gradients in cells based on UCNodes:
# 			# viscousFields2dX.cdUdxCells[C] 
# 		  	# viscousFields2dX.cdUdyCells[C] 
# 		 	# viscousFields2dX.cdVdxCells[C] 
# 		  	# viscousFields2dX.cdVdyCells[C] 
# 			# viscousFields2dX.cdEdxCells[C] 
# 		  	# viscousFields2dX.cdEdyCells[C] 
# 			nodesGradientReconstructionUconsFastSA(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX, UCNodes, viscousFields2dX);	
			
# 		end
		
# 		## then we need to reconstruct gradients to nodes: 
# 		Threads.@threads for p in 1:Threads.nthreads()
		
	
# 			#beginCell::Int32 = nodesThreadsX[p,1];
# 			#endCell::Int32 = nodesThreadsX[p,2];
# 			# compute :
# 			# viscousFields2dX.cdUdxNodes[C] 
# 		  	# viscousFields2dX.cdUdyNodes[C] 
# 		 	# viscousFields2dX.cdVdxNodes[C] 
# 		  	# viscousFields2dX.cdVdyNodes[C] 
# 			# viscousFields2dX.cdEdxNodes[C] 
# 		  	# viscousFields2dX.cdEdyNodes[C] 
			
# 			cells2nodesSolutionReconstructionWithStencilsViscousGradients(nodesThreadsX[p,1],nodesThreadsX[p,2], testMeshDistrX, viscousFields2dX);	
			
# 		end
		

		
# 		Threads.@threads for p in 1:Threads.nthreads()
	
	
# 			#beginCell::Int32 = cellsThreadsX[p,1];
# 			#endCell::Int32 = cellsThreadsX[p,2];		
# 			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdUdxCells,viscousFields2dX.cdUdyCells, viscousFields2dX.laplasUCuCells);
# 			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdVdxCells,viscousFields2dX.cdVdyCells, viscousFields2dX.laplasUCvCells);
# 			#nodesDivergenceReconstructionFastSA22(beginCell,endCell, testMeshDistrX,  viscousFields2dX.cdEdxCells,viscousFields2dX.cdEdyCells, viscousFields2dX.laplasUCeCells);

# 			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdUdxNodes,viscousFields2dX.cdUdyNodes, viscousFields2dX.laplasUCuCells);
# 			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdVdxNodes,viscousFields2dX.cdVdyNodes, viscousFields2dX.laplasUCvCells);
# 			nodesDivergenceReconstructionFastSA22(cellsThreadsX[p,1],cellsThreadsX[p,2], testMeshDistrX,  viscousFields2dX.cdEdxNodes,viscousFields2dX.cdEdyNodes, viscousFields2dX.laplasUCeCells);

			
# 		end
		
	

# 		Threads.@threads for p in 1:Threads.nthreads()	
	
	
# 			#beginCell::Int32 = cellsThreadsX[p,1];
# 			#endCell::Int32 = cellsThreadsX[p,2];
# 			#for i = beginCell:endCell

# 			for i = cellsThreadsX[p,1]:cellsThreadsX[p,2]
				
					
# 				# if (viscousFields2dX.artViscosityCells[i] > avEpsilon)
					
					
# 					# UConsDiffCells[i,2] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCuCells[i];
# 					# UConsDiffCells[i,3] = viscousFields2dX.artViscosityCells[i]*viscousFields2dX.laplasUCvCells[i];
# 					# UConsDiffCells[i,4] = viscousFields2dX.artViscosityCells[i]*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];

# 				# end

# 				T::Float64 = testfields2dX.pressureCells[i]/testfields2dX.densityCells[i]/thermoX.RGAS;
# 				mu::Float64 = calcAirViscosityPowerLawFromT(T);
# 				#Pr::Float64 = 3.0/4.0;
		
				
# 				UConsDiffCells[i,2] = mu*viscousFields2dX.laplasUCuCells[i];
# 				UConsDiffCells[i,3] = mu*viscousFields2dX.laplasUCvCells[i];
# 				#UConsDiffCells[i,4] = mu*thermoX.Cp/Pr*viscousFields2dX.laplasUCeCells[i];
# 				UConsDiffCells[i,4] = mu*calcAirConductivityFromT(T,thermoX)*viscousFields2dX.laplasUCeCells[i];


# 			end
		
		
					
# 		end
		


# end
