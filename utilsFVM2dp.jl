
# utilities for FVM 

@everywhere @inline function phs2dcns2dcellsSA( ACons::Array{Float64,2}, testFields::fields2d, gamma::Float64)

	#N::Int64 = size(testFields.densityCells,1);
	

	for i = 1:size(testFields.densityCells,1)
		ACons[i,1] = testFields.densityCells[i];
		ACons[i,2] = testFields.densityCells[i]*testFields.UxCells[i];
		ACons[i,3] = testFields.densityCells[i]*testFields.UyCells[i];
		ACons[i,4] = testFields.pressureCells[i]/(gamma-1.0) + 0.5*testFields.densityCells[i]*(	testFields.UxCells[i]*testFields.UxCells[i] +  testFields.UyCells[i]*testFields.UyCells[i] );

	end #for
	
end



@everywhere  @inline function cells2nodesSolutionReconstructionWithStencilsSA(
		testMesh::mesh2d_Int32,cell_solution::SharedArray{Float64,1} ) ::Array{Float64,1}

node_solution = zeros(Float64,testMesh.nNodes); 

for J=1:testMesh.nNodes
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
end

return node_solution;	

end


@everywhere  @inline function cells2nodesSolutionReconstructionWithStencilsSA(testMesh::mesh2d_Int32,cell_solution::Array{Float64,1}, node_solution::Array{Float64,1} )

#node_solution = zeros(Float64,testMesh.nNodes); 

	for J=1:testMesh.nNodes
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
	end

#return node_solution;	

end



@everywhere function cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX:: Array{Int32,2}, 
	testMesh::mesh2d_Int32, testFields::fields2d, dummy::Array{Float64,2})	

		#nNodes = size(testMesh.cell_clusters,1);
		#node_solution = SharedArray{Float64}(nNodes,4); 
	
		#@sync @distributed for p in workers()
		Threads.@threads for p in 1:Threads.nthreads()
		
			beginNode::Int32 = nodesThreadsX[p,1];
			endNode::Int32 = nodesThreadsX[p,2];
	
			
			for J=beginNode:endNode
	
				dummy[J,1] = 0.0;
				dummy[J,2] = 0.0;
				dummy[J,3] = 0.0;
				dummy[J,4] = 0.0;
	
				det::Float64 = 0.0;
				nNeibCells = size(testMesh.node_stencils,2);
				
				for j = 1:nNeibCells
		
					neibCell::Int32 = testMesh.cell_clusters[J,j]; 
			
					if (neibCell !=0)
						wi::Float64 = testMesh.node_stencils[J,j];
						#node_solution[J,:] += cell_solution[neibCell,:];
						dummy[J,1] += testFields.densityCells[neibCell]*wi;
						dummy[J,2] += testFields.UxCells[neibCell]*wi;
						dummy[J,3] += testFields.UyCells[neibCell]*wi;
						dummy[J,4] += testFields.pressureCells[neibCell]*wi;
				 
						det += wi;
					end
				end
				
				if (det!=0)
					dummy[J,1] = dummy[J,1]/det; 
					dummy[J,2] = dummy[J,2]/det; 
					dummy[J,3] = dummy[J,3]/det; 
					dummy[J,4] = dummy[J,4]/det; 
				end
				
			end

	
			for J = beginNode:endNode
	
				testFields.densityNodes[J] = dummy[J,1]; 
				testFields.UxNodes[J] 	   = dummy[J,2]; 
				testFields.UyNodes[J] 	   = dummy[J,3]; 
				testFields.pressureNodes[J] =  dummy[J,4]; 
		
			end
	
	
	
	
	
		end ## p workers
	

end



# @inline function cells2nodesSolutionReconstructionWithStencilsSerial(
	# testMesh::mesh2d_Int32,cell_solution::Array{Float64,1}, node_solution::Array{Float64,1} )


	# for J=1:testMesh.nNodes
		# det::Float64 = 0.0;
		# for j = 1:testMesh.nNeibCells
			# neibCell::Int32 = testMesh.cell_clusters[J,j]; 
			# if (neibCell !=0)
				# wi::Float64 = testMesh.node_stencils[J,j];
				# node_solution[J] += cell_solution[neibCell]*wi;
				# det += wi;
			# end
		# end
		# if (det!=0)
			# node_solution[J] = node_solution[J]/det; 
		# end
	# end


# end

# @inline function cells2nodesSolutionReconstructionWithStencilsSerial(
	# testMesh::mesh2d_Int32,cell_solution::SharedArray{Float64,1}, node_solution::SharedArray{Float64,1} )


	# for J=1:testMesh.nNodes
		# det::Float64 = 0.0;
		# for j = 1:testMesh.nNeibCells
			# neibCell::Int32 = testMesh.cell_clusters[J,j]; 
			# if (neibCell !=0)
				# wi::Float64 = testMesh.node_stencils[J,j];
				# node_solution[J] += cell_solution[neibCell]*wi;
				# det += wi;
			# end
		# end
		# if (det!=0)
			# node_solution[J] = node_solution[J]/det; 
		# end
	# end


# end


# @inline function cells2nodesSolutionReconstructionWithStencilsSerialUCons(
	# testMesh::mesh2d_Int32, cell_solution::SharedArray{Float64,2}, node_solution::SharedArray{Float64,2})

			
			
			# for J= 1 : testMesh.nNodes
			
				# det::Float64 = 0.0;
				
				# nNeibCells = size(testMesh.cell_clusters,2);
				
				# node_solution[J,1] = 0.0;
				# node_solution[J,2] = 0.0;
				# node_solution[J,3] = 0.0;
				# node_solution[J,4] = 0.0;
				
				# for j = 1:nNeibCells
				
					# neibCell::Int32 = testMesh.cell_clusters[J,j]; 
					
					# if (neibCell !=0)
						# wi::Float64 = testMesh.node_stencils[J,j];
						# node_solution[J,1] += cell_solution[neibCell,1]*wi;
						# node_solution[J,2] += cell_solution[neibCell,2]*wi;
						# node_solution[J,3] += cell_solution[neibCell,3]*wi;
						# node_solution[J,4] += cell_solution[neibCell,4]*wi;
					
						# det += wi;
					# end
				# end
				
				# if (det!=0)
					# node_solution[J,1] = node_solution[J,1]/det; 
					# node_solution[J,2] = node_solution[J,2]/det; 
					# node_solution[J,3] = node_solution[J,3]/det; 
					# node_solution[J,4] = node_solution[J,4]/det; 
				# end
				
				
			# end ## for

# end



# @inline function cells2nodesSolutionReconstructionWithStencilsSerial(
	# testMesh::mesh2d_Int32, testfields2d::fields2d_shared, viscfields2d::viscousFields2d_shared, 
	# cell_solution::SharedArray{Float64,2}, node_solution::SharedArray{Float64,2})

			
			
# @fastmath	for J= 1 : testMesh.nNodes
			
				# det::Float64 = 0.0;
				
				# nNeibCells = size(testMesh.cell_clusters,2);
				
				# testfields2d.densityNodes[J] 		= 0.0;
				# testfields2d.UxNodes[J] 			= 0.0;
				# testfields2d.UyNodes[J] 			= 0.0;
				# testfields2d.pressureNodes[J] 		= 0.0;
				# viscfields2d.artViscosityNodes[J] 	= 0.0;
				
				# node_solution[J,1] = 0.0;
				# node_solution[J,2] = 0.0;
				# node_solution[J,3] = 0.0;
				# node_solution[J,4] = 0.0;
				
				# for j = 1:nNeibCells
				
					# neibCell::Int32 = testMesh.cell_clusters[J,j]; 
					
					# if (neibCell !=0)
						# wi::Float64 = testMesh.node_stencils[J,j];
						
						
						# node_solution[J,1] 					+= cell_solution[neibCell,1]*wi;
						# node_solution[J,2] 					+= cell_solution[neibCell,2]*wi;
						# node_solution[J,3] 					+= cell_solution[neibCell,3]*wi;
						# node_solution[J,4] 					+= cell_solution[neibCell,4]*wi;
						
						# testfields2d.densityNodes[J] 		+=  testfields2d.densityCells[neibCell]*wi;
						# testfields2d.UxNodes[J] 			+=  testfields2d.UxCells[neibCell]*wi;
						# testfields2d.UyNodes[J] 			+=  testfields2d.UyCells[neibCell]*wi;
						# testfields2d.pressureNodes[J] 		+=  testfields2d.pressureCells[neibCell]*wi;
						
						# viscfields2d.artViscosityNodes[J] 	+= viscfields2d.artViscosityCells[neibCell]*wi;
						
						# det += wi;
					# end
				# end
				
				# if (det!=0)
				
					# testfields2d.densityNodes[J] = testfields2d.densityNodes[J]/det; 
					# testfields2d.UxNodes[J] = testfields2d.UxNodes[J]/det; 
					# testfields2d.UyNodes[J] = testfields2d.UyNodes[J]/det; 
					# testfields2d.pressureNodes[J] = testfields2d.pressureNodes[J]/det; 
					# viscfields2d.artViscosityNodes[J] = viscfields2d.artViscosityNodes[J]/det;
					
					# node_solution[J,1] = node_solution[J,1]/det; 
					# node_solution[J,2] = node_solution[J,2]/det; 
					# node_solution[J,3] = node_solution[J,3]/det; 
					# node_solution[J,4] = node_solution[J,4]/det; 
				# end
				
				
			# end ## for



# end


@everywhere @inline function cells2nodesSolutionReconstructionWithStencilsDistributed(beginNode::Int32, endNode::Int32,
	testMesh::mesh2d_Int32, testfields2d::fields2d, viscfields2d::viscousFields2d, 
	cell_solution::Array{Float64,2}, node_solution::Array{Float64,2})

			
			
@fastmath	for J= beginNode:endNode
			
				det::Float64 = 0.0;
				
				nNeibCells = size(testMesh.cell_clusters,2);
				
				testfields2d.densityNodes[J] 		= 0.0;
				testfields2d.UxNodes[J] 			= 0.0;
				testfields2d.UyNodes[J] 			= 0.0;
				testfields2d.pressureNodes[J] 		= 0.0;
				viscfields2d.artViscosityNodes[J] 	= 0.0;
				
				node_solution[J,1] = 0.0;
				node_solution[J,2] = 0.0;
				node_solution[J,3] = 0.0;
				node_solution[J,4] = 0.0;
				
				for j = 1:nNeibCells
				
					neibCell::Int32 = testMesh.cell_clusters[J,j]; 
					
					if (neibCell !=0)
						wi::Float64 = testMesh.node_stencils[J,j];
						
						
						node_solution[J,1] 					+= cell_solution[neibCell,1]*wi;
						node_solution[J,2] 					+= cell_solution[neibCell,2]*wi;
						node_solution[J,3] 					+= cell_solution[neibCell,3]*wi;
						node_solution[J,4] 					+= cell_solution[neibCell,4]*wi;
						
						testfields2d.densityNodes[J] 		+=  testfields2d.densityCells[neibCell]*wi;
						testfields2d.UxNodes[J] 			+=  testfields2d.UxCells[neibCell]*wi;
						testfields2d.UyNodes[J] 			+=  testfields2d.UyCells[neibCell]*wi;
						testfields2d.pressureNodes[J] 		+=  testfields2d.pressureCells[neibCell]*wi;
						
						viscfields2d.artViscosityNodes[J] 	+= viscfields2d.artViscosityCells[neibCell]*wi;
						
						det += wi;
					end
				end
				
				if (det!=0)
				
					testfields2d.densityNodes[J] = testfields2d.densityNodes[J]/det; 
					testfields2d.UxNodes[J] = testfields2d.UxNodes[J]/det; 
					testfields2d.UyNodes[J] = testfields2d.UyNodes[J]/det; 
					testfields2d.pressureNodes[J] = testfields2d.pressureNodes[J]/det; 
					viscfields2d.artViscosityNodes[J] = viscfields2d.artViscosityNodes[J]/det;
					
					node_solution[J,1] = node_solution[J,1]/det; 
					node_solution[J,2] = node_solution[J,2]/det; 
					node_solution[J,3] = node_solution[J,3]/det; 
					node_solution[J,4] = node_solution[J,4]/det; 
				end
				
				
			end ## for



end




@everywhere  @inline function cells2nodesSolutionReconstructionWithStencilsUCons(nodesThreadsX:: Array{Int32,2}, testMesh::mesh2d_Int32, 
	cell_solution::Array{Float64,2}, node_solution::Array{Float64,2} )

	
	
		#@sync @distributed for p in workers()
		Threads.@threads for p in 1:Threads.nthreads()
		
			beginNode::Int32 = nodesThreadsX[p,1];
			endNode::Int32 = nodesThreadsX[p,2];
			
			for J=beginNode:endNode
			
				det::Float64 = 0.0;
				
				nNeibCells = size(testMesh.cell_clusters,2);
				
				node_solution[J,1] = 0.0;
				node_solution[J,2] = 0.0;
				node_solution[J,3] = 0.0;
				node_solution[J,4] = 0.0;
				
				for j = 1:nNeibCells
				
					neibCell::Int32 = testMesh.cell_clusters[J,j]; 
					
					if (neibCell !=0)
						wi::Float64 = testMesh.node_stencils[J,j];
						node_solution[J,1] += cell_solution[neibCell,1]*wi;
						node_solution[J,2] += cell_solution[neibCell,2]*wi;
						node_solution[J,3] += cell_solution[neibCell,3]*wi;
						node_solution[J,4] += cell_solution[neibCell,4]*wi;
					
						det += wi;
					end
				end
				
				if (det!=0)
					node_solution[J,1] = node_solution[J,1]/det; 
					node_solution[J,2] = node_solution[J,2]/det; 
					node_solution[J,3] = node_solution[J,3]/det; 
					node_solution[J,4] = node_solution[J,4]/det; 
				end
				
				
			end ## for

		

		end ## p workers 


end
