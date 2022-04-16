

using Distributed;

const numThreads = 4;


if (numThreads != 1)

	if (nprocs() == 1)
		addprocs(numThreads,lazy=false); 
		display(workers());
	end
		
end

@everywhere using PyPlot;
@everywhere using WriteVTK;
@everywhere using CPUTime;
@everywhere using DelimitedFiles;
@everywhere using Printf
@everywhere using SharedArrays;
@everywhere using HDF5;
@everywhere using BenchmarkTools;

include("primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("initfields2d.jl");


function readMesh2dHDF5(pname::String)::mesh2d_Int32
	
	display("load mesh structure from *hdf5 ")
	CPUtic();
	
	filename = string(pname,".h5");


	nCells = h5open(filename,"r") do file
		read(file,"nCells");
	end
	nNodes = h5open(filename,"r") do file
		read(file,"nNodes");
	end
	nNeibCells = h5open(filename,"r") do file
		read(file,"nNeibCells");
	end	
	nBSets = h5open(filename,"r") do file
		read(file,"nBSets");
	end	
	xNodes = h5open(filename,"r") do file
		read(file,"xNodes");
	end	
	yNodes = h5open(filename,"r") do file
		read(file,"yNodes");
	end	
	mesh_connectivity = h5open(filename,"r") do file
		read(file,"mesh_connectivity");
	end	
	bc_data = h5open(filename,"r") do file
		read(file,"bc_data");
	end	
	bc_indexes = h5open(filename,"r") do file
		read(file,"bc_indexes");
	end	
	cell_nodes_X = h5open(filename,"r") do file
		read(file,"cell_nodes_X");
	end	
	cell_nodes_Y = h5open(filename,"r") do file
		read(file,"cell_nodes_Y");
	end	
	cell_mid_points = h5open(filename,"r") do file
		read(file,"cell_mid_points");
	end	
	cell_areas = h5open(filename,"r") do file
		read(file,"cell_areas");
	end	
	HX = h5open(filename,"r") do file
		read(file,"HX");
	end	
	cell_edges_Nx = h5open(filename,"r") do file
		read(file,"cell_edges_Nx");
	end	
	cell_edges_Ny = h5open(filename,"r") do file
		read(file,"cell_edges_Ny");
	end	
	cell_edges_length = h5open(filename,"r") do file
		read(file,"cell_edges_length");
	end
	cell_stiffness = h5open(filename,"r") do file
		read(file,"cell_stiffness");
	end
	cell_clusters = h5open(filename,"r") do file
		read(file,"cell_clusters");
	end
	node_stencils = h5open(filename,"r") do file
		read(file,"node_stencils");
	end
	node2cellL2up = h5open(filename,"r") do file
		read(file,"node2cellL2up");
	end
	node2cellL2down = h5open(filename,"r") do file
		read(file,"node2cellL2down");
	end
	cells2nodes = h5open(filename,"r") do file
		read(file,"cells2nodes");
	end


	testMesh = mesh2d_Int32(
		Int64(nCells),
		Int64(nNodes),
		nNeibCells,				## max number of neighbors 
		nBSets,					##  number of boundaries  
		xNodes,  				##  mesh_nodes[nNodesx3]
		yNodes, 				##	mesh_nodes[nNodesx3]
		mesh_connectivity, 		## [nCellsx7]
		bc_data,
		bc_indexes,
		cell_nodes_X, 			## [nCellsx4]
		cell_nodes_Y, 			## [nCellsx4]
		cell_mid_points, 		## [nCellsx2]
		cell_areas, 			## [nCellsx1]
		HX,
		cell_edges_Nx, 			## [nCellsx4]
		cell_edges_Ny, 			## [nCellsx4]
		cell_edges_length, 		## [nCellsx4]
		cell_stiffness, 		## [nCellsx4]
		cell_clusters, 			## [nNodesx8]
		node_stencils, 			## [nNodesx8]
		node2cellL2up, 			## [nCellsx8]
		node2cellL2down, 		## [nCellsx8]
		cells2nodes
	);
	
	CPUtoc();
	display("done")
	
	return testMesh;


end


@everywhere  @inline function cells2nodesSolutionReconstructionWithStencilsSA(
	testMesh::mesh2d_Int32,cell_solution::Array{Float64,2}, node_solution::Array{Float64,2} )



	for J=1:testMesh.nNodes
		det::Float64 = 0.0;
		for j = 1:testMesh.nNeibCells
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
	end



end


function cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX:: SharedArray{Int32,2}, 
	testMesh::mesh2d_shared, dummyCells::SharedArray{Float64,2}, dummyNodes::SharedArray{Float64,2})	
	
	
		nNeibCells = size(testMesh.node_stencils,2);
		@everywhere nNeibCellsX = $nNeibCells;
	
	
		@sync @distributed for p in workers()	
		
			beginNode::Int32 = nodesThreadsX[p-1,1];
			endNode::Int32 = nodesThreadsX[p-1,2];
	
			cell_clusters = deepcopy(testMesh.cell_clusters);
			node_stencils = deepcopy(testMesh.node_stencils);
			
			for J=beginNode:endNode
	
				dummyNodes[J,1] = 0.0;
				dummyNodes[J,2] = 0.0;
				dummyNodes[J,3] = 0.0;
				dummyNodes[J,4] = 0.0;
	
				det::Float64 = 0.0;
				#nNeibCells = size(testMesh.node_stencils,2);
				
				for j = 1:nNeibCellsX
		
					#neibCell::Int32 = testMesh.cell_clusters[J,j]; 
					neibCell::Int32 = cell_clusters[J,j]; 
			
					if (neibCell !=0)
						##wi::Float64 = testMesh.node_stencils[J,j];
						wi::Float64 = node_stencils[J,j];
						
						dummyNodes[J,1] += dummyCells[neibCell,1]*wi;
						dummyNodes[J,2] += dummyCells[neibCell,2]*wi;
						dummyNodes[J,3] += dummyCells[neibCell,3]*wi;
						dummyNodes[J,4] += dummyCells[neibCell,4]*wi;
				 
						det += wi;
					end
				end
				
				if (det!=0)
					dummyNodes[J,1] = dummyNodes[J,1]/det; 
					dummyNodes[J,2] = dummyNodes[J,2]/det; 
					dummyNodes[J,3] = dummyNodes[J,3]/det; 
					dummyNodes[J,4] = dummyNodes[J,4]/det; 
				end
				
			end
	
		end ## p workers
	

end




function test1()

	pname = "2dmixinglayerUp_delta3.bson";
	
	
	testMesh = readMesh2dHDF5(pname);
	
	dummyCells = zeros(Float64,testMesh.nCells,4);
	dummyNodes = zeros(Float64,testMesh.nNodes,4);
	
	
	dummyCellsX = SharedArray{Float64}(Int64(testMesh.nCells),4);
	dummyNodesX = SharedArray{Float64}(Int64(testMesh.nNodes),4);
	
	
	nodesThreadsX = distibuteNodesInThreadsSA(numThreads, Int64(testMesh.nNodes) );
	
	testMeshDistr = createMesh2dShared(testMesh);
	

	@everywhere testMeshDistrX = $testMeshDistr; 
	
	
	
	@time cells2nodesSolutionReconstructionWithStencilsSA(testMesh, dummyCells, dummyNodes); 
	@time cells2nodesSolutionReconstructionWithStencilsSA(testMesh, dummyCells, dummyNodes); 
	@time cells2nodesSolutionReconstructionWithStencilsSA(testMesh, dummyCells, dummyNodes); 


	@time cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX, testMeshDistr, dummyCellsX, dummyNodesX)	
	@time cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX, testMeshDistr, dummyCellsX, dummyNodesX)	
	@time cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX, testMeshDistr, dummyCellsX, dummyNodesX)	




end


test1();