




# function createMesh2dShared(testMesh::mesh2d_Int32)::mesh2d_shared



	# n = size(testMesh.mesh_connectivity,2);
	# mesh_connectivity = SharedArray{Int32}(testMesh.nCells,n);
	
	# n = size(testMesh.cell_edges_length,2);
	# cell_edges_length = SharedArray{Float64}(testMesh.nCells,n);
	
	# n = size(testMesh.cell_edges_Nx,2);
	# cell_edges_Nx = SharedArray{Float64}(testMesh.nCells,n);
	# cell_edges_Ny = SharedArray{Float64}(testMesh.nCells,n);
	
	# n = size(testMesh.cell_stiffness,2);
	# cell_stiffness = SharedArray{Int32}(testMesh.nCells,n);
	
	# #n = size(testMesh.Z,2);
	# Z = SharedArray{Float64}(testMesh.nCells);
	
	# cell_areas = SharedArray{Float64}(testMesh.nCells);
	# cell_mid_points = SharedArray{Float64}(testMesh.nCells,2);

	# cell_clusters = SharedArray{Int32}(testMesh.nNodes, Int64(testMesh.nNeibCells));
	# node_stencils = SharedArray{Float64}(testMesh.nNodes, Int64(testMesh.nNeibCells));
	
	
	# for i = 1:testMesh.nNodes
		# cell_clusters[i,:] = testMesh.cell_clusters[i,:];
		# node_stencils[i,:] = testMesh.node_stencils[i,:];
	# end
	
	
	# node2cellsL2up = SharedArray{Int32}(testMesh.nCells,8);
	# node2cellsL2down = SharedArray{Int32}(testMesh.nCells,8);
	
	
	# HX = SharedArray{Float64}(testMesh.nCells);
	# cells2nodes = SharedArray{Int32}(testMesh.nCells,8);
	
	
	# for i = 1:testMesh.nCells
	
		# mesh_connectivity[i,:] = testMesh.mesh_connectivity[i,:];
		# cell_mid_points[i,:] = testMesh.cell_mid_points[i,:];
		# cell_edges_length[i,:] = testMesh.cell_edges_length[i,:];
		# cell_edges_Nx[i,:] = testMesh.cell_edges_Nx[i,:];
		# cell_edges_Ny[i,:] = testMesh.cell_edges_Ny[i,:];
		# cell_stiffness[i,:] = testMesh.cell_stiffness[i,:];
		# Z[i] = 1.0/testMesh.cell_areas[i];
		# HX[i] = testMesh.HX[i];
		# cell_areas[i]  = testMesh.cell_areas[i];
		# node2cellsL2up[i,:] = testMesh.node2cellsL2up[i,:];
		# node2cellsL2down[i,:] = testMesh.node2cellsL2down[i,:];
		# cells2nodes[i,:] = testMesh.cells2nodes[i,:];
	
	# end
	
	
	
	# testMeshDistr = mesh2d_shared(
		# mesh_connectivity,
		# cell_mid_points,
		# cell_areas,
		# Z,
		# HX, 
		# cell_edges_Nx,
		# cell_edges_Ny,
		# cell_edges_length,
		# cell_stiffness,
		# cell_clusters,
		# node_stencils,
		# node2cellsL2up,
		# node2cellsL2down,
		# cells2nodes);
	
	
	# return testMeshDistr;

# end




function createFields2dLoadPrevResults_shared(testMesh::mesh2d_Int32, thermo::THERMOPHYSICS, filename::String, dynControls::DYNAMICCONTROLS )

	
	
	println("try to read previous solution from ", filename);

	@load filename solInst;
	
	
	
	densityCells = SharedArray{Float64}(testMesh.nCells); 
	UxCells = SharedArray{Float64}(testMesh.nCells); 
	UyCells = SharedArray{Float64}(testMesh.nCells); 
	pressureCells = SharedArray{Float64}(testMesh.nCells); 
	aSoundCells = SharedArray{Float64}(testMesh.nCells); #speed of sound
	VMAXCells = SharedArray{Float64}(testMesh.nCells); #max speed in domain
	
	densityNodes = SharedArray{Float64}(testMesh.nNodes); 
	UxNodes = SharedArray{Float64}(testMesh.nNodes); 
	UyNodes = SharedArray{Float64}(testMesh.nNodes); 
	pressureNodes = SharedArray{Float64}(testMesh.nNodes); 

	for i=1:testMesh.nCells

	
		densityCells[i] 	=  solInst.densityCells[i];
		UxCells[i] 			=  solInst.UxCells[i];
		UyCells[i] 			=  solInst.UyCells[i]; 
		pressureCells[i] 	=  solInst.pressureCells[i];
		
		aSoundCells[i] = sqrt( thermo.Gamma * pressureCells[i]/densityCells[i] );
		VMAXCells[i]  = sqrt( UxCells[i]*UxCells[i] + UyCells[i]*UyCells[i] ) + aSoundCells[i];
				
	end


	# create fields 
	testFields2d = fields2d_shared(
		densityCells,
		UxCells,
		UyCells,
		pressureCells,
		aSoundCells,
		VMAXCells,
		densityNodes,
		UxNodes,
		UyNodes,
		pressureNodes
		#UconsCellsOld,
		#UconsCellsNew
	);

	dynControls.flowTime = solInst.flowTime;
	
	#tmp = split(filename,"zzz");
	#num::Int64 = parse(Int64,tmp[2]); 
	#dynControls.curIter = num - 1000;

	return testFields2d, solInst;


end


function createViscousFields2d(nCells::Int64, nNodes::Int64)::viscousFields2d

	artViscosityCells = zeros(Float64,nCells);
	artViscosityNodes = zeros(Float64,nNodes);
	
	dUdxCells = zeros(Float64,nCells);
	dUdyCells = zeros(Float64,nCells);
	
	dVdxCells = zeros(Float64,nCells);
	dVdyCells = zeros(Float64,nCells);
	
	laplasUCuCells = zeros(Float64,nCells);
	laplasUCvCells = zeros(Float64,nCells);
	laplasUCeCells = zeros(Float64,nCells);
	
	cdUdxCells = zeros(Float64,nCells);
	cdUdyCells = zeros(Float64,nCells);
	
	cdVdxCells = zeros(Float64,nCells);
	cdVdyCells = zeros(Float64,nCells);
	cdEdxCells = zeros(Float64,nCells);
	cdEdyCells = zeros(Float64,nCells);
	
	viscous2d = viscousFields2d(
		artViscosityCells,
		artViscosityNodes,
		dUdxCells,
		dUdyCells,
		dVdxCells,
		dVdyCells,
		laplasUCuCells,
		laplasUCvCells,
		laplasUCeCells,
		cdUdxCells,
		cdUdyCells,
		cdVdxCells,
		cdVdyCells,
		cdEdxCells,
		cdEdyCells
	);

	return viscous2d; 
	

end


function createFields2d(testMesh::mesh2d_Int32, thermo::THERMOPHYSICS)


	densityCells =  zeros(Float64,testMesh.nCells); 
	UxCells =       zeros(Float64,testMesh.nCells); 
	UyCells =       zeros(Float64,testMesh.nCells); 
	pressureCells = zeros(Float64,testMesh.nCells); 
	aSoundCells   = zeros(Float64,testMesh.nCells); #speed of sound
	VMAXCells     = zeros(Float64,testMesh.nCells); #max speed in domain
	
	densityNodes  = zeros(Float64,testMesh.nNodes); 
	UxNodes       = zeros(Float64,testMesh.nNodes); 
	UyNodes       = zeros(Float64,testMesh.nNodes); 
	pressureNodes = zeros(Float64,testMesh.nNodes); 

	for i=1:testMesh.nCells

		densityCells[i] 	= 1.1766;
		UxCells[i] 			= 1232.6445;
		UyCells[i] 			= 0.0; 
		pressureCells[i] 	= 101325.0;
		
		aSoundCells[i] = sqrt( thermo.Gamma * pressureCells[i]/densityCells[i] );
		VMAXCells[i]  = sqrt( UxCells[i]*UxCells[i] + UyCells[i]*UyCells[i] ) + aSoundCells[i];
		#entropyCell[i] = UphysCells[i,1]/(thermo.Gamma-1.0)*log(UphysCells[i,4]/UphysCells[i,1]*thermo.Gamma);
				
	end
		


	# create fields 
	testFields2d = fields2d(
		densityCells,
		UxCells,
		UyCells,
		pressureCells,
		aSoundCells,
		VMAXCells,
		densityNodes,
		UxNodes,
		UyNodes,
		pressureNodes
		#UconsCellsOld,
		#UconsCellsNew
	);

	return testFields2d; 


end

