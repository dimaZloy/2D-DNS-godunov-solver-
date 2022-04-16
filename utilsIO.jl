
 function saveResults2VTK(
		 filename::String, 
		 testMesh::mesh2d_Int32, 
		 variable::Array{Float64,1}, variableName::String )

	 vtkfile = vtk_grid(filename, testMesh.xNodes, testMesh.yNodes, testMesh.VTKCells);
	 vtk_point_data(vtkfile, variable, variableName);
	 outfiles = vtk_save(vtkfile);
	
end

 function saveResults2VTK(
		 filename::String, 
		 testMesh::mesh2d_Int32, 
		 variable::SharedArray{Float64,1}, variableName::String )

	 vtkfile = vtk_grid(filename, testMesh.xNodes, testMesh.yNodes, testMesh.VTKCells);
	 vtk_point_data(vtkfile, variable, variableName);
	 outfiles = vtk_save(vtkfile);
	
end

function saveResults2VTK(filename::String, 
		testMesh::mesh2d_Int32,field::Array{Float64,1})

	vtkfile = vtk_grid(filename, testMesh.xNodes, testMesh.yNodes, testMesh.VTKCells);
	vtk_point_data(vtkfile, field, "density");
	outfiles = vtk_save(vtkfile);
	
end

function saveResults3VTK(filename::String, 
		testMesh::mesh2d_Int32,field1::Array{Float64,1},field2::Array{Float64,1})

	vtkfile = vtk_grid(filename, testMesh.xNodes, testMesh.yNodes, testMesh.VTKCells);
	vtk_point_data(vtkfile, field1, "density");
	vtk_point_data(vtkfile, field2, "artificialViscosity");
	outfiles = vtk_save(vtkfile);
	
end

function saveResults4VTK(filename::String, 
		testMesh::mesh2d_Int32,field1::Array{Float64,1},field2::Array{Float64,1},field3::Array{Float64,1},field4::Array{Float64,1})

	vtkfile = vtk_grid(filename, testMesh.xNodes, testMesh.yNodes, testMesh.VTKCells);
	vtk_point_data(vtkfile, field1, "density");
	vtk_point_data(vtkfile, field2, "Ux");
	vtk_point_data(vtkfile, field3, "Uy");
	vtk_point_data(vtkfile, field4, "pressure");
	outfiles = vtk_save(vtkfile);
	
end

function readSolution(fileName, nNodes)
	
	io = open(fileName,"r");
	line  = readline(io);
	line  = readline(io);
	
	xNodes = zeros(Float64,nNodes);
	yNodes = zeros(Float64,nNodes);
	nodesSolution = zeros(Float64,nNodes);
	
	
	for i=1:nNodes
		
		line = readline(io);
		z = split(line);
		xNodes[i] = parse(Float64,z[1]);
		yNodes[i] = parse(Float64,z[2]);
		nodesSolution[i] = parse(Float64,z[3]);
		
	end	
	close(io);
	
	return xNodes, yNodes, nodesSolution; 

end


function saveSolution(fileName, xNodes, yNodes, nodesSolution)

io = open(fileName,"w");
for i=1:size(xNodes,1)
	writedlm(io, [xNodes[i]  yNodes[i] nodesSolution[i,1] nodesSolution[i,2] nodesSolution[i,3] nodesSolution[i,4] ], '\t' );
end


close(io);

end

function saveResiduals(fileName, timeVector, residualsVector1, residualsVector2, residualsVector3, residualsVector4)

io = open(fileName,"w");
	
for i=1:size(timeVector,1)
	writedlm(io, [ timeVector[i]  residualsVector1[i] residualsVector2[i] residualsVector3[i] residualsVector4[i]  ], '\t' );
end

close(io);

end


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


	VTKCells = MeshCell[];
	triangles = zeros(Int32,nCells,3);
	## prepare elements stucture for PyPlot trincontourf and VTK output
	
	for i = 1:nCells
	
		## indexes of nodes in PyPLOT are started from Zero!!!!
		
		cellType::Int32 = mesh_connectivity[i,2];
	
		triangles[i,1] =  mesh_connectivity[i,4]-1;
		triangles[i,2] =  mesh_connectivity[i,5]-1;
		triangles[i,3] =  mesh_connectivity[i,6]-1;
		
		if (cellType == 2) ## quads
			inds = Array{Int32}(undef, 4);
			inds[1] =    mesh_connectivity[i,4];
			inds[2] =    mesh_connectivity[i,5];
			inds[3] =    mesh_connectivity[i,6];
			inds[4] =    mesh_connectivity[i,7];
			c = MeshCell(VTKCellTypes.VTK_QUAD, inds);
			push!(VTKCells, c);
		
		elseif (cellType == 3) ## triangle
	
			inds = Array{Int32}(undef, 3);
			inds[1] =   mesh_connectivity[i,4];
			inds[2] =   mesh_connectivity[i,5];
			inds[3] =   mesh_connectivity[i,6];
			c = MeshCell(VTKCellTypes.VTK_TRIANGLE, inds);
			push!(VTKCells, c);
		end
		
		
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
		node2cellL2up, 		## [nCellsx8]
		node2cellL2down, 		## [nCellsx8]
		cells2nodes,
		VTKCells, 
		triangles
	);


	# display(nCells);
	# display(nNodes);
	# display(nNeibCells)
	# display(nBSets);
	# display(xNodes);
	# display(yNodes);
	# display(mesh_connectivity);
	# display(bc_data);
	# display(bc_indexes);
	# display(cell_nodes_X);
	# display(cell_nodes_Y);
	# display(cell_mid_points);
	# display(cell_areas);
	# display(HX);
	# display(cell_edges_Nx);
	# display(cell_edges_Ny);
	# display(cell_edges_length);
	# display(cell_stiffness);
	# display(cell_clusters);
	# display(node_stencils);
	# display(node2cellL2up);
	# display(node2cellL2down);
	# display(cells2nodes);
	
	
	
	# h5open(fn,"r") do file
	
		# read(file,"nCells", nCells);
		# write(file,"nNodes", nNodes);
		# write(file,"nNeibCells", nNeibCells);
		# write(file,"nBSets",nBSets );
		# write(file,"xNodes",xNodes );
		# write(file,"yNodes",yNodes);
		# write(file,"mesh_connectivity",mesh_connectivity );
		# write(file,"bc_data", bc_data);
		# write(file,"bc_indexes",bc_indexes );
		# write(file,"cell_nodes_X", cell_nodes_X);
		# write(file,"cell_nodes_Y", cell_nodes_Y);
		# write(file,"cell_mid_points", cell_mid_points);
		# write(file,"cell_areas", cell_areas);
		# write(file,"HX", HX);
		# write(file,"cell_edges_Nx",cell_edges_Nx );
		# write(file,"cell_edges_Ny", cell_edges_Ny);
		# write(file,"cell_edges_length",cell_edges_length );
		# write(file,"cell_stiffness",cell_stiffness );
		# write(file,"cell_clusters", cell_clusters);
		# write(file,"node_stencils", node_stencils);
		# #write(file,"VTKCells", VTKCells);
		# write(file,"node2cellL2up",node2cellL2up );
		# write(file,"node2cellL2down", node2cellL2down);
		# write(file,"cells2nodes",cells2nodes );
	
	# end
	
	CPUtoc();
	display("done")
	
	return testMesh;


end
