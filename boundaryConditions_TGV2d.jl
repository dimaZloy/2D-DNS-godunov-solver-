


@inline function ComputeUPhysFromBoundaries(i::Int32,k::Int32,neib_cell::Int32, 
      cur_cell::Array{Float64,1}, nx::Float64,ny::Float64, y::Float64, gamma::Float64, t::Float64, bnd_cell::Array{Float64,1} )


	bnd_cell[1] = cur_cell[1];	
	bnd_cell[2] = cur_cell[2];	
	bnd_cell[3] = cur_cell[3];	
	bnd_cell[4] = cur_cell[4];	

	
end

@inline function ComputeUPhysFromBoundaries(i::Int32, index::Int32, cur_cell::Array{Float64,1}, testMesh::mesh2d_Int32, thermo::THERMOPHYSICS, t::Float64, bnd_cell::Vector{Float64} )


	 x0::Float64 = testMesh.cell_mid_points[i,1];
	 y0::Float64 = testMesh.cell_mid_points[i,2];

	 node1::Int32 = testMesh.cells2nodes[i,index];
	 node2::Int32 = testMesh.cells2nodes[i,index+1];
	 sp = zeros(Float64,2);
	 getSymmetricalPointAboutCellEdge(x0,y0, testMesh.xNodes[node1], testMesh.yNodes[node1], testMesh.xNodes[node2], testMesh.yNodes[node2], sp);

     computeTGV2d(sp[1], sp[2], t, thermo, bnd_cell);

  
end
