

@everywhere function nodesDivergenceReconstructionFastSA22( beginCell::Int32, endCell::Int32, testMesh::mesh2d_Int32, 
  gradX::Array{Float64,1}, gradY::Array{Float64,1}, divergence::Array{Float64,1})
  
  
		phiLeftX = zeros(Float64,4);
		phiLeftY = zeros(Float64,4);

		phiRightX = zeros(Float64,4);
		phiRightY = zeros(Float64,4);

		phiFaceX = zeros(Float64,4);
		phiFaceY  = zeros(Float64,4);

		side = zeros(Float64,4);
		nx = zeros(Float64,4);
		ny = zeros(Float64,4);
		
		T1::Int32 = 0;
		T2::Int32 = 0;
		T3::Int32 = 0;
		T4::Int32 = 0;
 

  for C = beginCell:endCell
  

		##numNodesInCell::Int32 = testMesh.mesh_connectivity[C,3]; ## CMatrix mesh_connectivity - first index == 1
		
		
		T1 = testMesh.mesh_connectivity[C,4];
		T2 = testMesh.mesh_connectivity[C,5];
		T3 = testMesh.mesh_connectivity[C,6];
	  
		side[1] = testMesh.cell_edges_length[C,1];
		side[2] = testMesh.cell_edges_length[C,2];
		side[3] = testMesh.cell_edges_length[C,3];
	  
		nx[1] = testMesh.cell_edges_Nx[C,1];
		nx[2] = testMesh.cell_edges_Nx[C,2];
		nx[3] = testMesh.cell_edges_Nx[C,3];
	  
		ny[1] = testMesh.cell_edges_Nx[C,1];
		ny[2] = testMesh.cell_edges_Nx[C,2];
		ny[3] = testMesh.cell_edges_Nx[C,3];
		

		phiLeftX[1] =  gradX[ testMesh.cells2nodes[C,1] ];
		phiLeftY[1] =  gradY[ testMesh.cells2nodes[C,1] ];
	
		phiRightX[1] = gradX[ testMesh.cells2nodes[C,2] ];
		phiRightY[1] = gradY[ testMesh.cells2nodes[C,2] ];

		phiLeftX[2] =  gradX[ testMesh.cells2nodes[C,3] ];
		phiLeftY[2] =  gradY[ testMesh.cells2nodes[C,3] ];
	
		phiRightX[2] = gradX[ testMesh.cells2nodes[C,4] ];
		phiRightY[2] = gradY[ testMesh.cells2nodes[C,4] ];
	
		phiLeftX[3] =  gradX[ testMesh.cells2nodes[C,5] ];
		phiLeftY[3] =  gradY[ testMesh.cells2nodes[C,5] ];
	
		phiRightX[3] = gradX[ testMesh.cells2nodes[C,6] ];
		phiRightY[3] = gradY[ testMesh.cells2nodes[C,6] ];


		if (testMesh.mesh_connectivity[C,3] == 4)
	  
			T4 = testMesh.mesh_connectivity[C,7];
			side[4] = testMesh.cell_edges_length[C,4];
			nx[4] = testMesh.cell_edges_Nx[C,4];
			ny[4] = testMesh.cell_edges_Nx[C,4];


			phiLeftX[4] =  gradX[ testMesh.cells2nodes[C,7] ];
			phiLeftY[4] =  gradY[ testMesh.cells2nodes[C,7] ];
	
			phiRightX[4] = gradX[ testMesh.cells2nodes[C,8] ];
			phiRightY[4] = gradY[ testMesh.cells2nodes[C,8] ];

		
		end


		#phiFaceY[T] = 0.5*(phiLeftY + phiRightY)*-ny*side;
        #phiFaceX[T] = 0.5*(phiLeftX + phiRightX)*-nx*side;

		phiFaceX[1] = 0.5*(phiLeftX[1] + phiRightX[1])*-nx[1]*side[1];		
		phiFaceY[1] = 0.5*(phiLeftY[1] + phiRightY[1])*-ny[1]*side[1];

		phiFaceX[2] = 0.5*(phiLeftX[2] + phiRightX[2])*-nx[2]*side[2];		
		phiFaceY[2] = 0.5*(phiLeftY[2] + phiRightY[2])*-ny[2]*side[2];
	  
		phiFaceX[3] = 0.5*(phiLeftX[3] + phiRightX[3])*-nx[3]*side[3];		
		phiFaceY[3] = 0.5*(phiLeftY[3] + phiRightY[3])*-ny[3]*side[3];
	  
		phiFaceX[4] = 0.5*(phiLeftX[4] + phiRightX[4])*-nx[4]*side[4];		
		phiFaceY[4] = 0.5*(phiLeftY[4] + phiRightY[4])*-ny[4]*side[4]; 	


		##areaCell::Float64 = testMesh.cell_areas[C]; 
		##divX::Float64 = (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4])/testMesh.cell_areas[C];
        ##divY::Float64 = (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4])/testMesh.cell_areas[C];


		divergence[C] = (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4])/testMesh.cell_areas[C] + (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4])/testMesh.cell_areas[C];


  end ## end of cells

  

end ## end of function


@everywhere function nodesDivergenceReconstructionFastSA( beginCell::Int32, endCell::Int32, testMesh::mesh2d_Int32, 
  gradX::Array{Float64,1}, gradY::Array{Float64,1}, divergence::Array{Float64,1})
  
  
 

  for C = beginCell:endCell
  
		phiLeftX = zeros(Float64,4);
		phiLeftY = zeros(Float64,4);

		phiRightX = zeros(Float64,4);
		phiRightY = zeros(Float64,4);

		phiFaceX = zeros(Float64,4);
		phiFaceY  = zeros(Float64,4);

		side = zeros(Float64,4);
		nx = zeros(Float64,4);
		ny = zeros(Float64,4);

		numNodesInCell::Int32 = testMesh.mesh_connectivity[C,3]; ## CMatrix mesh_connectivity - first index == 1
		
		
		T1::Int32 = testMesh.mesh_connectivity[C,4];
		T2::Int32 = testMesh.mesh_connectivity[C,5];
		T3::Int32 = testMesh.mesh_connectivity[C,6];
	  
		side[1] = testMesh.cell_edges_length[C,1];
		side[2] = testMesh.cell_edges_length[C,2];
		side[3] = testMesh.cell_edges_length[C,3];
	  
		nx[1] = testMesh.cell_edges_Nx[C,1];
		nx[2] = testMesh.cell_edges_Nx[C,2];
		nx[3] = testMesh.cell_edges_Nx[C,3];
	  
		ny[1] = testMesh.cell_edges_Nx[C,1];
		ny[2] = testMesh.cell_edges_Nx[C,2];
		ny[3] = testMesh.cell_edges_Nx[C,3];
		

		phiLeftX[1] =  gradX[ testMesh.cells2nodes[C,1] ];
		phiLeftY[1] =  gradY[ testMesh.cells2nodes[C,1] ];
	
		phiRightX[1] = gradX[ testMesh.cells2nodes[C,2] ];
		phiRightY[1] = gradY[ testMesh.cells2nodes[C,2] ];

		phiLeftX[2] =  gradX[ testMesh.cells2nodes[C,3] ];
		phiLeftY[2] =  gradY[ testMesh.cells2nodes[C,3] ];
	
		phiRightX[2] = gradX[ testMesh.cells2nodes[C,4] ];
		phiRightY[2] = gradY[ testMesh.cells2nodes[C,4] ];
	
		phiLeftX[3] =  gradX[ testMesh.cells2nodes[C,5] ];
		phiLeftY[3] =  gradY[ testMesh.cells2nodes[C,5] ];
	
		phiRightX[3] = gradX[ testMesh.cells2nodes[C,6] ];
		phiRightY[3] = gradY[ testMesh.cells2nodes[C,6] ];


		if (numNodesInCell == 4)
	  
			T4::Int32 = testMesh.mesh_connectivity[C,7];
			side[4] = testMesh.cell_edges_length[C,4];
			nx[4] = testMesh.cell_edges_Nx[C,4];
			ny[4] = testMesh.cell_edges_Nx[C,4];


			phiLeftX[4] =  gradX[ testMesh.cells2nodes[C,7] ];
			phiLeftY[4] =  gradY[ testMesh.cells2nodes[C,7] ];
	
			phiRightX[4] = gradX[ testMesh.cells2nodes[C,8] ];
			phiRightY[4] = gradY[ testMesh.cells2nodes[C,8] ];

		
		end


		#phiFaceY[T] = 0.5*(phiLeftY + phiRightY)*-ny*side;
        #phiFaceX[T] = 0.5*(phiLeftX + phiRightX)*-nx*side;

		phiFaceX[1] = 0.5*(phiLeftX[1] + phiRightX[1])*-nx[1]*side[1];		
		phiFaceY[1] = 0.5*(phiLeftY[1] + phiRightY[1])*-ny[1]*side[1];

		phiFaceX[2] = 0.5*(phiLeftX[2] + phiRightX[2])*-nx[2]*side[2];		
		phiFaceY[2] = 0.5*(phiLeftY[2] + phiRightY[2])*-ny[2]*side[2];
	  
		phiFaceX[3] = 0.5*(phiLeftX[3] + phiRightX[3])*-nx[3]*side[3];		
		phiFaceY[3] = 0.5*(phiLeftY[3] + phiRightY[3])*-ny[3]*side[3];
	  
		phiFaceX[4] = 0.5*(phiLeftX[4] + phiRightX[4])*-nx[4]*side[4];		
		phiFaceY[4] = 0.5*(phiLeftY[4] + phiRightY[4])*-ny[4]*side[4]; 	


		areaCell::Float64 = testMesh.cell_areas[C]; 

		divX::Float64 = (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4])/areaCell;
        divY::Float64 = (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4])/areaCell;


		divergence[C] = divX + divY;


  end ## end of cells

  

end ## end of function
