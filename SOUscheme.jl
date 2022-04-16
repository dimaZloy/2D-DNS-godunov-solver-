

@everywhere function SecondOrderUpwindM2(
	beginCell::Int32,endCell::Int32, bettaKJ::Float64, dt::Float64, flowTime::Float64, 
	testMesh::mesh2d_Int32, testFields::fields2d, thermo::THERMOPHYSICS, 
	UConsCellsOld::Array{Float64,2}, FLUXES::Array{Float64,2}, UconsDiffTerm::Array{Float64,2}, UconsCellsNew::Array{Float64,2})

	#nCells = size(UConsCellsOld,1);
	
	uLeftp = zeros(Float64,4);
	edge_flux1 = zeros(Float64,4);
	edge_flux2 = zeros(Float64,4);
	edge_flux3 = zeros(Float64,4);
	edge_flux4 = zeros(Float64,4);
	
	uUpp = zeros(Float64,4);
	uDownp = zeros(Float64,4);
	uRightp = zeros(Float64,4);
	
	##dummy = zeros(Float64,4);
	
	
	for i = beginCell:endCell
    
		##num_nodes::Int64 = testMesh.mesh_connectivity[i,3];
		
		
		uLeftp[1] = testFields.densityCells[i];
		uLeftp[2] = testFields.UxCells[i];
		uLeftp[3] = testFields.UyCells[i];
		uLeftp[4] = testFields.pressureCells[i];
		
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

		
	   
		
		if (testMesh.mesh_connectivity[i,3] == 3)
		
			#edge_flux1 = ( computeInterfaceSlope(i, Int32(1), testMesh, testFields, thermo, uLeftp, flowTime) );
			#edge_flux2 = ( computeInterfaceSlope(i, Int32(2), testMesh, testFields, thermo, uLeftp, flowTime) );
			#edge_flux3 = ( computeInterfaceSlope(i, Int32(3), testMesh, testFields, thermo, uLeftp, flowTime) );
			
			computeInterfaceSlope(i, Int32(1), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux1) ;
			computeInterfaceSlope(i, Int32(2), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux2) ;
			computeInterfaceSlope(i, Int32(3), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux3) ;
	
			
					
			FLUXES[i,1] = edge_flux1[1] + edge_flux2[1] + edge_flux3[1];
			FLUXES[i,2] = edge_flux1[2] + edge_flux2[2] + edge_flux3[2];
			FLUXES[i,3] = edge_flux1[3] + edge_flux2[3] + edge_flux3[3];
			FLUXES[i,4] = edge_flux1[4] + edge_flux2[4] + edge_flux3[4];
			

		elseif (testMesh.mesh_connectivity[i,3] == 4)
			
			# edge_flux1 = ( computeInterfaceSlope(i, Int32(1), testMesh, testFields, uLeftp, flowTime) );
			# edge_flux2 = ( computeInterfaceSlope(i, Int32(2), testMesh, testFields, uLeftp, flowTime) );
			# edge_flux3 = ( computeInterfaceSlope(i, Int32(3), testMesh, testFields, uLeftp, flowTime) );
			# edge_flux4 = ( computeInterfaceSlope(i, Int32(4), testMesh, testFields, uLeftp, flowTime) );
			
			
			computeInterfaceSlope(i, Int32(1), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux1) ;
			computeInterfaceSlope(i, Int32(2), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux2) ;
			computeInterfaceSlope(i, Int32(3), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux3) ;
			computeInterfaceSlope(i, Int32(4), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, edge_flux4) ;
			
			
				
			FLUXES[i,1] = edge_flux1[1] + edge_flux2[1] + edge_flux3[1] + edge_flux4[1];
			FLUXES[i,2] = edge_flux1[2] + edge_flux2[2] + edge_flux3[2] + edge_flux4[2];
			FLUXES[i,3] = edge_flux1[3] + edge_flux2[3] + edge_flux3[3] + edge_flux4[3];
			FLUXES[i,4] = edge_flux1[4] + edge_flux2[4] + edge_flux3[4] + edge_flux4[4];

		else
			
			display("something wrong in flux calculations ... ")
			
		end
		
		
		# UconsCellsNew[i,1] = ( UConsCellsOld[i,1] - FLUXES[i,1]*bettaKJ*dt*testMesh.Z[i] + bettaKJ*dt*UconsDiffTerm[i,1] );
		# UconsCellsNew[i,2] = ( UConsCellsOld[i,2] - FLUXES[i,2]*bettaKJ*dt*testMesh.Z[i] + bettaKJ*dt*UconsDiffTerm[i,2] );
		# UconsCellsNew[i,3] = ( UConsCellsOld[i,3] - FLUXES[i,3]*bettaKJ*dt*testMesh.Z[i] + bettaKJ*dt*UconsDiffTerm[i,3] );
		# UconsCellsNew[i,4] = ( UConsCellsOld[i,4] - FLUXES[i,4]*bettaKJ*dt*testMesh.Z[i] + bettaKJ*dt*UconsDiffTerm[i,4] );
      
		Rarea::Float64 = 1.0/testMesh.cell_areas[i];
	  
  		UconsCellsNew[i,1] = ( UConsCellsOld[i,1] - FLUXES[i,1]*bettaKJ*dt*Rarea + bettaKJ*dt*UconsDiffTerm[i,1] );
		UconsCellsNew[i,2] = ( UConsCellsOld[i,2] - FLUXES[i,2]*bettaKJ*dt*Rarea + bettaKJ*dt*UconsDiffTerm[i,2] );
		UconsCellsNew[i,3] = ( UConsCellsOld[i,3] - FLUXES[i,3]*bettaKJ*dt*Rarea + bettaKJ*dt*UconsDiffTerm[i,3] );
		UconsCellsNew[i,4] = ( UConsCellsOld[i,4] - FLUXES[i,4]*bettaKJ*dt*Rarea + bettaKJ*dt*UconsDiffTerm[i,4] );

	  
   
	end # i - loop for all cells


end

	


