

function loadPrevResults(testMesh::mesh2d_Int32, thermo::THERMOPHYSICS, filename::String, dynControls::DYNAMICCONTROLS, testfields2dX::fields2d )

	
	
	println("try to read previous solution from ", filename);

	@load filename solInst;



	for i=1:testMesh.nCells

	
		testfields2dX.densityCells[i] 	=  solInst.densityCells[i];
		testfields2dX.UxCells[i] 			=  solInst.UxCells[i];
		testfields2dX.UyCells[i] 			=  solInst.UyCells[i]; 
		testfields2dX.pressureCells[i] 	=  solInst.pressureCells[i];
		
		testfields2dX.aSoundCells[i] = sqrt( thermo.Gamma * solInst.pressureCells[i]/solInst.densityCells[i] );
		testfields2dX.VMAXCells[i]  = sqrt( solInst.UxCells[i]*solInst.UxCells[i] + solInst.UyCells[i]*solInst.UyCells[i] ) + testfields2dX.aSoundCells[i];
				
	end


	

	dynControls.flowTime = solInst.flowTime;
	
	#tmp = split(filename,"zzz");
	#num::Int64 = parse(Int64,tmp[2]); 
	#dynControls.curIter = num - 1000;

    println(" ... done  ");


end
