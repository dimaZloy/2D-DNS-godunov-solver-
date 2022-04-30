





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
	
	# dUdxNodes = zeros(Float64,nNodes);
	# dUdyNodes = zeros(Float64,nNodes);
	
	# dVdxNodes = zeros(Float64,nNodes);
	# dVdyNodes = zeros(Float64,nNodes);
	
	laplasUCuCells = zeros(Float64,nCells);
	laplasUCvCells = zeros(Float64,nCells);
	laplasUCeCells = zeros(Float64,nCells);
	
	cdUdxCells = zeros(Float64,nCells);
	cdUdyCells = zeros(Float64,nCells);
	cdVdxCells = zeros(Float64,nCells);
	cdVdyCells = zeros(Float64,nCells);
	cdEdxCells = zeros(Float64,nCells);
	cdEdyCells = zeros(Float64,nCells);
	
	cdUdxNodes = zeros(Float64,nNodes);
	cdUdyNodes = zeros(Float64,nNodes);
	cdVdxNodes = zeros(Float64,nNodes);
	cdVdyNodes = zeros(Float64,nNodes);
	cdEdxNodes = zeros(Float64,nNodes);
	cdEdyNodes = zeros(Float64,nNodes);
	
	
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
		cdEdyCells,
		cdUdxNodes,
		cdUdyNodes,
		cdVdxNodes,
		cdVdyNodes,
		cdEdxNodes,
		cdEdyNodes
		
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
	temperatureCells = zeros(Float64,testMesh.nCells); 
	gammaCells = zeros(Float64,testMesh.nCells); 
	kCells= zeros(Float64,testMesh.nCells); 
	cflCells= zeros(Float64,testMesh.nCells); 
	
	densityNodes  = zeros(Float64,testMesh.nNodes); 
	UxNodes       = zeros(Float64,testMesh.nNodes); 
	UyNodes       = zeros(Float64,testMesh.nNodes); 
	pressureNodes = zeros(Float64,testMesh.nNodes); 

	for i=1:testMesh.nCells

		densityCells[i] 	= 0.6380;
		UxCells[i] 			= 712.145;
		UyCells[i] 			= 0.0; 
		pressureCells[i] 	= 50000.0;
		temperatureCells[i] 	= pressureCells[i]/densityCells[i]/ thermo.RGAS;
		kCells[i] 	= thermo.kFromT(temperatureCells[i]);
		gammaCells[i] = thermo.gammaFromT(temperatureCells[i]);
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
		temperatureCells,
		gammaCells,
		kCells,
		cflCells, 
		densityNodes,
		UxNodes,
		UyNodes,
		pressureNodes
		#UconsCellsOld,
		#UconsCellsNew
	);

	return testFields2d; 


end

