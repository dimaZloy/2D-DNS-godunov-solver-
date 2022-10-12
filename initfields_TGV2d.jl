

@inline function computeTGV2d(x::Float64, y::Float64, time::Float64, thermo::THERMOPHYSICS, primitiveVars::Vector{Float64})

	P::Float64 = 100000.0;
	U::Float64 = 40.0;
	L::Float64 = 1.0;
	mu::Float64 = 1e-2;
	T::Float64 = P/thermo.RGAS;


	F::Float64 = exp(-2.0*mu*time/L/L);
	Fp::Float64 = exp(-4.0*mu*time/L/L);

	primitiveVars[1] = P/thermo.RGAS/T;
	primitiveVars[2] = U*sin(x/L)*cos(y/L)*F;
	primitiveVars[3] = -U*cos(x/L)*sin(y/L)*F;
	#primitiveVars[4] = P/(thermo.Gamma-1.0) + 0.25*( cos(2*x/L)+cos(2*y/L) )*Fp;
	primitiveVars[4] = P + 0.25*U*U*( cos(2*x/L)+cos(2*y/L) )*Fp;
 
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

	dummy4 = zeros(Float64,4);

	for i=1:testMesh.nCells

		x::Float64 = testMesh.cell_mid_points[i,1];
		y::Float64 = testMesh.cell_mid_points[i,2];

		computeTGV2d(x, y, 0.0, thermo, dummy4);

		densityCells[i] 	= dummy4[1];
		UxCells[i] 			= dummy4[2];
		UyCells[i] 			= dummy4[3]; 
		pressureCells[i] 	= dummy4[4];
		
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

