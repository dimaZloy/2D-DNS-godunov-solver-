
function createViscousFields2d(nCells::Int64, nNodes::Int64)::viscousFields2d

	viscosityCells = zeros(Float64,nCells);
	viscosityNodes = zeros(Float64,nNodes);
	
	#dUdxCells = zeros(Float64,nCells);
	#dUdyCells = zeros(Float64,nCells);
	
	#dVdxCells = zeros(Float64,nCells);
	#dVdyCells = zeros(Float64,nCells);
	
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
		viscosityCells,
		viscosityNodes,
		#dUdxCells,
		#dUdyCells,
		#dVdxCells,
		#dVdyCells,
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

