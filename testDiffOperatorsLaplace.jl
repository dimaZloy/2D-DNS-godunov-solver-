
@inline function ULtest(x::Float64, y::Float64)::Float64
    return 1.0/3.0*(x*x*x*x + y*y*y*y);
end

@inline function calcLaplaceUTest(x::Float64, y::Float64)::Float64
    return 4.0*x*x + 4.0*y*y;
end

function testCalcLaplacian(meshname::String)

    testMesh = readMesh2dHDF5(meshname);

    cellsThreads = distibuteCellsInThreadsSA(Threads.nthreads(), testMesh.nCells); ## partition mesh 
	nodesThreads = distibuteNodesInThreadsSA(Threads.nthreads(), testMesh.nNodes); ## partition mesh 

	NCells::Int64 = testMesh.nCells;
    NNodes::Int64 = testMesh.nNodes;

    U = zeros(Float64,NCells);
    ULap = zeros(Float64,NCells);
    ULapApprox = zeros(Float64,NCells);

    UNodes = zeros(Float64,NNodes);
    ULapNodes = zeros(Float64,NNodes);
    ULapApproxNodes = zeros(Float64,NNodes);
    
    
    UGradXApprox = zeros(Float64,NCells);
    UGradYApprox = zeros(Float64,NCells);
    UGradXApproxNodes = zeros(Float64,NNodes);
    UGradYApproxNodes = zeros(Float64,NNodes);
    
    


    for i = 1:NCells
        U[i] = ULtest(testMesh.cell_mid_points[i,1],testMesh.cell_mid_points[i,2]);
        ULap[i] = calcLaplaceUTest(testMesh.cell_mid_points[i,1],testMesh.cell_mid_points[i,2])
    end

    #display(U.-UL)

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, U, UNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, ULap, ULapNodes);


    #display(UNodes.-ULNodes)

    
    
   
    calcScalarFieldGradient(cellsThreads, testMesh, UNodes, UGradXApprox, UGradYApprox)
    
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradXApprox, UGradXApproxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradYApprox, UGradYApproxNodes);
        
    #nodesDivergenceReconstructionFastSA44(cellsThreads, testMesh, U, UGradXApproxNodes, UGradYApproxNodes, ULapApprox);
    nodesDivergenceReconstructionFastSA22(cellsThreads, testMesh, UGradXApproxNodes, UGradYApproxNodes, ULapApprox);

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, ULapApprox, ULapApproxNodes);

    #display(ULNodes.-ULApproxNodes)

    aMax = findmax(ULapNodes);
    aMin = findmin(ULapNodes);

    nMax = findmax(ULapApproxNodes);
    nMin = findmin(ULapApproxNodes);


    saveResults2VTK("zzzLap1", testMesh, ULapNodes, "Lap theory" )
    saveResults2VTK("zzzLap2", testMesh, ULapApproxNodes, "Lap approx" )

    display(aMax)
    display(aMin)
    display(nMax)
    display(nMin)


    figure(3)
    clf()
    subplot(1,3,1)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UNodes);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("U2");
	axis("equal");
    subplot(1,3,2)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, ULapNodes, vmin = aMin[1], vmax=aMax[1]);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("Laplace theory");
	axis("equal");
    subplot(1,3,3)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, ULapApproxNodes,vmin = aMin[1], vmax=aMax[1]);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("Laplace approx");
	axis("equal");



end
