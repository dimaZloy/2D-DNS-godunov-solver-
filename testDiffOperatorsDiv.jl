

function testCalcDivergence(meshname::String)

    testMesh = readMesh2dHDF5(meshname);

    cellsThreads = distibuteCellsInThreadsSA(Threads.nthreads(), testMesh.nCells); ## partition mesh 
	nodesThreads = distibuteNodesInThreadsSA(Threads.nthreads(), testMesh.nNodes); ## partition mesh 

	NCells::Int64 = testMesh.nCells;
    NNodes::Int64 = testMesh.nNodes;

    Ux = zeros(Float64,NCells);
    Uy = zeros(Float64,NCells);
    dUxdx = zeros(Float64,NCells);
    dUydy = zeros(Float64,NCells);


    UDiv = zeros(Float64,NCells);
    UDivXApprox = zeros(Float64,NCells);
    UDivYApprox = zeros(Float64,NCells);
    UDivApprox = zeros(Float64,NCells);

    UxNodes = zeros(Float64,NNodes);
    UyNodes = zeros(Float64,NNodes);
    UDivNodes = zeros(Float64,NNodes);
    UDivApproxNodes = zeros(Float64,NNodes);

    dUxdxNodes = zeros(Float64,NNodes);
    dUydyNodes = zeros(Float64,NNodes);


    UxGradXApprox = zeros(Float64,NCells);
    UxGradYApprox = zeros(Float64,NCells);
    UyGradXApprox = zeros(Float64,NCells);
    UyGradYApprox = zeros(Float64,NCells);


    UxGradXApproxNodes = zeros(Float64,NNodes);
    UxGradYApproxNodes = zeros(Float64,NNodes);
    UyGradXApproxNodes = zeros(Float64,NNodes);
    UyGradYApproxNodes = zeros(Float64,NNodes);
    
    
    #UDtest(NCells, testMesh.cell_mid_points[:,1],testMesh.cell_mid_points[:,2], Ux, Uy); 

    for i = 1:NCells
        Ux[i] = cos(testMesh.cell_mid_points[i,1] + 2.0*testMesh.cell_mid_points[i,2]);
        Uy[i] = sin(testMesh.cell_mid_points[i,1] - 2.0*testMesh.cell_mid_points[i,2]);
        dUxdx[i] = -2.0*cos(testMesh.cell_mid_points[i,1] - 2.0*testMesh.cell_mid_points[i,2]);
        dUydy[i] = -sin(testMesh.cell_mid_points[i,1] + 2.0*testMesh.cell_mid_points[i,2]);
        UDiv[i] = dUxdx[i] + dUydy[i]
    end


    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, Ux, UxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, Uy, UyNodes);

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, dUxdx, dUxdxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, dUydy, dUydyNodes);

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UDiv, UDivNodes);
    



     calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, Ux, UxNodes, UyGradXApprox, UyGradYApprox);
     calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, Uy, UyNodes, UxGradXApprox, UxGradYApprox);
    
     cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UxGradXApprox, UxGradXApproxNodes);
     cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UyGradYApprox, UyGradYApproxNodes);
    
    
       for i = 1:NCells
             UDivApprox[i] = (UxGradXApprox[i] + UyGradYApprox[i] );
       end


    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UDivApprox, UDivApproxNodes);

    saveResults2VTK(meshname*"DivTheory", testMesh, UDivNodes, "div theory" )
    saveResults2VTK(meshname*"DivApprox", testMesh, UDivApproxNodes, "div approx" )

    #display(ULNodes.-ULApproxNodes)

    adUxMax = findmax(dUxdxNodes);
    adUxMin = findmin(dUxdxNodes);

    adUyMax = findmax(dUydyNodes);
    adUyMin = findmin(dUydyNodes);

    display(adUxMax)
    display(adUxMin)
    display(adUyMax)
    display(adUyMin)

    aMax = findmax(UDivNodes);
    aMin = findmin(UDivNodes);

    nMax = findmax(UDivApproxNodes);
    nMin = findmin(UDivApproxNodes);


    display(aMax)
    display(aMin)
    display(nMax)
    display(nMin)

    # figure(1)
    # clf()
    # subplot(2,2,1)
    # tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UxNodes);	
	# set_cmap("jet");
	# xlabel("x");
	# ylabel("y");
	# title("Ux theory");

    # subplot(2,2,2)
    # tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UyNodes);	
	# set_cmap("jet");
	# xlabel("x");
	# ylabel("y");
	# title("Uy theory");

    figure(2)
    clf()
    subplot(2,2,1)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, dUxdxNodes, vmin = adUxMin[1], vmax=adUxMax[1]);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("dUxdx theory");

    subplot(2,2,2)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, dUydyNodes,vmin = adUyMin[1], vmax=adUyMax[1]);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("dUydy theory");

    subplot(2,2,3)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UxGradXApproxNodes,vmin = adUxMin[1], vmax=adUxMax[1]  );	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("dUxdx approx");

    subplot(2,2,4)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UyGradYApproxNodes,vmin = adUyMin[1], vmax=adUyMax[1] );	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("dUydy approx");

     figure(3)
     clf()
      subplot(1,2,1)
      tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UDivNodes, vmin = -3.0, vmax=3.0);	
	  set_cmap("jet");
	  xlabel("x");
	  ylabel("y");
	  title("Divergence theory");
	  axis("equal");
      subplot(1,2,2)
      tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UDivApproxNodes, vmin = -3.0, vmax=3.0);	
	  set_cmap("jet");
	  xlabel("x");
	  ylabel("y");
	  title("Divergence approx");
	  axis("equal");


end
