
@inline function ULtest(x::Float64, y::Float64)::Float64
    return 1.0/3.0*(x*x*x*x + y*y*y*y);
end

@inline function calcLaplaceUTest(x::Float64, y::Float64)::Float64
    return 4.0*x*x + 4.0*y*y;
end


# function calcScalarFieldLaplacian(cellsThreads::Array{Int32,2}, test_mesh::mesh2d_Int32, 
#     scalarFieldCells::Vector{Float64}, scalarFieldNodes::Vector{Float64},  LaplaceCell::Vector{Float64})
  
#     ## scalarField in Mesh Nodes !!!!
  
#     Threads.@threads for p in 1:Threads.nthreads()
           

#                 phiFaceX = zeros(Float64,4);
#                 phiFaceY = zeros(Float64,4);
                
#                 phiLeft = zeros(Float64,4);
#                 phiRight = zeros(Float64,4);
                
#                 side = zeros(Float64,4);
#                 nx = zeros(Float64,4);
#                 ny = zeros(Float64,4);

#                 #phiCell::Float64 = 0.0;

                
            
#                 for C =  cellsThreads[p,1]:cellsThreads[p,2]
                
#                     phi0 = scalarFieldCells[C];
                     

#                     if (test_mesh.mesh_connectivity[C,3] == 4) ## if number of node cells == 4 
                        
#                         phi1 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,4] ];
#                         phi2 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,5] ];
#                         phi3 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,6] ];
#                         phi4 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,7] ];

#                         LaplaceCell[C] = 3.0/4.0*(phi1 + phi2 + phi3 + phi4 - 4.0*phi0)/test_mesh.HX[C]/test_mesh.HX[C];
#                         ## works fine for quad grids!!!!
#                     else
#                         phi1 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,4] ];
#                         phi2 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,5] ];
#                         phi3 = scalarFieldNodes[ test_mesh.mesh_connectivity[C,6] ];
#                         LaplaceCell[C] = (phi1 + phi2 + phi3  - 3.0*phi0)/sqrt(test_mesh.cell_areas[C]);
#                         ## doesn't work at all !!!!
#                     end
            
#                 end ## end global loop for cells
  
#             end ## threads
  
#   end



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

    ddUyGradXApprox = zeros(Float64,NCells);
    ddUyGradYApprox = zeros(Float64,NCells);
    ddUxGradXApprox = zeros(Float64,NCells);
    ddUxGradYApprox = zeros(Float64,NCells);
    

    UGradXApproxNodes = zeros(Float64,NNodes);
    UGradYApproxNodes = zeros(Float64,NNodes);
    

    for i = 1:NCells
        U[i] = ULtest(testMesh.cell_mid_points[i,1],testMesh.cell_mid_points[i,2]);
        ULap[i] = calcLaplaceUTest(testMesh.cell_mid_points[i,1],testMesh.cell_mid_points[i,2])
    end


    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, U, UNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, ULap, ULapNodes);


    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, U, UNodes, UGradXApprox, UGradYApprox);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradXApprox, UGradXApproxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradYApprox, UGradYApproxNodes);

    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UGradXApprox, UGradXApproxNodes, ddUxGradXApprox, ddUxGradYApprox);
    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, UGradYApprox, UGradYApproxNodes, ddUyGradXApprox, ddUyGradYApprox);
   
   
    for i = 1:NCells
        ULapApprox[i] = (ddUxGradXApprox[i] + ddUyGradYApprox[i] );
    end

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, ULapApprox, ULapApproxNodes);

    

    aMax = findmax(ULapNodes);
    aMin = findmin(ULapNodes);

    nMax = findmax(ULapApproxNodes);
    nMin = findmin(ULapApproxNodes);


    saveResults2VTK(meshname*"LapTheory", testMesh, ULapNodes, "Lap theory" )
    saveResults2VTK(meshname*"LapApprox", testMesh, ULapApproxNodes, "Lap approx" )

    display(aMax)
    display(aMin)
    display(nMax)
    display(nMin)


    figure(4)
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
