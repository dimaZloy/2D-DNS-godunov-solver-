

@inline function UGtest(x::Float64, y::Float64)::Float64
    return x*x*y*y*y;
end

@inline function calcGradientUTest(nCells::Int64, x::Vector{Float64}, y::Vector{Float64}, gradX::Vector{Float64}, gradY::Vector{Float64})
    
    for i = 1:nCells
        gradX[i] = 2.0*x[i]*y[i]*y[i]*y[i];
        gradY[i] = 3.0*x[i]*x[i]*y[i]*y[i];
    end

end

@inline function calcGradienMagUTest(nCells::Int64, gradX::Vector{Float64}, gradY::Vector{Float64},gradMag::Vector{Float64} )
    
    for i = 1:nCells
        gradMag[i] = sqrt(gradX[i]*gradX[i] + gradY[i]*gradY[i]);
    end

end


function calcScalarFieldGradient(cellsThreads::Array{Int32,2}, testMesh::mesh2d_Int32, 
    scalarField::Array{Float64,1}, gradX::Array{Float64,1}, gradY::Array{Float64,1})
  
    ## scalarField in Mesh Nodes !!!!
  
    Threads.@threads for p in 1:Threads.nthreads()
           

                phiFaceX = zeros(Float64,4);
                phiFaceY = zeros(Float64,4);
                
                phiLeft = zeros(Float64,4);
                phiRight = zeros(Float64,4);
                
                side = zeros(Float64,4);
                nx = zeros(Float64,4);
                ny = zeros(Float64,4);
                
            
                for C =  cellsThreads[p,1]:cellsThreads[p,2]
                
                    
                    side[1] = testMesh.cell_edges_length[C,1];
                    side[2] = testMesh.cell_edges_length[C,2];
                    side[3] = testMesh.cell_edges_length[C,3];
                    
                    nx[1] = testMesh.cell_edges_Nx[C,1];
                    nx[2] = testMesh.cell_edges_Nx[C,2];
                    nx[3] = testMesh.cell_edges_Nx[C,3];
                    
                    ny[1] = testMesh.cell_edges_Ny[C,1];
                    ny[2] = testMesh.cell_edges_Ny[C,2];
                    ny[3] = testMesh.cell_edges_Ny[C,3];
                    
            
                    phiLeft[1] =  scalarField[ testMesh.cells2nodes[C,1] ];
                    phiRight[1] = scalarField[ testMesh.cells2nodes[C,2] ];
            
                    phiLeft[2] =  scalarField[ testMesh.cells2nodes[C,3] ];
                    phiRight[2] = scalarField[ testMesh.cells2nodes[C,4] ];
            
                    phiLeft[3] =  scalarField[ testMesh.cells2nodes[C,5] ];
                    phiRight[3] = scalarField[ testMesh.cells2nodes[C,6] ];
                    
            
                    if (testMesh.mesh_connectivity[C,3] == 4) ## if number of node cells == 4 
                    
                        side[4] = testMesh.cell_edges_length[C,4];
                        nx[4] = testMesh.cell_edges_Nx[C,4];
                        ny[4] = testMesh.cell_edges_Ny[C,4];
                
                        phiLeft[4] =  scalarField[ testMesh.cells2nodes[C,7] ];
                        phiRight[4] = scalarField[ testMesh.cells2nodes[C,8] ];
                    
                    end
                    
                    
                    phiFaceX[1] = 0.5*(phiLeft[1] + phiRight[1])*-nx[1]*side[1];		
                    phiFaceY[1] = 0.5*(phiLeft[1] + phiRight[1])*-ny[1]*side[1];
            
                    phiFaceX[2] = 0.5*(phiLeft[2] + phiRight[2])*-nx[2]*side[2];		
                    phiFaceY[2] = 0.5*(phiLeft[2] + phiRight[2])*-ny[2]*side[2];
                    
                    phiFaceX[3] = 0.5*(phiLeft[3] + phiRight[3])*-nx[3]*side[3];		
                    phiFaceY[3] = 0.5*(phiLeft[3] + phiRight[3])*-ny[3]*side[3];
                    
                    phiFaceX[4] = 0.5*(phiLeft[4] + phiRight[4])*-nx[4]*side[4];		
                    phiFaceY[4] = 0.5*(phiLeft[4] + phiRight[4])*-ny[4]*side[4];
            
                    gradX[C] =  (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4]) / testMesh.cell_areas[C];
                    gradY[C] =  (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4]) / testMesh.cell_areas[C];
                    
        
            
                end ## end global loop for cells
  
            end ## threads
  
  end




function testCalcGradients(meshname::String)

    testMesh = readMesh2dHDF5(meshname);

    cellsThreads = distibuteCellsInThreadsSA(Threads.nthreads(), testMesh.nCells); ## partition mesh 
	nodesThreads = distibuteNodesInThreadsSA(Threads.nthreads(), testMesh.nNodes); ## partition mesh 

	NCells::Int64 = testMesh.nCells;
    NNodes::Int64 = testMesh.nNodes;

    U = zeros(Float64,NCells);
    UNodes = zeros(Float64,NNodes);
    
    UGradX = zeros(Float64,NCells);
    UGradY = zeros(Float64,NCells);
    UGradMag = zeros(Float64,NCells);

    UGradXApprox = zeros(Float64,NCells);
    UGradYApprox = zeros(Float64,NCells);
    UGradMagApprox = zeros(Float64,NCells);

    UGradXNodes = zeros(Float64,NNodes);
    UGradYNodes = zeros(Float64,NNodes);
    UGradXApproxNodes = zeros(Float64,NNodes);
    UGradYApproxNodes = zeros(Float64,NNodes);
    
    UGradMagNodes = zeros(Float64,NNodes);
    UGradMagApproxNodes = zeros(Float64,NNodes);



    for i = 1:NCells

        U[i] = UGtest(testMesh.cell_mid_points[i,1],testMesh.cell_mid_points[i,2]);
        
    end

    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, U, UNodes);
    
    calcGradientUTest(NCells, testMesh.cell_mid_points[:,1],testMesh.cell_mid_points[:,2], UGradX, UGradY);
    calcGradienMagUTest(NCells, UGradX, UGradY, UGradMag);


    # Threads.@threads for p in 1:Threads.nthreads()    
    #      nodesGradientReconstructionFastPerThread22(cellsThreads[p,1],cellsThreads[p,2], testMesh, UNodes, UGradXApprox, UGradYApprox)
    # end

    calcScalarFieldGradient(cellsThreads,testMesh, UNodes, UGradXApprox, UGradYApprox);


    for i = 1: NCells
        UGradMagApprox[i] =     sqrt( UGradXApprox[i]*UGradXApprox[i] +  UGradYApprox[i]*UGradYApprox[i] )
    end



   
    #cells2nodesSolutionReconstructionWithStencilsVector( testMesh, UL, ULnodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradX, UGradXNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradY, UGradYNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradMag, UGradMagNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradXApprox, UGradXApproxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradYApprox, UGradYApproxNodes);
    cells2nodesSolutionReconstructionWithStencilsVector( nodesThreads, testMesh, UGradMagApprox, UGradMagApproxNodes);

    saveResults2VTK(meshname*"GradTheory", testMesh, UGradMagNodes, "grad theory" )
    saveResults2VTK(meshname*"GradApprox", testMesh, UGradMagApproxNodes, "grad approx" )


    figure(1)
    clf()
    subplot(1,3,1)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UNodes);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("U2");
	axis("equal");
    subplot(1,3,2)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UGradMagNodes);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("Mag(GragU2) theory");
	axis("equal");
    subplot(1,3,3)
    tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, UGradMagApproxNodes);	
	set_cmap("jet");
	xlabel("x");
	ylabel("y");
	title("Mag(GragU2) approx");
	axis("equal");



end
