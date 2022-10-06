

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


@inline function calcPhiInCellCluster(Node::Int32, Cell::Int32, testMesh::mesh2d_Int32, scalarFieldCells::Vector{Float64})::Float64

    phi::Float64 = 0.0;
    counter::Int64 = 0;
    for i = 1:testMesh.nNeibCells
        cell::Int32 = testMesh.cell_clusters[Node,i];
        if cell > 0 && cell <= testMesh.nCells #&& cell != Cell
            phi +=  scalarFieldCells[cell];
            counter += 1;
        end
    end

    #return phi/counter;
    return counter!=0 ? phi/counter : 0.0; 

end 


@inline function  getSymmetricalPointAboutCellEdge(
        px::Float64,py::Float64, 
        x0::Float64, y0::Float64, 
        x1::Float64, y1::Float64, 
        sp::Vector{Float64})

    dx::Float64 = x1-x0;
    dy::Float64 = y1-y0;
    a::Float64 = (dx*dx - dy*dy)/(dx*dx + dy*dy);
    b::Float64 = 2.0*dx*dy/(dx*dx + dy*dy);
    sp[1] = a*(px-x0) + b*(py-y0) + x0;
    sp[2] = b*(px-x0) - a*(py-y0) + y0;

end

function calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads::Array{Int32,2}, testMesh::mesh2d_Int32, 
    scalarFieldCells::Vector{Float64}, scalarFieldNodes::Vector{Float64}, gradX::Vector{Float64}, gradY::Vector{Float64})
  
    ## calculate X and Y gradients of scalarField in cell-centers
    ## use Green-Gauss noded-based method for interior cells and
    ## unweighted LSQ method for boundary cells (test scalar function (BC) shall be provided)
    ## based on the paper by 
    ## Anderson, W.K. and Bonhaus, D.L, 
    ## An implicit upwind algorithm for computing turbulent flows on unstructured grids,  
    ## Comput Fluids, 23(1), 1-21 (1994)
  
    Threads.@threads for p in 1:Threads.nthreads()
           

                phiFaceX = zeros(Float64,4);
                phiFaceY = zeros(Float64,4);
                
                phiLeft = zeros(Float64,4);
                phiRight = zeros(Float64,4);
                
                side = zeros(Float64,4);
                nx = zeros(Float64,4);
                ny = zeros(Float64,4);
                
                ## looping for interior cells
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
                    
            
                    # phiLeft[1] =  scalarField[ testMesh.cells2nodes[C,1] ];
                    # phiRight[1] = scalarField[ testMesh.cells2nodes[C,2] ];
            
                    # phiLeft[2] =  scalarField[ testMesh.cells2nodes[C,3] ];
                    # phiRight[2] = scalarField[ testMesh.cells2nodes[C,4] ];
            
                    # phiLeft[3] =  scalarField[ testMesh.cells2nodes[C,5] ];
                    # phiRight[3] = scalarField[ testMesh.cells2nodes[C,6] ];
                    
                    phiLeft[1]  = calcPhiInCellCluster(testMesh.cells2nodes[C,1], C, testMesh, scalarFieldCells );
                    phiRight[1] = calcPhiInCellCluster(testMesh.cells2nodes[C,2], C, testMesh, scalarFieldCells );
                    phiLeft[2]  = calcPhiInCellCluster(testMesh.cells2nodes[C,3], C, testMesh, scalarFieldCells );
                    phiRight[2] = calcPhiInCellCluster(testMesh.cells2nodes[C,4], C, testMesh, scalarFieldCells );
                    phiLeft[3]  = calcPhiInCellCluster(testMesh.cells2nodes[C,5], C, testMesh, scalarFieldCells );
                    phiRight[3] = calcPhiInCellCluster(testMesh.cells2nodes[C,6], C, testMesh, scalarFieldCells );
            

                    if (testMesh.mesh_connectivity[C,3] == 4) ## if number of node cells == 4 
                    
                        side[4] = testMesh.cell_edges_length[C,4];
                        nx[4] = testMesh.cell_edges_Nx[C,4];
                        ny[4] = testMesh.cell_edges_Ny[C,4];

                        phiLeft[4]  = calcPhiInCellCluster(testMesh.cells2nodes[C,7], C, testMesh, scalarFieldCells );
                        phiRight[4] = calcPhiInCellCluster(testMesh.cells2nodes[C,8], C, testMesh, scalarFieldCells );
    
                        # phiLeft[4] =  scalarField[ testMesh.cells2nodes[C,7] ];
                        # phiRight[4] = scalarField[ testMesh.cells2nodes[C,8] ];
                       
                    
                       
                    end
                
                    
                    phiFaceX[1] = 0.5*(phiLeft[1] + phiRight[1])*-ny[1]*side[1];		
                    phiFaceY[1] = 0.5*(phiLeft[1] + phiRight[1])*-nx[1]*side[1];
            
                    phiFaceX[2] = 0.5*(phiLeft[2] + phiRight[2])*-ny[2]*side[2];		
                    phiFaceY[2] = 0.5*(phiLeft[2] + phiRight[2])*-nx[2]*side[2];
                    
                    phiFaceX[3] = 0.5*(phiLeft[3] + phiRight[3])*-ny[3]*side[3];		
                    phiFaceY[3] = 0.5*(phiLeft[3] + phiRight[3])*-nx[3]*side[3];
                    
                    phiFaceX[4] = 0.5*(phiLeft[4] + phiRight[4])*-ny[4]*side[4];		
                    phiFaceY[4] = 0.5*(phiLeft[4] + phiRight[4])*-nx[4]*side[4];
            
                    gradX[C] =  (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4]) / testMesh.cell_areas[C];
                    gradY[C] =  (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4]) / testMesh.cell_areas[C];
                    
        
            
                end ## end global loop for cells
  
            end ## threads

            ## looping for boundary cells 

            for B = 1:length(testMesh.bc_indexes)
                BCell = testMesh.bc_data[B,1];
                gradX[BCell] = gradY[BCell] = 0.0;    
                phi0 = scalarFieldCells[BCell];
                x0 = testMesh.cell_mid_points[BCell,1];
                y0 = testMesh.cell_mid_points[BCell,2];   
                xI = zeros(Float64,4);
                yI = zeros(Float64,4);
                phiI = zeros(Float64,4);
                #indexI = ones(Float64,4);
                WiX = zeros(Float64,4);
                WiY = zeros(Float64,4);

                v = 1;
                for neibI = 1:4

                    cellI = testMesh.cell_stiffness[BCell,neibI];    

                    if cellI >=1 && cellI <= testMesh.nCells ## interior cells
                        xI[neibI] =  testMesh.cell_mid_points[cellI,1];  
                        yI[neibI] =  testMesh.cell_mid_points[cellI,2];  
                        phiI[neibI] = scalarFieldCells[cellI];
                        #indexI[neibI] = 1.0;
                    elseif cellI < 0 ## boundary cells
                        ## apply BCs here 
                        
                        # node1 = testMesh.cells2nodes[BCell,v]
                        # node2 = testMesh.cells2nodes[BCell,v+1]
                        # sp = zeros(Float64,2);
                        # getSymmetricalPointAboutCellEdge(x0,y0, testMesh.xNodes[node1], testMesh.yNodes[node1], testMesh.xNodes[node2], testMesh.yNodes[node2], sp);
                        # xI[neibI] = sp[1];
                        # yI[neibI] = sp[2];
                        # phiI[neibI] =  UGtest(xI[neibI],yI[neibI]);

                        #xI[neibI] = x0;
                        #yI[neibI] = y0;
                        #phiI[neibI] =  phi0;


                       
                    end
                    v += 2;

                end

                r11 = sqrt( (xI[1]-x0)*(xI[1]-x0) + (xI[2]-x0)*(xI[2]-x0) + (xI[3]-x0)*(xI[3]-x0) + (xI[4]-x0)*(xI[4]-x0)  ) ;
                r12 = 1.0/r11*( (xI[1]-x0)*(yI[1]-y0) + (xI[2]-x0)*(yI[2]-y0) + (xI[3]-x0)*(yI[3]-y0) + (xI[4]-x0)*(yI[4]-y0) ) ;
                r22 = sqrt( ((yI[1]-y0) - (xI[1]-x0)*r12/r11)^2 + ((yI[2]-y0) - (xI[2]-x0)*r12/r11)^2 + ((yI[3]-y0) - (xI[3]-x0)*r12/r11)^2 + ((yI[4]-y0) - (xI[4]-x0)*r12/r11)^2 ) ;
                
                # WiY[1] = 1.0/r22/r22*( (yI[1]-y0) - (xI[1]-x0)*r12/r11)*indexI[1] ;  
                # WiY[2] = 1.0/r22/r22*( (yI[2]-y0) - (xI[2]-x0)*r12/r11)*indexI[2] ;  
                # WiY[3] = 1.0/r22/r22*( (yI[3]-y0) - (xI[3]-x0)*r12/r11)*indexI[3] ;  
                # WiY[4] = 1.0/r22/r22*( (yI[4]-y0) - (xI[4]-x0)*r12/r11)*indexI[4] ;  

                # WiX[1] = (1.0/r11/r11*(xI[1]-x0) - r12/r11/r22/r22*( (yI[1] -y0) - (xI[1]-x0)*r12/r11 ))*indexI[1];
                # WiX[2] = (1.0/r11/r11*(xI[2]-x0) - r12/r11/r22/r22*( (yI[2] -y0) - (xI[2]-x0)*r12/r11 ))*indexI[2];
                # WiX[3] = (1.0/r11/r11*(xI[3]-x0) - r12/r11/r22/r22*( (yI[3] -y0) - (xI[3]-x0)*r12/r11 ))*indexI[3];
                # WiX[4] = (1.0/r11/r11*(xI[4]-x0) - r12/r11/r22/r22*( (yI[4] -y0) - (xI[4]-x0)*r12/r11 ))*indexI[4];

                WiY[1] = 1.0/r22/r22*( (yI[1]-y0) - (xI[1]-x0)*r12/r11) ;  
                WiY[2] = 1.0/r22/r22*( (yI[2]-y0) - (xI[2]-x0)*r12/r11) ;  
                WiY[3] = 1.0/r22/r22*( (yI[3]-y0) - (xI[3]-x0)*r12/r11) ;  
                WiY[4] = 1.0/r22/r22*( (yI[4]-y0) - (xI[4]-x0)*r12/r11) ;  

                WiX[1] = (1.0/r11/r11*(xI[1]-x0) - r12/r11/r22/r22*( (yI[1] -y0) - (xI[1]-x0)*r12/r11 )) ;
                WiX[2] = (1.0/r11/r11*(xI[2]-x0) - r12/r11/r22/r22*( (yI[2] -y0) - (xI[2]-x0)*r12/r11 )) ;
                WiX[3] = (1.0/r11/r11*(xI[3]-x0) - r12/r11/r22/r22*( (yI[3] -y0) - (xI[3]-x0)*r12/r11 )) ;
                WiX[4] = (1.0/r11/r11*(xI[4]-x0) - r12/r11/r22/r22*( (yI[4] -y0) - (xI[4]-x0)*r12/r11 )) ;



                gradX[BCell] = (WiX[1]*(phiI[1]-phi0) + WiX[2]*(phiI[2]-phi0) + WiX[3]*(phiI[3]-phi0) + WiX[4]*(phiI[4]-phi0) )*testMesh.cell_areas[BCell] ;
                gradY[BCell] = (WiY[1]*(phiI[1]-phi0) + WiY[2]*(phiI[2]-phi0) + WiY[3]*(phiI[3]-phi0) + WiY[4]*(phiI[4]-phi0) )*testMesh.cell_areas[BCell] ; 
               

            end
  
  end



# DO NOT USE!!!
# function calcScalarFieldGradient(cellsThreads::Array{Int32,2}, testMesh::mesh2d_Int32, 
#     scalarField::Array{Float64,1}, gradX::Array{Float64,1}, gradY::Array{Float64,1})
  
#     ## scalarField in Mesh Nodes !!!!
  
#     Threads.@threads for p in 1:Threads.nthreads()
           

#                 phiFaceX = zeros(Float64,4);
#                 phiFaceY = zeros(Float64,4);
                
#                 phiLeft = zeros(Float64,4);
#                 phiRight = zeros(Float64,4);
                
#                 side = zeros(Float64,4);
#                 nx = zeros(Float64,4);
#                 ny = zeros(Float64,4);
                
            
#                 for C =  cellsThreads[p,1]:cellsThreads[p,2]
                
                    
#                     side[1] = testMesh.cell_edges_length[C,1];
#                     side[2] = testMesh.cell_edges_length[C,2];
#                     side[3] = testMesh.cell_edges_length[C,3];
                    
#                     nx[1] = testMesh.cell_edges_Nx[C,1];
#                     nx[2] = testMesh.cell_edges_Nx[C,2];
#                     nx[3] = testMesh.cell_edges_Nx[C,3];
                    
#                     ny[1] = testMesh.cell_edges_Ny[C,1];
#                     ny[2] = testMesh.cell_edges_Ny[C,2];
#                     ny[3] = testMesh.cell_edges_Ny[C,3];
                    
            
#                     phiLeft[1] =  scalarField[ testMesh.cells2nodes[C,1] ];
#                     phiRight[1] = scalarField[ testMesh.cells2nodes[C,2] ];
            
#                     phiLeft[2] =  scalarField[ testMesh.cells2nodes[C,3] ];
#                     phiRight[2] = scalarField[ testMesh.cells2nodes[C,4] ];
            
#                     phiLeft[3] =  scalarField[ testMesh.cells2nodes[C,5] ];
#                     phiRight[3] = scalarField[ testMesh.cells2nodes[C,6] ];
                    
            
#                     if (testMesh.mesh_connectivity[C,3] == 4) ## if number of node cells == 4 
                    
#                         side[4] = testMesh.cell_edges_length[C,4];
#                         nx[4] = testMesh.cell_edges_Nx[C,4];
#                         ny[4] = testMesh.cell_edges_Ny[C,4];
                
#                         phiLeft[4] =  scalarField[ testMesh.cells2nodes[C,7] ];
#                         phiRight[4] = scalarField[ testMesh.cells2nodes[C,8] ];
                    
#                     end
                    
                    
#                     phiFaceX[1] = 0.5*(phiLeft[1] + phiRight[1])*-nx[1]*side[1];		
#                     phiFaceY[1] = 0.5*(phiLeft[1] + phiRight[1])*-ny[1]*side[1];
            
#                     phiFaceX[2] = 0.5*(phiLeft[2] + phiRight[2])*-nx[2]*side[2];		
#                     phiFaceY[2] = 0.5*(phiLeft[2] + phiRight[2])*-ny[2]*side[2];
                    
#                     phiFaceX[3] = 0.5*(phiLeft[3] + phiRight[3])*-nx[3]*side[3];		
#                     phiFaceY[3] = 0.5*(phiLeft[3] + phiRight[3])*-ny[3]*side[3];
                    
#                     phiFaceX[4] = 0.5*(phiLeft[4] + phiRight[4])*-nx[4]*side[4];		
#                     phiFaceY[4] = 0.5*(phiLeft[4] + phiRight[4])*-ny[4]*side[4];
            
#                     gradX[C] =  (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4]) / testMesh.cell_areas[C];
#                     gradY[C] =  (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4]) / testMesh.cell_areas[C];
                    
        
            
#                 end ## end global loop for cells
  
#             end ## threads
  
#   end




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

    #calcScalarFieldGradient(cellsThreads,testMesh, UNodes, UGradXApprox, UGradYApprox);

    calcScalarFieldGreenGaussNodesBasedGradient(cellsThreads, testMesh, U, UNodes, UGradXApprox, UGradYApprox);
    

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
