

using Distributed;
using PyPlot;

using WriteVTK;
using CPUTime;
using DelimitedFiles;
using Printf
using BSON: @load
using BSON: @save
using SharedArrays;

using HDF5;
using ProfileView;
using CUDA;


include("primeObjects.jl");
#include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");


#include("AUSMflux2dFast.jl"); #AUSM+ inviscid flux calculation 
#include("RoeFlux2dFast.jl");
#include("AUSMflux2dCUDA.jl");
#include("RoeFlux2dCUDA.jl");

include("utilsFVM2dp.jl"); #FVM utililities
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsImplicitSA
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsSA
## utilsFVM2dp::phs2dcns2dcellsSA

include("partMesh2d.jl");
#include("calcGrad.jl");
include("calcDiv.jl");
#include("calcArtViscosity.jl");

include("testDiffOperatorsGrads.jl");
include("testDiffOperatorsLaplace.jl");
include("testDiffOperatorsDiv.jl");

@inline function getRightSolutionCells(i::Int32, k::Int32, uLeftp::Float64, testMesh::mesh2d_Int32, solutionCells::Vector{Float64})::Float64

    ek::Int32 = testMesh.cell_stiffness[i,k]; ## get right cell 
	#ek_type::Int32 = testMesh.mesh_connectivity[i,2];
		
	if (ek >=1 && ek<=testMesh.nCells)
								   
		return solutionCells[ek];
        #x2 = testMesh.cell_mid_points[ek,1];
        #y2 = testMesh.cell_mid_points[ek,2];
    
	else
					
       return uLeftp;
        #x2 = y2 = 0.0;
	
	end
				

end

@inline function calcRadiusToRightSolutionCells(i::Int32, k::Int32, testMesh::mesh2d_Int32)

    ek::Int32 = testMesh.cell_stiffness[i,k]; ## get right cell 
	
    xLeft::Float64 = testMesh.cell_mid_points[i,1];
    yLeft::Float64 = testMesh.cell_mid_points[i,2];

    xRight::Float64 = 2.0*xLeft;
    yRight::Float64 = 2.0*yLeft;


	if (ek >=1 && ek<=testMesh.nCells)
		
        xRight = testMesh.cell_mid_points[ek,1];
        yRight = testMesh.cell_mid_points[ek,2];    
	end
		
    return sqrt( (xRight-xLeft)*(xRight-xLeft) + (yRight-yLeft)*(yRight-yLeft)  ), (xRight-xLeft), (yRight-yLeft); 

end



function nodesDivergenceReconstructionFastSA44( cellsThreads::Array{Int32,2}, testMesh::mesh2d_Int32, solutionCells::Vector{Float64},  
    gradX::Vector{Float64}, gradY::Vector{Float64}, divergence::Vector{Float64})
    
    Threads.@threads for p in 1:Threads.nthreads()
    
          phiLeftX = zeros(Float64,4);
          phiLeftY = zeros(Float64,4);
  
          phiRightX = zeros(Float64,4);
          phiRightY = zeros(Float64,4);
  
          phiFaceX = zeros(Float64,4);
          phiFaceY  = zeros(Float64,4);
  
          side = zeros(Float64,4);
          nx = zeros(Float64,4);
          ny = zeros(Float64,4);
          
          # T1::Int32 = 0;
          # T2::Int32 = 0;
          # T3::Int32 = 0;
          # T4::Int32 = 0;
   
  
          ULeft::Float64 = 0.0;
          URight1::Float64 = 0.0;
          URight2::Float64 = 0.0;
          URight3::Float64 = 0.0;
          URight4::Float64 = 0.0;

          dr1 ::Float64 = 0.0;
          dr2 ::Float64 = 0.0;
          dr3 ::Float64 = 0.0;
          dr4 ::Float64 = 0.0;

          dx1 ::Float64 = 0.0;
          dx2 ::Float64 = 0.0;
          dx3 ::Float64 = 0.0;
          dx4 ::Float64 = 0.0;

          dy1 ::Float64 = 0.0;
          dy2 ::Float64 = 0.0;
          dy3 ::Float64 = 0.0;
          dy4 ::Float64 = 0.0;

          alpha::Float64 = 4.0/3.0;

        for C =cellsThreads[p,1]:cellsThreads[p,2]
    
  
          ##numNodesInCell::Int32 = testMesh.mesh_connectivity[C,3]; ## CMatrix mesh_connectivity - first index == 1
          
          ULeft = solutionCells[C];
          URight1 = getRightSolutionCells(C, Int32(1), ULeft, testMesh, solutionCells);
          URight2 = getRightSolutionCells(C, Int32(2), ULeft, testMesh, solutionCells);
          URight3 = getRightSolutionCells(C, Int32(3), ULeft, testMesh, solutionCells);
          
          
          (dr1,dx1,dy1) = calcRadiusToRightSolutionCells(C, Int32(1), testMesh);
          (dr2,dx2,dy2) = calcRadiusToRightSolutionCells(C, Int32(2), testMesh);
          (dr3,dx3,dy3) = calcRadiusToRightSolutionCells(C, Int32(3), testMesh);
         
          
          
          side[1] = testMesh.cell_edges_length[C,1];
          side[2] = testMesh.cell_edges_length[C,2];
          side[3] = testMesh.cell_edges_length[C,3];
        
          nx[1] = testMesh.cell_edges_Nx[C,1];
          nx[2] = testMesh.cell_edges_Nx[C,2];
          nx[3] = testMesh.cell_edges_Nx[C,3];
        
          ny[1] = testMesh.cell_edges_Ny[C,1];
          ny[2] = testMesh.cell_edges_Ny[C,2];
          ny[3] = testMesh.cell_edges_Ny[C,3];
          
  
          phiLeftX[1] =  gradX[ testMesh.cells2nodes[C,1] ];
          phiLeftY[1] =  gradY[ testMesh.cells2nodes[C,1] ];
      
          phiRightX[1] = gradX[ testMesh.cells2nodes[C,2] ];
          phiRightY[1] = gradY[ testMesh.cells2nodes[C,2] ];
  
          phiLeftX[2] =  gradX[ testMesh.cells2nodes[C,3] ];
          phiLeftY[2] =  gradY[ testMesh.cells2nodes[C,3] ];
      
          phiRightX[2] = gradX[ testMesh.cells2nodes[C,4] ];
          phiRightY[2] = gradY[ testMesh.cells2nodes[C,4] ];
      
          phiLeftX[3] =  gradX[ testMesh.cells2nodes[C,5] ];
          phiLeftY[3] =  gradY[ testMesh.cells2nodes[C,5] ];
      
          phiRightX[3] = gradX[ testMesh.cells2nodes[C,6] ];
          phiRightY[3] = gradY[ testMesh.cells2nodes[C,6] ];
  
  
           phiFaceX[1] = 0.5*(phiLeftX[1] + phiRightX[1])*-nx[1]*side[1]; 	
           phiFaceY[1] = 0.5*(phiLeftY[1] + phiRightY[1])*-ny[1]*side[1]; 
  
           phiFaceX[2] = 0.5*(phiLeftX[2] + phiRightX[2])*-nx[2]*side[2]; 
           phiFaceY[2] = 0.5*(phiLeftY[2] + phiRightY[2])*-ny[2]*side[2]; 
        
           phiFaceX[3] = 0.5*(phiLeftX[3] + phiRightX[3])*-nx[3]*side[3]; 
           phiFaceY[3] = 0.5*(phiLeftY[3] + phiRightY[3])*-ny[3]*side[3]; 


        #   phiFaceX[1] = phiFaceX[1] + 0.5*alpha/abs(dr1*nx[1])*(URight1  - ULeft );
        #   phiFaceY[1] = phiFaceY[1] + 0.5*alpha/abs(dr1*ny[1])*(URight1  - ULeft );

        #   phiFaceX[2] = phiFaceX[2] + 0.5*alpha/abs(dr2*nx[2])*(URight2  - ULeft );
        #   phiFaceY[2] = phiFaceY[2] + 0.5*alpha/abs(dr2*ny[2])*(URight2  - ULeft );

        #   phiFaceX[3] = phiFaceX[3] + 0.5*alpha/abs(dr3*nx[3])*(URight3  - ULeft );
        #   phiFaceY[3] = phiFaceY[3] + 0.5*alpha/abs(dr3*ny[3])*(URight3  - ULeft );


          if (testMesh.mesh_connectivity[C,3] == 4)
        
            
              side[4] = testMesh.cell_edges_length[C,4];
              nx[4] = testMesh.cell_edges_Nx[C,4];
              ny[4] = testMesh.cell_edges_Ny[C,4];
  
  
              phiLeftX[4] =  gradX[ testMesh.cells2nodes[C,7] ];
              phiLeftY[4] =  gradY[ testMesh.cells2nodes[C,7] ];
      
              phiRightX[4] = gradX[ testMesh.cells2nodes[C,8] ];
              phiRightY[4] = gradY[ testMesh.cells2nodes[C,8] ];
  
              URight4 = getRightSolutionCells(C, Int32(4), ULeft, testMesh, solutionCells);
              (dr4,dx4,dy4) = calcRadiusToRightSolutionCells(C, Int32(4), testMesh);
              
               phiFaceX[4] = 0.5*(phiLeftX[4] + phiRightX[4])*-nx[4]*side[4] ;		
               phiFaceY[4] = 0.5*(phiLeftY[4] + phiRightY[4])*-ny[4]*side[4] ; 	

            #   phiFaceX[4] = phiFaceX[4] + 0.5*alpha/abs(dr4*nx[4])*(URight4  - ULeft );
            #   phiFaceY[4] = phiFaceY[4] + 0.5*alpha/abs(dr4*ny[4])*(URight4  - ULeft );
              
          end

    
          #divX::Float64 = (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4])/testMesh.cell_areas[C];
          #divY::Float64 = (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4])/testMesh.cell_areas[C];
  
  
          divergence[C] = (phiFaceX[1] + phiFaceX[2] + phiFaceX[3] + phiFaceX[4])/testMesh.cell_areas[C] + (phiFaceY[1] + phiFaceY[2] + phiFaceY[3] + phiFaceY[4])/testMesh.cell_areas[C];
          
  
        end ## end of cells
  
    end ## end per thread
  
  end ## end of function
  


@inline function cells2nodesSolutionReconstructionWithStencilsVector(nodesThreads::Array{Int32,2}, testMesh::mesh2d_Int32, cell_solution::Array{Float64,1}, node_solution::Array{Float64,1} )

    Threads.@threads for p in 1:Threads.nthreads()
        
        for J = nodesThreads[p,1]:nodesThreads[p,2]
    
            
                det::Float64 = 0.0;
                for j = 1:testMesh.nNeibCells
                    neibCell::Int32 = testMesh.cell_clusters[J,j]; 
                    if (neibCell !=0)
                        wi::Float64 = testMesh.node_stencils[J,j];
                        node_solution[J] += cell_solution[neibCell]*wi;
                        det += wi;
                    end
                end
                if (det!=0)
                    node_solution[J] = node_solution[J]/det; 
                end
            

        end ## J
        
    end ## threads

end
    



function computeCellGradients(i::Int32, k::Int32, testMesh::mesh2d_Int32, scalarField::Vector{Float64}) 
	#gradX::Float64, gradY::Float64)
	

	##nCells = size(testMesh.cell_stiffness,1);
	ek::Int32 = testMesh.cell_stiffness[i,k]; ##; %% get right cell 
	ek_type::Int32 = testMesh.mesh_connectivity[i,2];
	

	uLeftp = scalarField[i];
    uRightp::Float64 = 0.0;
    
    x1::Float64 = testMesh.cell_mid_points[i,1];
    y1::Float64 = testMesh.cell_mid_points[i,2];
	x2::Float64 = 0.0;
    y2::Float64 = 0.0;		
	
	if (ek >=1 && ek<=testMesh.nCells)
								   
		uRightp = scalarField[ek];
        x2 = testMesh.cell_mid_points[ek,1];
        y2 = testMesh.cell_mid_points[ek,2];
    
					
	else
					
        uRightp = uLeftp;
        x2 = y2 = 0.0;
	
	end
				
	gradX::Float64 = 	(uRightp - uLeftp)/(x2-x1);
    gradY::Float64 = 	(uRightp - uLeftp)/(y2-y1);


    return gradX, gradY;
						
end





#testCalcGradients("cyl2d_laminar_test")
#testCalcDivergence("oblickShock2dl00F1n")
#testCalcDivergence("cyl2d_laminar_test")
testCalcLaplacian("cyl2d_laminar_test")

