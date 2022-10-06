

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
include("calcGrad.jl");
include("calcDiv.jl");
#include("calcArtViscosity.jl");

include("testDiffOperatorsGrads.jl");
include("testDiffOperatorsLaplace.jl");
include("testDiffOperatorsDiv.jl");

# @inline function getRightSolutionCells(i::Int32, k::Int32, uLeftp::Float64, testMesh::mesh2d_Int32, solutionCells::Vector{Float64})::Float64

#     ek::Int32 = testMesh.cell_stiffness[i,k]; ## get right cell 
# 	#ek_type::Int32 = testMesh.mesh_connectivity[i,2];
		
# 	if (ek >=1 && ek<=testMesh.nCells)
								   
# 		return solutionCells[ek];
#         #x2 = testMesh.cell_mid_points[ek,1];
#         #y2 = testMesh.cell_mid_points[ek,2];
    
# 	else
					
#        return uLeftp;
#         #x2 = y2 = 0.0;
	
# 	end
				
# end

# @inline function calcRadiusToRightSolutionCells(i::Int32, k::Int32, testMesh::mesh2d_Int32)

#     ek::Int32 = testMesh.cell_stiffness[i,k]; ## get right cell 
	
#     xLeft::Float64 = testMesh.cell_mid_points[i,1];
#     yLeft::Float64 = testMesh.cell_mid_points[i,2];

#     xRight::Float64 = 2.0*xLeft;
#     yRight::Float64 = 2.0*yLeft;


# 	if (ek >=1 && ek<=testMesh.nCells)
		
#         xRight = testMesh.cell_mid_points[ek,1];
#         yRight = testMesh.cell_mid_points[ek,2];    
# 	end
		
#     return sqrt( (xRight-xLeft)*(xRight-xLeft) + (yRight-yLeft)*(yRight-yLeft)  ), (xRight-xLeft), (yRight-yLeft); 

# end


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
    


testCalcGradients("testQuadSquareDomain100x100")
testCalcGradients("testTriSquareDomain100x100")

testCalcDivergence("testQuadSquareDomain100x100")
testCalcDivergence("testTriSquareDomain100x100")

testCalcLaplacian("testQuadSquareDomain100x100")
testCalcLaplacian("testTriSquareDomain100x100")

