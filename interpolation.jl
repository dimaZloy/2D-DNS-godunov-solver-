



using Distributed;
using PyPlot;
using ScatteredInterpolation;


@everywhere using PyPlot;
@everywhere using WriteVTK;
@everywhere using CPUTime;
@everywhere using DelimitedFiles;
@everywhere using Printf
@everywhere using BSON: @load
@everywhere using BSON: @save
@everywhere using SharedArrays;


include("primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");
include("RoeFlux2d.jl")
include("AUSMflux2d.jl"); #AUSM+ inviscid flux calculation 
include("utilsFVM2dp.jl"); #FVM utililities
include("calcGrad.jl");
include("calcDiv.jl");
include("calcArtViscosity.jl");
include("calcDiffterm.jl");


display("load 2d-mesh 600x240... ")
@time @load "2mixinglayer_600x240.bson" testMesh
display("done... ")

testMesh600x240 = deepcopy(testMesh)	
triangles600x240 = zeros(Int64,testMesh600x240.nCells,3);


for i = 1:testMesh600x240.nCells
	
	## indexes of nodes in PyPLOT are started from Zero!!!!	
	triangles600x240[i,1] = testMesh600x240.mesh_connectivity[i,4]-1;
	triangles600x240[i,2] = testMesh600x240.mesh_connectivity[i,5]-1;
	triangles600x240[i,3] = testMesh600x240.mesh_connectivity[i,6]-1;
end
	
display("load 2d-solution 600x240 ... ")
@time @load "2dMixingLayer600x240" solInst;
display("done... ")

solInst600x240 = deepcopy(solInst);


density600x240   	= zeros(Float64, testMesh600x240.nNodes);
Ux600x240 			= zeros(Float64, testMesh600x240.nNodes);
Uy600x240 			= zeros(Float64, testMesh600x240.nNodes);
pressure600x240 	= zeros(Float64, testMesh600x240.nNodes);

cells2nodesSolutionReconstructionWithStencilsSA(testMesh600x240, solInst600x240.densityCells, density600x240 ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh600x240, solInst600x240.UxCells, Ux600x240 ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh600x240, solInst600x240.UyCells, Uy600x240 ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh600x240, solInst600x240.pressureCells, pressure600x240 ); 

figure(1)
clf()
subplot(2,1,1);	
tricontourf(testMesh600x240.xNodes,testMesh600x240.yNodes, triangles600x240, density600x240);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of density");
axis("equal");



#points600x240 = [testMesh600x240.xNodes testMesh600x240.yNodes]';
#itpRho = interpolate(Multiquadratic(), points600x240, density600x240 );


display("load 2d-mesh 1200x480... ")
@time @load "2mixinglayer_1200x480.bson" testMesh
display("done... ")

testMesh1200x480 = deepcopy(testMesh);
nCells1200x480 = testMesh1200x480.nCells;

triangles1200x480 = zeros(Int64,testMesh1200x480.nCells,3);


for i = 1:testMesh1200x480.nCells
	
	## indexes of nodes in PyPLOT are started from Zero!!!!	
	triangles1200x480[i,1] = testMesh1200x480.mesh_connectivity[i,4]-1;
	triangles1200x480[i,2] = testMesh1200x480.mesh_connectivity[i,5]-1;
	triangles1200x480[i,3] = testMesh1200x480.mesh_connectivity[i,6]-1;
end


density1200x480   	= zeros(Float64, testMesh1200x480.nCells);
Ux1200x480 			= zeros(Float64, testMesh1200x480.nCells);
Uy1200x480 			= zeros(Float64, testMesh1200x480.nCells);
pressure1200x480 	= zeros(Float64, testMesh1200x480.nCells);

densityNodes1200x480   	= zeros(Float64, testMesh1200x480.nNodes);
UxNodes1200x480 		= zeros(Float64, testMesh1200x480.nNodes);
UyNodes1200x480 		= zeros(Float64, testMesh1200x480.nNodes);
pressureNodes1200x480 	= zeros(Float64, testMesh1200x480.nNodes);



# for i = 1:nCells1200x480
	# xc = testMesh1200x480.cell_mid_points[i,1];
	# yc = testMesh1200x480.cell_mid_points[i,2];
	
	# density1200x480[i] = evaluate(itpRho, [xc; yc])
	
# end

# cells2nodesSolutionReconstructionWithStencilsSA(testMesh1200x480, density1200x480, densityNodes1200x480 ); 

subplot(2,1,2);	
tricontourf(testMesh1200x480.xNodes,testMesh1200x480.yNodes, triangles1200x480, densityNodes1200x480);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of density");
axis("equal");



