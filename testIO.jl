

using Distributed;
using PyPlot;


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


display("load 2d-mesh ... ")
@load "2mixinglayer_600x240.bson" testMesh

	
triangles = zeros(Int64,testMesh.nCells,3);
for i = 1:testMesh.nCells
	
	## indexes of nodes in PyPLOT are started from Zero!!!!	
	triangles[i,1] = testMesh.mesh_connectivity[i,4]-1;
	triangles[i,2] = testMesh.mesh_connectivity[i,5]-1;
	triangles[i,3] = testMesh.mesh_connectivity[i,6]-1;
end
	
display("load 2d-solution ... ")
@load "zzz15700" solInst;

display(solInst.dt)
display(solInst.flowTime)
display(solInst.nCells)

density = zeros(Float64, testMesh.nNodes);
Ux = zeros(Float64, testMesh.nNodes);
Uy = zeros(Float64, testMesh.nNodes);
pressure = zeros(Float64, testMesh.nNodes);

cells2nodesSolutionReconstructionWithStencilsSA(testMesh, solInst.densityCells, density ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh, solInst.UxCells, Ux ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh, solInst.UyCells, Uy ); 
cells2nodesSolutionReconstructionWithStencilsSA(testMesh, solInst.pressureCells, pressure ); 

figure(1)
clf()
subplot(2,2,1);	
tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, density);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of density");
axis("equal");

subplot(2,2,2);	
tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, Ux);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of Ux");
axis("equal");

subplot(2,2,3);	
tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, Uy);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of Ux");
axis("equal");

subplot(2,2,4);	
tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, pressure);			
set_cmap("jet");
xlabel("x");
ylabel("y");
title("Contours of pressure");
axis("equal");



