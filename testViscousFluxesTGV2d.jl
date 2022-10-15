

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
using LaTeXStrings;


include("primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");


include("AUSMflux2dFast.jl"); #AUSM+ inviscid flux calculation 
#include("RoeFlux2dFast.jl");
#include("AUSMflux2dCUDA.jl");
include("AUSMflux2dCUDAx32.jl");
#include("RoeFlux2dCUDA.jl");

include("utilsFVM2dp.jl"); #FVM utililities
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsImplicitSA
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsSA
## utilsFVM2dp::phs2dcns2dcellsSA

include("partMesh2d.jl");

include("calcGradients.jl");
#include("calcDiv.jl");
#include("calcArtViscosity.jl");
include("calcDiffterm.jl");

#include("bcInviscidWall.jl"); 

include("createViscousFields.jl")
#include("boundaryConditions_jet2d.jl");
#include("initfields_jet2d.jl");

include("loadPrevResults.jl");

#include("boundaryConditions_ML2d.jl");
#include("initfields_ML2d.jl");

#include("boundaryConditions_cyl2d.jl");
#include("initfields_cyl2d.jl");

#include("boundaryConditions_step2d.jl");
#include("initfields_step2d.jl");

include("boundaryConditions_TGV2d.jl");
include("initfields_TGV2d.jl");


include("evaluate2d.jl"); 
## propagate2d::updateResidualSA()
## propagate2d::updateVariablesSA()
## propagate2d::updateOutputSA()


##include("computeslope2d.jl");
#include("SOUscheme.jl");


include("limiters.jl");
include("computeslope2d.jl");
include("computeMUSCLStencilsCUDA.jl");
include("calcOneStageCUDAx32.jl");
include("calcOneStageMUSCL.jl");
include("calcOneStageMUSCLcpuFast.jl");



## computeslope2d:: computeInterfaceSlope()
## SOUscheme:: SecondOrderUpwindM2()

#include("propagate2d.jl");
## propagate:: calcOneStage() expilict Euler first order
## propagate:: doExplicitRK3TVD() expilict RK3-TVD


function computeL2norm(testMesh::mesh2d_Int32, testFields::fields2d, thermo::THERMOPHYSICS)::Float64

	diffusion::Float64 = 0.0

	for i = 1: testMesh.nCells
		diffusion = diffusion + (testFields.pressureCells[i]/testFields.densityCells[i])^(thermo.Gamma);
	end

	return diffusion/testMesh.nCells; 

end

 function plotEkDecay(fileTri::String, fileQuad::String)

	 

 	@load fileTri*"Ek_TGV2d_CPUx64" timeVector kVectorTheory kVectorApprox
	dataTheory = kVectorTheory
 	dataTri = kVectorApprox
 	@load fileQuad*"Ek_TGV2d_CPUx64" kVectorTheory kVectorApprox
 	dataQuad = kVectorApprox
 	figure(1)
 	clf()
 	plot(timeVector, dataTheory ,"-k", label="theory")
	plot(timeVector, dataTri ,"--g", label="tri mesh")
	plot(timeVector, dataQuad ,"--b", label="quad mesh")
 	xlabel("time [s]");
 	ylabel("Ek(t)");
	xlim(0.0,0.25)
	ylim(0.9,1.0)
 	title("Decay of Ek for the compressible Tailor-Green Vortex 2d problem");
	yscale("log");	
	grid();
	legend();

	# @load "testStep2dBaseTriSmoothCPUx64" timeVector L2normVector 

 end

function calcAnalyticalTGV2d(time::Float64, nodesThreads::Array{Int32,2}, testMesh::mesh2d_Int32)

	ucells = zeros(Float64,testMesh.nCells,4);
	unodes = zeros(Float64,testMesh.nNodes,4);

	dummy4 = zeros(Float64,4)
	for i = 1: testMesh.nCells
		x::Float64 = testMesh.cell_mid_points[i,1];
		y::Float64 = testMesh.cell_mid_points[i,2];
		computeTGV2d(x, y, time, thermo, dummy4);
		ucells[i,1] = dummy4[1];
		ucells[i,2] = dummy4[2];
		ucells[i,3] = dummy4[3];
		ucells[i,4] = dummy4[4];
	end

		rhoNodes = zeros(Float64,testMesh.nNodes);
		uxNodes = zeros(Float64,testMesh.nNodes);
		uyNodes = zeros(Float64,testMesh.nNodes);
		pNodes = zeros(Float64,testMesh.nNodes);
	

		cells2nodesSolutionReconstructionWithStencilsUCons(nodesThreads, testMesh, ucells,  unodes );	
	
		for i = 1:testMesh.nNodes
			rhoNodes[i] 		= unodes[i,1];
			uxNodes[i] 			= unodes[i,2];
			uyNodes[i] 			= unodes[i,3];
			pNodes[i] 			= unodes[i,4];
		end
				
		println("Saving  Analytical TGV2d solution  ... ");
		saveResults4VTK("analticalSolutionTGV2dTime1e-3", testMesh, rhoNodes, uxNodes, uyNodes, pNodes);
		println("done ...  ");	

end

function displayInfo(thermo::THERMOPHYSICS)

	println("###################################################################");
	println("Compressible Taylor-Green vortex 2D problem ");

	println("rho   = ", 1.0, " kg/m3")
	println("gamma = ", thermo.Gamma, "  -")
	println("P     = ", 100000.0, " Pa")
    println("T     = ", 100000.0/thermo.RGAS, " K")
	println("U     = ", 40.0, " m/s")
	println("a     = ", sqrt(thermo.Gamma*100000.0/1.0), " m/s")
	println("M     = ", 40.0/sqrt(thermo.Gamma*100000.0/1.0), " m/s")
	println("mu    = ", 1e-2, " Pa*s")
	println("L     = ", 1.0, " m")
	println("Re    = ", 1.0*40.0*1.0/1e-2, " m")

	println("###################################################################");

end


@inline function computeTGV2d_Ek_Approx(cellsThreads::Array{Int32,2}, testMesh::mesh2d_Int32, testfields2d::fields2d, E0::Float64)::Float64

	#Threads.@threads for p in 1:Threads.nthreads()	
	
		Ek::Float64 = 0.0;
		#for i = 1:cellsThreads[p,1]: cellsThreads[p,2]

		for i = 1:testMesh.nCells	
			Ek += 0.5*testfields2d.densityCells[i]*( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] )*testMesh.cell_areas[i];
			#Ek += 0.5*( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] )*testMesh.cell_areas[i];
		end # end i-cell


	#end ## end p-thread

	return Ek/E0;

end

function godunov2dthreads(pname::String, outputfile::String, coldrun::Bool, useCuda::Bool )




	#useCuda = true;
	debug = false;	
	viscous = true;
	damping = false;
	flag2loadPreviousResults = false;
	



	testMesh = readMesh2dHDF5(pname);
		
	cellsThreads = distibuteCellsInThreadsSA(Threads.nthreads(), testMesh.nCells); ## partition mesh 
	nodesThreads = distibuteNodesInThreadsSA(Threads.nthreads(), testMesh.nNodes); ## partition mesh 
	




	#include("setupSolver_ML2d.jl"); #setup FVM and numerical schemes
	#include("setupSolver_jet2d.jl"); #setup FVM and numerical schemes
	#include("setupSolver_cyl2d.jl"); #setup FVM and numerical schemes
    #include("setupSolver_step2d.jl"); #setup FVM and numerical schemes
	include("setupSolver_TGV2d.jl"); #setup FVM and numerical schemes
	
	displayInfo(thermo)

	pause

	## init primitive variables 
	println("set initial and boundary conditions ...");
	
	
	#testfields2d = createFields2d_shared(testMesh, thermo);
	testfields2d = createFields2d(testMesh, thermo);
	
	solInst = solutionCellsT(
		0.0,
		0.0,
		testMesh.nCells,
		testfields2d.densityCells,
		testfields2d.UxCells,
		testfields2d.UyCells,
		testfields2d.pressureCells,
	);
	
	
	if (flag2loadPreviousResults)
		loadPrevResults(testMesh, thermo, "jet2d_v03tri.tmp", dynControls, testfields2d);
	end
	
	

	viscfields2d = createViscousFields2d(testMesh.nCells, testMesh.nNodes);
	
	println("nCells:\t", testMesh.nCells);
	println("nNodes:\t", testMesh.nNodes);
	
	## init conservative variables 	
	
	UconsCellsOldX = zeros(Float64,testMesh.nCells,4);
	UconsNodesOldX = zeros(Float64,testMesh.nNodes,4);
	UconsCellsNewX = zeros(Float64,testMesh.nCells,4);
		
	UConsDiffCellsX = zeros(Float64,testMesh.nCells,4);
	UConsDiffNodesX = zeros(Float64,testMesh.nNodes,4);
	
	DeltaX = zeros(Float64,testMesh.nCells,4);
	iFLUXX  = zeros(Float64,testMesh.nCells,4);
	dummy  = zeros(Float64,testMesh.nNodes,4);

	 uRight1 = zeros(Float64,4,testMesh.nCells);
	 uRight2 = zeros(Float64,4,testMesh.nCells);
	 uRight3 = zeros(Float64,4,testMesh.nCells);
	 uRight4 = zeros(Float64,4,testMesh.nCells);

	 uLeft1 = zeros(Float64,4,testMesh.nCells);
	 uLeft2 = zeros(Float64,4,testMesh.nCells);
	 uLeft3 = zeros(Float64,4,testMesh.nCells);
	 uLeft4 = zeros(Float64,4,testMesh.nCells);

	iFluxV1 = zeros(Float64,testMesh.nCells);
	iFluxV2 = zeros(Float64,testMesh.nCells);
	iFluxV3 = zeros(Float64,testMesh.nCells);
	iFluxV4 = zeros(Float64,testMesh.nCells);

	# iFlux1 = zeros(Float64,4,testMesh.nCells);
	# iFlux2 = zeros(Float64,4,testMesh.nCells);
	# iFlux3 = zeros(Float64,4,testMesh.nCells);
	# iFlux4 = zeros(Float64,4,testMesh.nCells);
	
	###################################################################################
	## Vectors for CUDA 


	# uRight1 = zeros(Float32,4,testMesh.nCells);
	# uRight2 = zeros(Float32,4,testMesh.nCells);
	# uRight3 = zeros(Float32,4,testMesh.nCells);
	# uRight4 = zeros(Float32,4,testMesh.nCells);

	# uLeft1 = zeros(Float32,4,testMesh.nCells);
	# uLeft2 = zeros(Float32,4,testMesh.nCells);
	# uLeft3 = zeros(Float32,4,testMesh.nCells);
	# uLeft4 = zeros(Float32,4,testMesh.nCells);

 	
    SidesV = zeros(Float32,testMesh.nCells*4);
	nxV = zeros(Float32,testMesh.nCells*4);
	nyV = zeros(Float32,testMesh.nCells*4);

	SidesV[1:testMesh.nCells] 						= testMesh.cell_edges_length[1:end,1];
	SidesV[testMesh.nCells*1+1: testMesh.nCells*2] 	= testMesh.cell_edges_length[1:end,2];
	SidesV[testMesh.nCells*2+1: testMesh.nCells*3] 	= testMesh.cell_edges_length[1:end,3];
	SidesV[testMesh.nCells*3+1: end] 				= testMesh.cell_edges_length[1:end,4];

	nxV[1:testMesh.nCells] 						= testMesh.cell_edges_Nx[1:end,1];
	nxV[testMesh.nCells*1+1: testMesh.nCells*2] = testMesh.cell_edges_Nx[1:end,2];
	nxV[testMesh.nCells*2+1: testMesh.nCells*3] = testMesh.cell_edges_Nx[1:end,3];
	nxV[testMesh.nCells*3+1: end] 				= testMesh.cell_edges_Nx[1:end,4];

	nyV[1:testMesh.nCells] 						= testMesh.cell_edges_Ny[1:end,1];
	nyV[testMesh.nCells*1+1: testMesh.nCells*2] = testMesh.cell_edges_Ny[1:end,2];
	nyV[testMesh.nCells*2+1: testMesh.nCells*3] = testMesh.cell_edges_Ny[1:end,3];
	nyV[testMesh.nCells*3+1: end] 				= testMesh.cell_edges_Ny[1:end,4];

	numNeibs = copy(testMesh.mesh_connectivity[:,3]);

	cuNeibsV = CuVector{Int32, Mem.DeviceBuffer}(undef,testMesh.nCells)

	curLeftV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuULeftV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuVLeftV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuPLeftV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)

	curRightV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuURightV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuVRightV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuPRightV = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells) 

	cuSideV1234 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuNxV1234 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuNyV1234 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuFluxV1234 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells*4)

	cuFluxV1 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuFluxV2 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuFluxV3 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuFluxV4 = CuVector{Float32, Mem.DeviceBuffer}(undef,testMesh.nCells)


	copyto!(cuSideV1234, SidesV)
	copyto!(cuNxV1234, nxV)
	copyto!(cuNyV1234, nyV)

	copyto!(cuNeibsV,numNeibs )
	###################################################################################
	# compute analytical TGV-2d solution for t = controls.stopTime 

	calcAnalyticalTGV2d(solControls.stopTime, nodesThreads, testMesh); 

	## kinetic energy at t = 0;
	E0Theory::Float64 = computeTGV2d_Ek_Theory(40.0, 1e-2, 1.0, 0.0);
	E0Approx::Float64 = computeTGV2d_Ek_Approx(cellsThreads, testMesh, testfields2d, 1.0);

	display("Initial Kinetic Energy: ")
	display(E0Theory)
	display(E0Approx)

	kVectorTheory = [];
	kVectorApprox = [];

	###################################################################################
	
	phs2dcns2dcellsSA(UconsCellsOldX,testfields2d, thermo.Gamma);	
	phs2dcns2dcellsSA(UconsCellsNewX,testfields2d, thermo.Gamma);	
	
	
	cells2nodesSolutionReconstructionWithStencilsUCons(nodesThreads, testMesh, UconsCellsOldX,  UconsNodesOldX );	

	

	timeVector = [];
	residualsVector1 = []; 
	residualsVector2 = []; 
	residualsVector3 = []; 
	residualsVector4 = []; 
	residualsVectorMax = ones(Float64,4);
	convergenceCriteria = [1e-5;1e-5;1e-5;1e-5;];
	L2normVector = []; 
	
	maxEdge,id = findmax(testMesh.HX);

	dt::Float64 =  solControls.dt;  
	# @everywhere dtX = $dt; 
	# @everywhere maxEdgeX = $maxEdge; 


	
	println("Start calculations ...");
	println(output.header);
	
	##if (!coldrun)
	
	
		#for l = 1:2
		while (dynControls.isRunSimulation == 1)
		
			
			##CPUtic();	
			start = time();
			
			
			# PROPAGATE STAGE: 
			(dynControls.velmax,id) = findmax(testfields2d.VMAXCells);
			# #dynControls.tau = solControls.CFL * testMesh.maxEdgeLength/(max(dynControls.velmax,1.0e-6)); !!!!
			dynControls.tau = solControls.CFL * maxEdge/(max(dynControls.velmax,1.0e-6));
		
			
			if (viscous)
				calcDiffTerm(cellsThreads, nodesThreads, testMesh, testfields2d, viscfields2d, thermo, UconsCellsOldX, UconsNodesOldX, UConsDiffCellsX);
			end
	
			
			

			if (CUDA.has_cuda() && useCuda)
			
				 
				computeMUSCLStencilsCUDA(cellsThreads, testMesh, testfields2d, thermo,	
				uLeft1, uLeft2, uLeft3, uLeft4, uRight1, uRight2, uRight3, uRight4, numNeibs, dynControls.flowTime);
				
			 	 calcOneStageCUDAx32(1.0, solControls.dt, dynControls.flowTime, 
				 		testMesh , testfields2d, thermo, cellsThreads,  
				 		UconsCellsOldX, UConsDiffCellsX,  UconsCellsNewX, 
			 	 		uLeft1, uLeft2, uLeft3, uLeft4, 
						uRight1, uRight2, uRight3, uRight4,
						iFluxV1, iFluxV2, iFluxV3, iFluxV4, 
						curLeftV, cuULeftV, cuVLeftV, cuPLeftV, 
						curRightV, cuURightV, cuVRightV, cuPRightV, 
					  	cuNxV1234, #cuNxV1, cuNxV2, cuNxV3, cuNxV4, 
						cuNyV1234, #cuNyV1, cuNyV2, cuNyV3, cuNyV4, 
						cuSideV1234, #cuSideV1, cuSideV2, cuSideV3, cuSideV4,
					  	cuFluxV1, cuFluxV2, cuFluxV3, cuFluxV4,  
						cuNeibsV);


			else

			 	## Explicit Euler first-order	
			 	

				 calcOneStageMUSCLcpuFast(cellsThreads, 1.0, solControls.dt, dynControls.flowTime,  
				 	testMesh, testfields2d, thermo, UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);	

			

				 #calcOneStageMUSCL(1.0, solControls.dt, dynControls.flowTime, 
				 #	testMesh, testfields2d, thermo, cellsThreads, UconsCellsOldX, UConsDiffCellsX,  UconsCellsNewX, 
			 	 # 		uLeft1,  uLeft2, uLeft3, uLeft4,  uRight1, uRight2, uRight3, uRight4, iFLUXX);


			end
			
			#doExplicitRK3TVD(1.0, dtX, testMeshDistrX , testfields2dX, thermoX, cellsThreadsX,  UconsCellsOldX, iFLUXX,  UConsDiffCellsX, 
			#  UconsCellsNew1X,UconsCellsNew2X,UconsCellsNew3X,UconsCellsNewX);
			
			
			
			updateVariablesSA(cellsThreads, thermo.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2d, solControls, dynControls );
			
			cells2nodesSolutionReconstructionWithStencils(nodesThreads,	testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
			


			# update time step
			push!(timeVector, dynControls.flowTime); 
			push!(kVectorTheory, computeTGV2d_Ek_Theory(40.0, 1e-2, 1.0, dynControls.flowTime)/E0Theory);
			push!(kVectorApprox, computeTGV2d_Ek_Approx(cellsThreads, testMesh, testfields2d, E0Approx));

			dynControls.curIter += 1; 
			dynControls.verIter += 1;
			#push!(L2normVector, computeL2norm(testMesh, testfields2d, thermo)); 
				
			
			updateResidualSA(DeltaX, 
				residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax,  
				convergenceCriteria, dynControls);
			
			
			updateOutputSA(timeVector,residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax, 
				testMesh, testfields2d, viscfields2d,  solControls, output, dynControls, solInst, thermo, kVectorTheory, kVectorApprox);
	
			
			# EVALUATE STAGE:
			
			dynControls.flowTime += dt; 
			##flowTimeX += dt;
			
			# if (solControlsX.timeStepMethod == 1)
				# dynControlsX.flowTime += dynControlsX.tau;  	
			# else
				# dynControlsX.flowTime += solControlsX.dt;  
			# end
			

	
			if (flowTime>= solControls.stopTime || dynControls.isSolutionConverged == 1) 
			#if (flowTime>= solControls.stopTime)
				dynControls.isRunSimulation = 0;
		
				if (dynControls.isSolutionConverged == true)
					println("Solution converged! ");
				else
					println("Simultaion flow time reached the set Time!");
				end
			
				if (output.saveResiduals == 1)
					#println("Saving Residuals ... ");
					#cd(dynControlsX.localTestPath);
					#saveResiduals(output.fileNameResiduals, timeVector, residualsVector1, residualsVector2, residualsVector3, residualsVector4);
					#cd(dynControlsX.globalPath);
				end
				if (output.saveResults == 1)
					#println("Saving Results ... ");
					#cd(dynControlsX.localTestPath);
					#saveSolution(output.fileNameResults, testMeshX.xNodes, testMeshX.yNodes, UphysNodes);
					#cd(dynControlsX.globalPath);
				end
			
				
			
			end

			#dynControlsX.cpuTime  += CPUtoq(); 
			elapsed = time() - start;
			dynControls.cpuTime  += elapsed ; 
			
			if (dynControls.flowTime >= solControls.stopTime)
				dynControls.isRunSimulation = 0;
			end
			
		end ## end while
		 
		 
		
		
		solInst.dt = solControls.dt;
		solInst.flowTime = dynControls.flowTime;
		for i = 1 : solInst.nCells
		 	solInst.densityCells[i] 	= testfields2d.densityCells[i];
		 	solInst.UxCells[i] 			= testfields2d.UxCells[i];
		 	solInst.UyCells[i] 			= testfields2d.UyCells[i];
		 	solInst.pressureCells[i] 	= testfields2d.pressureCells[i];
		end
		

		rhoNodes = zeros(Float64,testMesh.nNodes);
		uxNodes = zeros(Float64,testMesh.nNodes);
		uyNodes = zeros(Float64,testMesh.nNodes);
		pNodes = zeros(Float64,testMesh.nNodes);
	
		cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreads, testMesh, testfields2d, dummy); 
	
		for i = 1:testMesh.nNodes
			rhoNodes[i] 		= testfields2d.densityNodes[i];
			uxNodes[i] 			= testfields2d.UxNodes[i];
			uyNodes[i] 			= testfields2d.UyNodes[i];
			pNodes[i] 			= testfields2d.pressureNodes[i];
		end
				
		println("Saving  solution to  ", outputfile);

		if useCuda
			saveResults4VTK(outputfile*"TGV2d_GPUx32", testMesh, rhoNodes, uxNodes, uyNodes, pNodes);
			@save outputfile*"Ek_TGV2d_GPUx32" solInst timeVector kVectorTheory kVectorApprox
			
		else
			saveResults4VTK(outputfile*"TGV2d_CPUx64", testMesh, rhoNodes, uxNodes, uyNodes, pNodes);
			@save outputfile*"Ek_TGV2d_CPUx64" solInst timeVector kVectorTheory kVectorApprox
			
		end	
		println("done ...  ");	
		
		 
		 
		
	#end ## if debug
	
end


godunov2dthreads("testTriSquareDomain100x100",  "testTriSquareDomain100x100", false, false);  ## CPUx64
#godunov2dthreads("testQuadSquareDomain100x100",  "testQuadSquareDomain100x100", false, false);  ## CPUx64

#plotEkDecay("testTriSquareDomain100x100", "testQuadSquareDomain100x100")





