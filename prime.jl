

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
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");


include("AUSMflux2dFast.jl"); #AUSM+ inviscid flux calculation 
include("AUSMflux2dCUDA.jl");
#include("RoeFlux2dCUDA.jl");

include("utilsFVM2dp.jl"); #FVM utililities
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsImplicitSA
## utilsFVM2dp::cells2nodesSolutionReconstructionWithStencilsSA
## utilsFVM2dp::phs2dcns2dcellsSA

include("partMesh2d.jl");

include("calcGrad.jl");
include("calcDiv.jl");
include("calcArtViscosity.jl");
include("calcDiffterm.jl");

include("bcInviscidWall.jl"); 
include("boundaryConditions2d.jl"); 

include("initfields2d.jl");
## initfields2d::distibuteCellsInThreadsSA()
## initfields2d::createFields2d_shared()

include("evaluate2d.jl"); 
## propagate2d::updateResidualSA()
## propagate2d::updateVariablesSA()
## propagate2d::updateOutputSA()


##include("computeslope2d.jl");
#include("SOUscheme.jl");


include("limiters.jl");
include("computeslope2d.jl");
include("SOUscheme.jl");


## computeslope2d:: computeInterfaceSlope()
## SOUscheme:: SecondOrderUpwindM2()

include("propagate2d.jl");
## propagate:: calcOneStage() expilict Euler first order
## propagate:: doExplicitRK3TVD() expilict RK3-TVD






function godunov2dthreads(pname::String, outputfile::String, coldrun::Bool)


	useCuda = false;

	flag2loadPreviousResults = false;

	testMesh = readMesh2dHDF5(pname);
		
	cellsThreads = distibuteCellsInThreadsSA(Threads.nthreads(), testMesh.nCells); ## partition mesh 
	nodesThreads = distibuteNodesInThreadsSA(Threads.nthreads(), testMesh.nNodes); ## partition mesh 
	

	include("setupSolver2d.jl"); #setup FVM and numerical schemes
	
	
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
	
	
	#(testfields2d, solInst) = createFields2dLoadPrevResults_shared(testMesh, thermo, "zzz13700", dynControls);
	
	
	
	#viscfields2d = createViscousFields2d_shared(testMesh.nCells, testMesh.nNodes);
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

	uLeft = zeros(Float64,4,testMesh.nCells);

	iFluxV1 = zeros(Float64,testMesh.nCells);
	iFluxV2 = zeros(Float64,testMesh.nCells);
	iFluxV3 = zeros(Float64,testMesh.nCells);
	iFluxV4 = zeros(Float64,testMesh.nCells);
	
	###################################################################################
	## Vectors for CUDA 

	
	SidesV = zeros(Float64,testMesh.nCells*4);
	nxV = zeros(Float64,testMesh.nCells*4);
	nyV = zeros(Float64,testMesh.nCells*4);

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


	## DO NOT USE Mem.UnifiedBuffer!!!!

	#= cuGammaV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)

	curLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuULeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuVLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuPLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)

	curRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuURightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuVRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuPRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells) 


	cuNxV1 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNxV2 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNxV3 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNxV4 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)

    cuNyV1 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNyV2 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNyV3 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuNyV4 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)

    cuSideV1 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuSideV2 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuSideV3 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	cuSideV4 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)

    cuFluxV1 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
    cuFluxV2 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
    cuFluxV3 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells)
    cuFluxV4 = CuVector{Float64, Mem.UnifiedBuffer}(undef,testMesh.nCells) 
	
	cuNeibsV = CuVector{Int32, Mem.UnifiedBuffer}(undef,testMesh.nCells)
	=#


	
	cuNeibsV = CuVector{Int32, Mem.DeviceBuffer}(undef,testMesh.nCells)
	
	curLeftV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuULeftV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuVLeftV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuPLeftV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)

	curRightV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuURightV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuVRightV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
	cuPRightV = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells) 

	cuSideV1234 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuNxV1234 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuNyV1234 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells*4)
	cuFluxV1234 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells*4)

    cuFluxV1 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
    cuFluxV2 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
    cuFluxV3 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)
    cuFluxV4 = CuVector{Float64, Mem.DeviceBuffer}(undef,testMesh.nCells)




	copyto!(cuSideV1234, SidesV)
	copyto!(cuNxV1234, nxV)
	copyto!(cuNyV1234, nyV)
	
	copyto!(cuNeibsV,numNeibs )
	
	
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
	convergenceCriteria= [1e-5;1e-5;1e-5;1e-5;];
	
	
	# debugSaveInit = false;
	# if (debugSaveInit)
	
		# rhoNodes = zeros(Float64,testMesh.nNodes);
		# uxNodes = zeros(Float64,testMesh.nNodes);
		# uyNodes = zeros(Float64,testMesh.nNodes);
		# pNodes = zeros(Float64,testMesh.nNodes);
	
		# cells2nodesSolutionReconstructionWithStencilsImplicitSA(nodesThreadsX, testMeshDistrX, testfields2dX, dummy); 
	
		# for i = 1:testMesh.nNodes
			# rhoNodes[i] = testfields2dX.densityNodes[i];
			# uxNodes[i] = testfields2dX.UxNodes[i];
			# uyNodes[i] = testfields2dX.UyNodes[i];
			# pNodes[i] = testfields2dX.pressureNodes[i];
		# end
		
		# outputfileZero = string(outputfile,"_t=0");
		# println("Saving  solution to  ", outputfileZero);
			# #saveResults2VTK(outputfile, testMesh, densityF);
			# saveResults4VTK(outputfileZero, testMesh, rhoNodes, uxNodes, uyNodes, pNodes);
		# println("done ...  ");	
		
		
		# @save outputfileZero solInst
		
	# end
	
	
	
	maxEdge,id = findmax(testMesh.HX);

	dt::Float64 =  solControls.dt;  
	# @everywhere dtX = $dt; 
	# @everywhere maxEdgeX = $maxEdge; 

	debug = true;	
	useArtViscoistyDapming = true;

	
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
		
			
			if (useArtViscoistyDapming)
			
				calcArtificialViscositySA( cellsThreads, testMesh, thermo, testfields2d, viscfields2d);		
				calcDiffTerm(cellsThreads, testMesh, testfields2d, viscfields2d, thermo, UconsNodesOldX, UConsDiffCellsX, UConsDiffNodesX);
			
			end
	

			if (CUDA.has_cuda() && useCuda)
			

			 	 calcOneStageCUDA(1.0, solControls.dt, dynControls.flowTime, 
				 		testMesh , testfields2d, thermo, cellsThreads,  
				 		UconsCellsOldX, UConsDiffCellsX,  UconsCellsNewX, 
			 	 		uLeft, 
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
			 	## calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);

				Threads.@threads for p in 1:Threads.nthreads()	
					SecondOrderUpwindM2(cellsThreads[p,1], cellsThreads[p,2], 1.0, solControls.dt, dynControls.flowTime,  
						testMesh, testfields2d, thermo, UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);
				end

			end


			
			#doExplicitRK3TVD(1.0, dtX, testMeshDistrX , testfields2dX, thermoX, cellsThreadsX,  UconsCellsOldX, iFLUXX,  UConsDiffCellsX, 
			#  UconsCellsNew1X,UconsCellsNew2X,UconsCellsNew3X,UconsCellsNewX);
			
						
			
			#@sync @distributed for p in workers()
			Threads.@threads for p in 1:Threads.nthreads()			
	
				#beginCell::Int32 = cellsThreads[p,1];
				#endCell::Int32 = cellsThreads[p,2];
				#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);										
				#updateVariablesSA(beginCell, endCell, thermo.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2d);
				updateVariablesSA(cellsThreads[p,1], cellsThreads[p,2], thermo.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2d);
		
			end
			
			
			
			
			 #@sync @distributed for p in workers()	
			 Threads.@threads for p in 1:Threads.nthreads()
	
				#  beginNode::Int32 = nodesThreads[p,1];
				#  endNode::Int32 = nodesThreads[p,2];														
				#  cells2nodesSolutionReconstructionWithStencilsDistributed(beginNode, endNode, 
				# 	testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
				cells2nodesSolutionReconstructionWithStencilsDistributed(nodesThreads[p,1],nodesThreads[p,2],
				 	testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
		
			 end
			
			
			
			
			#cells2nodesSolutionReconstructionWithStencilsSerial(testMeshX,testfields2dX, viscfields2dX, UconsCellsOldX,  UconsNodesOldX);
								
	
			(dynControls.rhoMax,id) = findmax(testfields2d.densityCells);
			(dynControls.rhoMin,id) = findmin(testfields2d.densityCells);
			

			push!(timeVector, dynControls.flowTime); 
			dynControls.curIter += 1; 
			dynControls.verIter += 1;
				
			
			
			
			updateResidualSA(DeltaX, 
				residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax,  
				convergenceCriteria, dynControls);
			
			
			updateOutputSA(timeVector,residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax, 
				testMesh, testfields2d, viscfields2d,  solControls, output, dynControls, solInst);
	
			
			# EVALUATE STAGE:
			
			dynControls.flowTime += dt; 
			##flowTimeX += dt;
			
			# if (solControlsX.timeStepMethod == 1)
				# dynControlsX.flowTime += dynControlsX.tau;  	
			# else
				# dynControlsX.flowTime += solControlsX.dt;  
			# end
			

	

			if (flowTime>= solControls.stopTime || dynControls.isSolutionConverged == 1)
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
			saveResults4VTK(outputfile, testMesh, rhoNodes, uxNodes, uyNodes, pNodes);
			@save outputfile solInst
		println("done ...  ");	
		
		 
		 
		
	#end ## if debug
	
end




##@time godunov2dthreads("2dmixinglayerUp_delta3.bson", numThreads, "2dMixingLayer_delta3", false); 
##@profview godunov2dthreads("2dmixinglayerUp_delta2.bson", numThreads, "2dMixingLayer_delta2", false); 


#godunov2dthreads("cyl2d_supersonic1",  "cyl2d_supersonic1", false); 
#godunov2dthreads("cyl2d_supersonic2",  "cyl2d_supersonic2", false); 
godunov2dthreads("cyl2d_supersonic3",  "cyl2d_supersonic3", false); 



