

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

include("primeObjects.jl");
include("thermo.jl"); #setup thermodynamics
include("utilsIO.jl");
#include("RoeFlux2d.jl")

include("AUSMflux2dFast.jl"); #AUSM+ inviscid flux calculation 

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


	flag2loadPreviousResults = true;

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
	
	
	phs2dcns2dcellsSA(UconsCellsOldX,testfields2d, thermo.Gamma);	
	phs2dcns2dcellsSA(UconsCellsNewX,testfields2d, thermo.Gamma);	
	
	
	cells2nodesSolutionReconstructionWithStencilsUCons(nodesThreads, testMesh, UconsCellsOldX,  UconsNodesOldX );	
	

	
	# testMeshDistr = createMesh2dShared(testMesh);
	
	
	#@everywhere trianglesX = $triangles;
	#@everywhere testMeshDistrX = $testMeshDistr; 
	# @everywhere testMeshX = $testMesh; 
	# @everywhere thermoX   = $thermo;
	# @everywhere cellsThreadsX = $cellsThreads;
	# @everywhere nodesThreadsX = $nodesThreads;
	# @everywhere testfields2dX  = $testfields2d;
	# @everywhere viscfields2dX  = $viscfields2d;
		
	# @everywhere dynControlsX = $dynControls;
	# @everywhere solControlsX = $solControls;
	# @everywhere pControlsX = $pControls;
	# @everywhere outputX = $output;

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
	
	
		for l = 1:3
		#while (dynControls.isRunSimulation == 1)
		
			
			##CPUtic();	
			start = time();
			
			
			# PROPAGATE STAGE: 
			(dynControls.velmax,id) = findmax(testfields2d.VMAXCells);
			# #dynControls.tau = solControls.CFL * testMesh.maxEdgeLength/(max(dynControls.velmax,1.0e-6)); !!!!
			dynControls.tau = solControls.CFL * maxEdge/(max(dynControls.velmax,1.0e-6));
		
			
			if (useArtViscoistyDapming)
			
				@time calcArtificialViscositySA( cellsThreads, testMesh, testfields2d, viscfields2d);
					
				@time calcDiffTerm(cellsThreads, testMesh, testfields2d, viscfields2d, thermo, UconsNodesOldX, UConsDiffCellsX, UConsDiffNodesX);
			
			end
	
				
			## Explicit Euler first-order	
			@time  calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);
			
			#doExplicitRK3TVD(1.0, dtX, testMeshDistrX , testfields2dX, thermoX, cellsThreadsX,  UconsCellsOldX, iFLUXX,  UConsDiffCellsX, 
			#  UconsCellsNew1X,UconsCellsNew2X,UconsCellsNew3X,UconsCellsNewX);
			
						
			
			#@sync @distributed for p in workers()
			@time Threads.@threads for p in 1:Threads.nthreads()			
	
				beginCell::Int32 = cellsThreads[p,1];
				endCell::Int32 = cellsThreads[p,2];
				#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
														
				updateVariablesSA(beginCell, endCell, thermo.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2d);
		
			end
			
			#@everywhere finalize(updateVariablesSA);	
			
			
			 #@sync @distributed for p in workers()	
			 @time Threads.@threads for p in 1:Threads.nthreads()
	
				 beginNode::Int32 = nodesThreads[p,1];
				 endNode::Int32 = nodesThreads[p,2];
				
														
				 cells2nodesSolutionReconstructionWithStencilsDistributed(beginNode, endNode, 
					testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
		
			 end
			
			# @everywhere finalize(cells2nodesSolutionReconstructionWithStencilsDistributed);	
			
			
			
			#cells2nodesSolutionReconstructionWithStencilsSerial(testMeshX,testfields2dX, viscfields2dX, UconsCellsOldX,  UconsNodesOldX);
								
	
			(dynControls.rhoMax,id) = findmax(testfields2d.densityCells);
			(dynControls.rhoMin,id) = findmin(testfields2d.densityCells);
			

			push!(timeVector, dynControls.flowTime); 
			dynControls.curIter += 1; 
			dynControls.verIter += 1;
				
			
			
			
			@time updateResidualSA(DeltaX, 
				residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax,  
				convergenceCriteria, dynControls);
			
			
			@time updateOutputSA(timeVector,residualsVector1,residualsVector2,residualsVector3,residualsVector4, residualsVectorMax, 
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


##godunov2dthreads("testStep2dBaseTriSmooth",  "testStep2dBaseTriSmooth", false); 
godunov2dthreads("2dmixinglayerUp_delta3.bson", "2dMixingLayer_delta3", false); 



