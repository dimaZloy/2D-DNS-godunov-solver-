

using Distributed;
using PyPlot;

using WriteVTK;
using CPUTime;
using DelimitedFiles;
using Printf
using Dierckx
using BSON: @load
using BSON: @save
using SharedArrays;

using HDF5;
using ProfileView;


include("primeObjects.jl");
include("sutherland.jl"); # calc air viscosity using Sutherland law
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

#include("bcInviscidWall.jl"); ## depricated
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


	flag2loadPreviousResults = false;

	testMesh = readMesh2dHDF5(pname);
	
	
	#display(testMesh.bc_data)
	
	# figure(222)
	# clf();
	# for i = 1:size(testMesh.bc_indexes,1)

		# idb = testMesh.bc_data[i,1]; ## cell id 
		# idv1 = testMesh.bc_data[i,2]; ##  cell type		
		# idv2 = testMesh.bc_data[i,3]; ## node id
					
		# p2 = testMesh.mesh_connectivity[idb,3+idv2];
	
		# if (testMesh.bc_indexes[i] == -4); 
			# plot(testMesh.xNodes[p2],testMesh.yNodes[p2],"ok");			

		# elseif (testMesh.bc_indexes[i] == -3); 
					
			# plot(testMesh.xNodes[p2],testMesh.yNodes[p2],"og");
			
		# elseif (testMesh.bc_indexes[i] == -2); 
			# plot(testMesh.xNodes[p2],testMesh.yNodes[p2],"or");			
			
		# elseif (testMesh.bc_indexes[i] == -1); 
			# plot(testMesh.xNodes[p2],testMesh.yNodes[p2],"ob");			
			
		# end
	# end
	
	#pause(10000)
	
	
	# display(testMesh.cells2nodes)
	# display( findall(x->x==0, testMesh.cells2nodes) )
	
		
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
	
	UconsCellsNewTmpX1 = zeros(Float64,testMesh.nCells,4);
	UconsCellsNewTmpX2 = zeros(Float64,testMesh.nCells,4);
	UconsCellsNewTmpX3 = zeros(Float64,testMesh.nCells,4);
		
	UConsDiffCellsX = zeros(Float64,testMesh.nCells,4);
	UConsDiffNodesX = zeros(Float64,testMesh.nNodes,4);
	
	DeltaX = zeros(Float64,testMesh.nCells,4);
	iFLUXX  = zeros(Float64,testMesh.nCells,4);
	dummy  = zeros(Float64,testMesh.nNodes,4);
	
	#localTau = zeros(Float64,testMesh.nCells,1);
	

	localDampNodes  = zeros(Float64,testMesh.nNodes);
	localDampCells  = zeros(Float64,testMesh.nCells);

	for i = 1:testMesh.nCells
		##rad::Float64 = sqrt(testMesh.cell_mid_points[i,1]*testMesh.cell_mid_points[i,1] + testMesh.cell_mid_points[i,2]*testMesh.cell_mid_points[i,2]);
		##localDampCells[i] = 1.0 - exp(  -(rad/25.0)^3.0); 
		localDampCells[i] = 1.0;
		
	end

	
	for i = 1:testMesh.nNodes
		
		#rad::Float64 = sqrt(testMesh.xNodes[i]*testMesh.xNodes[i] + testMesh.yNodes[i]*testMesh.yNodes[i]);
		#localDampNodes[i] = 1.0 - exp(  -(rad/25.0)^3.0); 
		localDampNodes[i] = 1.0;
	end
	
	
	
	
	
	phs2dcns2dcellsSA(UconsCellsOldX,testfields2d, thermo.Gamma);	
	phs2dcns2dcellsSA(UconsCellsNewX,testfields2d, thermo.Gamma);	
	
	
	#cells2nodesSolutionReconstructionWithStencilsUCons(nodesThreads, testMesh, UconsCellsOldX,  UconsNodesOldX );	

	#@sync @distributed for p in workers()	
	Threads.@threads for p in 1:Threads.nthreads()
	
		 beginNode::Int32 = nodesThreads[p,1];
		 endNode::Int32 = nodesThreads[p,2];
																	
		 cells2nodesSolutionReconstructionWithStencils(beginNode, endNode, 
			testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
		
	 end

	

	figure(101)
	

	timeVector = [];
	residualsVector1 = []; 
	residualsVector2 = []; 
	residualsVector3 = []; 
	residualsVector4 = []; 
	residualsVectorMax = ones(Float64,4);
	convergenceCriteria= [1e-10;1e-10;1e-10;1e-10;];
	
	
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
	
	
	
	#maxEdge,id = findmax(testMesh.HX);
	
	# dynControls.tau = 0.0;
	
	for i = 1:testMesh.nCells
		
		# localTau[i] = solControls.CFL*testMesh.HX[i]*testMesh.HX[i]/( *testMesh.HX[i]*testfields2d.VMAXCells[i] + 2.0*thermoX.Gamma/thermoX.Pr * viscfields2d.artViscosityCells[i]/testfields2d.densityCells[i]);
		
		testfields2d.localCFLCells[i] = min(solControls.CFL*0.5*testMesh.HX[i]/testfields2d.VMAXCells[i], 
								testMesh.HX[i]*testMesh.HX[i]*0.25*testfields2d.densityCells[i]/viscfields2d.artViscosityCells[i]);
	
	end	
	
	
	#minTimeStep,id = findmax(testfields2d.localCFLCells);
	#dynControls.tau = minTimeStep;

	dt::Float64 =  solControls.dt;  
	# @everywhere dtX = $dt; 
	# @everywhere maxEdgeX = $maxEdge; 

	@everywhere localDampCellsX = $localDampCells;

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
			#(dynControls.velmax,id) = findmax(testfields2d.VMAXCells);
			# #dynControls.tau = solControls.CFL * testMesh.maxEdgeLength/(max(dynControls.velmax,1.0e-6)); !!!!
			#dynControls.tau = solControls.CFL * maxEdge/(max(dynControls.velmax,1.0e-6));
			
			(dynControls.tau,id) = findmin(testfields2d.localCFLCells);
		
			
			if (useArtViscoistyDapming)
			
				calcArtificialViscositySA( cellsThreads, testMesh, thermo, testfields2d, viscfields2d);					
				calcDiffTerm(cellsThreads, testMesh, testfields2d, viscfields2d, thermo, UconsNodesOldX, UConsDiffCellsX, UConsDiffNodesX, localDampCellsX);
			
			end
	
				
			## Explicit Euler first-order	
			calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);
					
			## RK3-TVD
			# calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsOldX, iFLUXX, UConsDiffCellsX,  UconsCellsNewTmpX1);
			# calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsNewTmpX1, iFLUXX, UConsDiffCellsX,  UconsCellsNewTmpX2);
			
			# Threads.@threads for p in 1:Threads.nthreads()			
	
				# beginCell::Int32 = cellsThreads[p,1];
				# endCell::Int32 = cellsThreads[p,2];
				# #println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
				
				# for i=beginCell:endCell
					# UconsCellsNewTmpX3[i,1] = 0.75*UconsCellsOldX[i,1]+0.25*UconsCellsNewTmpX2[i,1];
					# UconsCellsNewTmpX3[i,2] = 0.75*UconsCellsOldX[i,2]+0.25*UconsCellsNewTmpX2[i,2];
					# UconsCellsNewTmpX3[i,3] = 0.75*UconsCellsOldX[i,3]+0.25*UconsCellsNewTmpX2[i,3];
					# UconsCellsNewTmpX3[i,4] = 0.75*UconsCellsOldX[i,4]+0.25*UconsCellsNewTmpX2[i,4];
				# end
		
			# end
						
			# calcOneStage(1.0, solControls.dt, dynControls.flowTime, testMesh , testfields2d, thermo, cellsThreads,  UconsCellsNewTmpX3, iFLUXX, UConsDiffCellsX,  UconsCellsNewX);
			# end RK3-TVD
			
			#@sync @distributed for p in workers()
			Threads.@threads for p in 1:Threads.nthreads()			
	
				beginCell::Int32 = cellsThreads[p,1];
				endCell::Int32 = cellsThreads[p,2];
				#println("worker: ",p,"\tbegin cell: ",beginCell,"\tend cell: ", endCell);
														
				#updateVariablesSA(beginCell, endCell, thermo.Gamma,  UconsCellsNewX, UconsCellsOldX, DeltaX, testfields2d);
				updateVariablesSA(beginCell, endCell, solControls.CFL,  thermo,  UconsCellsNewX, UconsCellsOldX, DeltaX, testMesh, testfields2d, viscfields2d);
		
			end
			
			
			
			
			 #@sync @distributed for p in workers()	
			 Threads.@threads for p in 1:Threads.nthreads()
	
				 beginNode::Int32 = nodesThreads[p,1];
				 endNode::Int32 = nodesThreads[p,2];
				
														
				 cells2nodesSolutionReconstructionWithStencils(beginNode, endNode, 
					testMesh, testfields2d, viscfields2d, UconsCellsOldX,  UconsNodesOldX);
		
			 end
			
		
			for i = 1:size(testMesh.bc_indexes,1)
				if (testMesh.bc_indexes[i] == -3); 
					
					idb = testMesh.bc_data[i,1]; ## cell id 
					idv1 = testMesh.bc_data[i,2]; ##  cell type		
					idv2 = testMesh.bc_data[i,3]; ## node id
	
					
					p2 = testMesh.mesh_connectivity[idb,3+idv2];
				
					UconsNodesOldX[p2,1] = 0.0;
					UconsNodesOldX[p2,2] = 0.0;
					UconsNodesOldX[p2,3] = 0.0;
					UconsNodesOldX[p2,4] = 0.0;
				
					
				end
			end
								
	
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
#godunov2dthreads("cyl2d_supersonic1_BL",  "cyl2d_supersonic1_BL", false); 
#godunov2dthreads("cyl2d_supersonic1_BL_test",  "cyl2d_supersonic1_BL_test", false); 

godunov2dthreads("oblickShock2dl00F1n",  "oblickShock2dl00F1n", false); 




