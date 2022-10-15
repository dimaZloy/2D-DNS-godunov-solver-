




@everywhere function updateResidualSA(
	Delta::Array{Float64,2},
	residualsVector1::Array{Any,1},
	residualsVector2::Array{Any,1},
	residualsVector3::Array{Any,1},
	residualsVector4::Array{Any,1},
	residualsVectorMax::Array{Float64,1},
	convergenceCriteria::Array{Float64,1},
	dynControls::DYNAMICCONTROLS,
	)

	residuals1::Float64   =sum( Delta[:,1].*Delta[:,1] );
	residuals2::Float64  = sum( Delta[:,2].*Delta[:,2] );
	residuals3::Float64  = sum( Delta[:,3].*Delta[:,3] );
	residuals4::Float64  = sum( Delta[:,4].*Delta[:,4] );
	
	push!(residualsVector1, residuals1);
	push!(residualsVector2, residuals2);
	push!(residualsVector3, residuals3);
	push!(residualsVector4, residuals4);
	
	if (dynControls.curIter<6 && dynControls.curIter>1)

   		(residualsVectorMax[1],id1) = findmax(residualsVector1[1:dynControls.curIter]);	
   		(residualsVectorMax[2],id2) = findmax(residualsVector2[1:dynControls.curIter]);		
   		(residualsVectorMax[3],id3) = findmax(residualsVector3[1:dynControls.curIter]);		
   		(residualsVectorMax[4],id4) = findmax(residualsVector4[1:dynControls.curIter]);		

	end

	if ( (dynControls.curIter>5) && 
    	(residualsVector1[dynControls.curIter]./residualsVectorMax[1] <= convergenceCriteria[1]) &&
     	(residualsVector2[dynControls.curIter]./residualsVectorMax[2] <= convergenceCriteria[2]) &&
     	(residualsVector3[dynControls.curIter]./residualsVectorMax[3] <= convergenceCriteria[3]) &&
     	(residualsVector4[dynControls.curIter]./residualsVectorMax[4] <= convergenceCriteria[4]) )

	 	dynControls.isSolutionConverged  = 1; 

	end



end



function updateVariablesSA(
	 cellsThreadsX::Array{Int32,2},
	 Gamma::Float64,
	 UconsCellsNew::Array{Float64,2},
	 UconsCellsOld::Array{Float64,2},
	 Delta::Array{Float64,2},
	 testfields2d::fields2d, 
	 solControls::CONTROLS,
	 dynControls::DYNAMICCONTROLS)


	  if (solControls.densityConstrained==1)

	 	Threads.@threads for p in 1:Threads.nthreads()
	 		for i = cellsThreadsX[p,1]:cellsThreadsX[p,2]
				
	 			if  UconsCellsNewX[i,1] >= solControls.maxDensityConstrained
	 				UconsCellsNewX[i,1] = solControls.maxDensityConstrained;
	 			end		
	 			if  UconsCellsNewX[i,1] <= solControls.minDensityConstrained
	 				UconsCellsNewX[i,1] = solControls.minDensityConstrained;
	 			end		

	 		end
	 	end
	 end

	 (dynControls.rhoMax,id) = findmax(testfields2d.densityCells);
	 (dynControls.rhoMin,id) = findmin(testfields2d.densityCells);


	 Threads.@threads for p in 1:Threads.nthreads()	
	
			for i = cellsThreadsX[p,1]: cellsThreadsX[p,2]

				testfields2d.densityCells[i] = UconsCellsNew[i,1];
				testfields2d.UxCells[i] 	  = UconsCellsNew[i,2]/UconsCellsNew[i,1];
				testfields2d.UyCells[i] 	  = UconsCellsNew[i,3]/UconsCellsNew[i,1];
				testfields2d.pressureCells[i] = (Gamma-1.0)*( UconsCellsNew[i,4] - 0.5*( UconsCellsNew[i,2]*UconsCellsNew[i,2] + UconsCellsNew[i,3]*UconsCellsNew[i,3] )/UconsCellsNew[i,1] );

				testfields2d.aSoundCells[i] = sqrt( Gamma * testfields2d.pressureCells[i]/testfields2d.densityCells[i] );
				testfields2d.VMAXCells[i]  = sqrt( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] ) + testfields2d.aSoundCells[i];
				
				Delta[i,1] = UconsCellsNew[i,1] - UconsCellsOld[i,1];
				Delta[i,2] = UconsCellsNew[i,2] - UconsCellsOld[i,2];
				Delta[i,3] = UconsCellsNew[i,3] - UconsCellsOld[i,3];
				Delta[i,4] = UconsCellsNew[i,4] - UconsCellsOld[i,4];

				
				UconsCellsOld[i,1] = UconsCellsNew[i,1];
				UconsCellsOld[i,2] = UconsCellsNew[i,2];
				UconsCellsOld[i,3] = UconsCellsNew[i,3];
				UconsCellsOld[i,4] = UconsCellsNew[i,4];
	
			end # end i-cell

		end ## end p-thread

 end



# @everywhere function updateVariablesSA(
# 	beginCell::Int32,endCell::Int32,Gamma::Float64,
# 	 UconsCellsNew::Array{Float64,2},
# 	 UconsCellsOld::Array{Float64,2},
# 	 Delta::Array{Float64,2},
# 	 testfields2d::fields2d)
	
# 	for i=beginCell:endCell
	
# 		testfields2d.densityCells[i] = UconsCellsNew[i,1];
# 		testfields2d.UxCells[i] 	  = UconsCellsNew[i,2]/UconsCellsNew[i,1];
# 		testfields2d.UyCells[i] 	  = UconsCellsNew[i,3]/UconsCellsNew[i,1];
# 		testfields2d.pressureCells[i] = (Gamma-1.0)*( UconsCellsNew[i,4] - 0.5*( UconsCellsNew[i,2]*UconsCellsNew[i,2] + UconsCellsNew[i,3]*UconsCellsNew[i,3] )/UconsCellsNew[i,1] );

# 		testfields2d.aSoundCells[i] = sqrt( Gamma * testfields2d.pressureCells[i]/testfields2d.densityCells[i] );
# 		testfields2d.VMAXCells[i]  = sqrt( testfields2d.UxCells[i]*testfields2d.UxCells[i] + testfields2d.UyCells[i]*testfields2d.UyCells[i] ) + testfields2d.aSoundCells[i];
		
# 		Delta[i,1] = UconsCellsNew[i,1] - UconsCellsOld[i,1];
# 		Delta[i,2] = UconsCellsNew[i,2] - UconsCellsOld[i,2];
# 		Delta[i,3] = UconsCellsNew[i,3] - UconsCellsOld[i,3];
# 		Delta[i,4] = UconsCellsNew[i,4] - UconsCellsOld[i,4];

		
# 		UconsCellsOld[i,1] = UconsCellsNew[i,1];
# 		UconsCellsOld[i,2] = UconsCellsNew[i,2];
# 		UconsCellsOld[i,3] = UconsCellsNew[i,3];
# 		UconsCellsOld[i,4] = UconsCellsNew[i,4];
		
# 	end
	
#  end



@everywhere function updateOutputSA(
	timeVector::Array{Any,1},
	residualsVector1::Array{Any,1},
	residualsVector2::Array{Any,1},
	residualsVector3::Array{Any,1},
	residualsVector4::Array{Any,1},
	residualsVectorMax::Array{Float64,1}, 
	testMesh::mesh2d_Int32,
	testFields::fields2d,
	testFieldsViscous::viscousFields2d,
	solControls::CONTROLS,
	output::outputCONTROLS,
	dynControls::DYNAMICCONTROLS,
	solInst::solutionCellsT,
	thermo::THERMOPHYSICS,
	kVectorTheory::Array{Any,1},
	kVectorApprox::Array{Any,1})

	if (dynControls.verIter == output.verbosity)


		densityWarn = @sprintf("Density Min/Max: %f/%f", dynControls.rhoMin, dynControls.rhoMax);
		out = @sprintf("%0.10f\t %0.10f \t %0.6f \t %0.6f \t %0.6f \t %0.6f \t %0.6f", 
			dynControls.flowTime,
			dynControls.tau,
			residualsVector1[dynControls.curIter]./residualsVectorMax[1],
			residualsVector2[dynControls.curIter]./residualsVectorMax[2],
			residualsVector3[dynControls.curIter]./residualsVectorMax[3],
			residualsVector4[dynControls.curIter]./residualsVectorMax[4],
			dynControls.cpuTime
			 );
		#outputS = string(output, cpuTime);
		#println(outputS); 
		println(out); 
		println(densityWarn);
		
		
		
		
		if (output.saveDataToVTK == 1)
		
		
		
			solInst.dt = solControls.dt;
			solInst.flowTime = dynControls.flowTime;
			for i = 1 : solInst.nCells
				solInst.densityCells[i] = testFields.densityCells[i];
				solInst.UxCells[i] = testFields.UxCells[i];
				solInst.UyCells[i] = testFields.UyCells[i];
				solInst.pressureCells[i] = testFields.pressureCells[i];
			end
		
			filename = string("zzz",dynControls.curIter+1000); 	
			saveResults2VTK(filename, testMesh, testFields.densityNodes, "density");
			@save filename solInst
			
		end
		 
			
	
		if (solControls.plotResidual == 1)	



			TGV2dDensityNodes  = ones(Float64,testMesh.nNodes); 
			TGV2dUMagNodes  = zeros(Float64,testMesh.nNodes); 
			TGV2dPressureNodes  = zeros(Float64,testMesh.nNodes); 

			dummy4 = zeros(Float64,4);

			maxP::Float64 = 100000.0;
			minP::Float64 = 100000.0;

			for i=1:testMesh.nNodes

				computeTGV2d(testMesh.xNodes[i], testMesh.yNodes[i], dynControls.flowTime, thermo, dummy4);

				TGV2dDensityNodes[i] 	= dummy4[1];
				TGV2dUMagNodes[i] 		= sqrt(dummy4[2]*dummy4[2] + dummy4[3]*dummy4[3]);
				TGV2dPressureNodes[i] 	= dummy4[4];
				
				# if TGV2dPressureNodes[i] >= maxP
				# 	maxP = TGV2dPressureNodes[i];
				# end

				# if TGV2dPressureNodes[i] <  minP
				# 	minP = TGV2dPressureNodes[i];
				# end

						
			end

			
			subplot(3,3,2);	
			cla();
			
			##tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, densityF,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, testFields.densityNodes);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of density");
			axis("equal");

			subplot(3,3,5);	
			cla();
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, testFields.pressureNodes);#, vmin=minP,vmax=maxP);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of pressure");
			axis("equal");
			

			subplot(3,3,8);	
			cla();
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, sqrt.(testFields.UxNodes.*testFields.UxNodes .+ testFields.UyNodes.*testFields.UyNodes) );
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of Umag");
			axis("equal");
			
			#############################################################################################

			subplot(3,3,1);	
			cla();
			
			##tricontourf(testMesh.xNodes,testMesh.yNodes, triangles, densityF,pControls.nContours,vmin=pControls.rhoMINcont,vmax=pControls.rhoMAXcont);
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, TGV2dDensityNodes);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of density");
			axis("equal");

			subplot(3,3,4);	
			cla();
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, TGV2dPressureNodes);#, vmin=minP,vmax=maxP);
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of pressure");
			axis("equal");
			

			subplot(3,3,7);	
			cla();
			
			tricontourf(testMesh.xNodes,testMesh.yNodes, testMesh.triangles, TGV2dUMagNodes );
			
			set_cmap("jet");
			xlabel("x");
			ylabel("y");
			title("Contours of Umag");
			axis("equal");
			
			
			subplot(3,3,3);
			 cla();
			
			 if (size(timeVector,1) >1)
			 	plot(timeVector, kVectorTheory,"-r",label="EK/E0 theory"); 
			 	plot(timeVector, kVectorApprox,"-g",label="EK/E0 approx"); 
			# 	plot(timeVector, residualsVector3./residualsVectorMax[3],"-b",label="momentum uy"); 
			# 	plot(timeVector, residualsVector4./residualsVectorMax[4],"-c",label="energy"); 
			 end
			
			 yscale("log");	
			 xlabel("flow time [s]");
			 ylabel("E/E0");
			# title("Residuals");
			 legend();
			 grid();
			
			pause(1.0e-5);
			
			
		end

   

		#pause(1.0e-3);
		dynControls.verIter = 0; 

	end



end

