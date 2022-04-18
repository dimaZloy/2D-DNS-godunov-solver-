



@everywhere @inline function calcArtificialViscositySA( cellsThreadsX::Array{Int32,2}, testMesh::mesh2d_Int32, 
    thermoX::THERMOPHYSICS, testfields2d::fields2d, viscfields2dX::viscousFields2d)
	
	Threads.@threads for p in 1:Threads.nthreads()
	
		beginCell::Int32 = cellsThreadsX[p,1];
		endCell::Int32 = cellsThreadsX[p,2];
						
		nodesGradientReconstructionFastPerThread22(beginCell, endCell, testMesh, testfields2d.UxNodes, viscfields2dX.dUdxCells,viscfields2dX.dUdyCells);
		nodesGradientReconstructionFastPerThread22(beginCell, endCell, testMesh, testfields2d.UyNodes, viscfields2dX.dVdxCells,viscfields2dX.dVdyCells);

		# nodesGradientReconstructionFastPerThread33(beginCell, endCell, testMesh, testfields2d.UxNodes, viscfields2dX.dUdxCells,viscfields2dX.dUdyCells);
		# nodesGradientReconstructionFastPerThread33(beginCell, endCell, testMesh, testfields2d.UyNodes, viscfields2dX.dVdxCells,viscfields2dX.dVdyCells);
				
		calcDynamicViscosityPerThread( beginCell, endCell, testMesh, testfields2d, viscfields2dX, thermoX.RGAS);
		#calcArtificialViscosityPerThread( beginCell, endCell, testMesh, testfields2d, viscfields2dX);
					
	end
	
	
end



@everywhere @inline function calcDynamicViscosityPerThread( beginCell::Int32, endCell::Int32, 
	testMesh::mesh2d_Int32, testfields2d::fields2d, viscfields2dX::viscousFields2d, Rgas::Float64)

	 
     for i = beginCell:endCell
	 
		#T::Float64 = testfields2d.pressureCells[i]/testfields2d.densityCells[i]/Rgas;     
		#viscfields2dX.artViscosityCells[i] = calcSutherlandViscosityOF(T);
		
		viscfields2dX.artViscosityCells[i] = calcSutherlandViscosityOF(testfields2d.temperatureCells[i]); 
		
		
		
     end ## for	

end


@everywhere @inline function calcArtificialViscosityPerThread( beginCell::Int32, endCell::Int32, 
	testMesh::mesh2d_Int32, testfields2d::fields2d, viscfields2dX::viscousFields2d)


	 T::Float64 = 0.0;
	 
     for i = beginCell:endCell
	 
		
		divU  = viscfields2dX.dUdxCells[i] + viscfields2dX.dVdyCells[i];
	
		a  = 0.05*testfields2d.aSoundCells[i]/testMesh.HX[i]/sqrt(2.0); 
			
         if (-divU >  a)
         
             ##tmp = divU[i]*divU[i] - (Cth*aSound[i]/h)*(Cth*aSound[i]/h);
			 tmp = divU*divU - a*a;			 
			 
             if (tmp > 0.0)
				    ##Cth::Float64 = 0.05;
					##Cav::Float64 = 0.5;
					##PSI::Float64 = 1.0e-5;
					##viscfields2dX.artViscosityCells[i] = Cav*testfields2d.densityCells[i]*h*h*sqrt(tmp)*PSI;
				viscfields2dX.artViscosityCells[i] = 0.5*testfields2d.densityCells[i]*testMesh.HX[i]/sqrt(2.0)*testMesh.HX[i]/sqrt(2.0)*sqrt(tmp)*1.0e-5;
			 end

         end 


     end ## for
	 

end