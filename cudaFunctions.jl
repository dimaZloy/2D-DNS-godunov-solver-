

function c_P_m(M::Float64,AUSM_ALFA::Float64)::Float64	
	return (CUDA.abs(M)>=1.0) ? 0.5*(1.0-CUDA.sign(M)) : c_Palfa_m(M,AUSM_ALFA)
end

function c_P_p(M::Float64,AUSM_ALFA::Float64)::Float64
	return (CUDA.abs(M)>=1.0) ? 0.5*(1.0+CUDA.sign(M)) : c_Palfa_p(M,AUSM_ALFA)
end

function c_Palfa_m(M::Float64,AUSM_ALFA::Float64)::Float64
	return  0.25*(M-1.0)*(M-1.0)*(2.0+M)-AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

function c_Palfa_p(M::Float64,AUSM_ALFA::Float64)::Float64
	return  0.25*(M+1.0)*(M+1.0)*(2.0-M)+AUSM_ALFA*M*(M*M-1.0)*(M*M-1.0);
end

function c_Mbetta_p(M::Float64,AUSM_BETTA::Float64)::Float64
	return  0.25*(M+1.0)*(M+1.0)+AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

function c_Mbetta_m(M::Float64,AUSM_BETTA::Float64)::Float64
	return -0.25*(M-1.0)*(M-1.0)-AUSM_BETTA*(M*M-1.0)*(M*M-1.0);
end

function c_M_p(M::Float64,AUSM_BETTA::Float64)::Float64
	return (CUDA.abs(M)>=1.0) ? 0.5*(M+CUDA.abs(M)) : c_Mbetta_p(M,AUSM_BETTA)
end

function c_M_m(M::Float64,AUSM_BETTA::Float64)::Float64
	return (CUDA.abs(M)>=1.0) ? 0.5*(M-CUDA.abs(M)) : c_Mbetta_m(M,AUSM_BETTA)
end

function calcAUSMfluxCUDA(rhoL, _UL, _VL, PL, rhoR, _UR, _VR, PR, nx, ny, side, gamma, flux, 
					htL,htR,htN,aL_tilda,aR_tilda,a12,m_dot12,p12)
	
	i = threadIdx().x
	
	
	htL[i] =  PL[i]/rhoL[i]/(gamma-1.0) + 0.5*(_UL[i]*_UL[i] + _VL[i]*_VL[i]) +  PL[i]/rhoL[i]; 
	htR[i] =  PR[i]/rhoR[i]/(gamma-1.0) + 0.5*(_UR[i]*_UR[i] + _VR[i]*_VR[i]) +  PR[i]/rhoR[i];
	
	htN[i]  = 0.5*(htL[i] + htR[i] - 0.5*((_UL[i]*ny[i] - _VL[i]*nx[i])*(_UL[i]*ny[i] - _VL[i]*nx[i]) + (_UR[i]*ny[i] - _VR[i]*nx[i])*(_UR[i]*ny[i] - _VR[i]*nx[i])));
	
	aL_tilda[i] = CUDA.sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htL[i]) * CUDA.min(1.0, (CUDA.sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htL[i]))/CUDA.abs( _UL[i]*nx[i] + _VL[i]*ny[i] ));
	aR_tilda[i] = CUDA.sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htR[i]) * CUDA.min(1.0, (CUDA.sqrt(2.0*( gamma-1.0)/(gamma+1.0)*htR[i]))/CUDA.abs( _UR[i]*nx[i] + _VR[i]*ny[i] ));
	a12[i] = CUDA.min(aL_tilda[i],aR_tilda[i]);

	m_dot12[i] = c_M_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12[i],1.0/8.0)     + c_M_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12[i],1.0/8.0);
	p12[i]     = c_P_p( (_UL[i]*nx[i] + _VL[i]*ny[i])/a12[i],3.0/16.0)*PL[i]  + c_P_m( (_UR[i]*nx[i] + _VR[i]*ny[i])/a12[i],3.0/16.0)*PR[i];

	
	flux[i,1] = -( a12[i]*( 0.5*(m_dot12[i] + CUDA.abs(m_dot12[i]))*rhoL[i]         + 0.5*(m_dot12[i]-CUDA.abs(m_dot12[i]))*rhoR[i]       ) + 0.0          )*side[i];
	flux[i,2] = -( a12[i]*( 0.5*(m_dot12[i] + CUDA.abs(m_dot12[i]))*rhoL[i]*_UL[i]  + 0.5*(m_dot12[i]-CUDA.abs(m_dot12[i]))*rhoR[i]*_UR[i]) + p12[i]*nx[i] )*side[i];
	flux[i,3] = -( a12[i]*( 0.5*(m_dot12[i] + CUDA.abs(m_dot12[i]))*rhoL[i]*_VL[i]  + 0.5*(m_dot12[i]-CUDA.abs(m_dot12[i]))*rhoR[i]*_VR[i]) + p12[i]*ny[i] )*side[i];
	flux[i,4] = -( a12[i]*( 0.5*(m_dot12[i] + CUDA.abs(m_dot12[i]))*rhoL[i]*htL[i]  + 0.5*(m_dot12[i]-CUDA.abs(m_dot12[i]))*rhoR[i]*htR[i]) + 0.0          )*side[i];
	
    return
end




@everywhere function c_computeInterfaceDataIndex(i::Int32, k::Int32,  testMesh::mesh2d_Int32, testFields::fields2d, thermo::THERMOPHYSICS, 
	uLeftp::Array{Float64,1}, uUpp::Array{Float64,1},uDownp::Array{Float64,1}, uRightp::Array{Float64,1},  
	flowTime::Float64, UpLeft::Array{Float64,3}, UpRight::Array{Float64,3})
	
	## i - local  index for cell
	## k - local index for neib cell 

	##nCells = size(testMesh.cell_stiffness,1);
	ek::Int32 = testMesh.cell_stiffness[i,k]; ##; %% get right cell 
	
	ek_type::Int32 = testMesh.mesh_connectivity[i,2];
	
	side::Float64 = testMesh.cell_edges_length[i,k];
	nx::Float64   = testMesh.cell_edges_Nx[i,k];
	ny::Float64   = testMesh.cell_edges_Ny[i,k];
				

	uUpp[1] = uUpp[2] = uUpp[3] = uUpp[4] = 0.0;
	uDownp[1] = uDownp[2] = uDownp[3] = uDownp[4] = 0.0;
	uRightp[1] = uRightp[2] = uRightp[3] = uRightp[4] = 0.0;
		
	index::Int32 = 0;
	if (k == 1)
		index = 1;
	elseif (k == 2)
		index = 3;
	elseif (k == 3)
		index = 5;
	elseif (k == 4)
		index = 7;	
	end
				
	pDown1::Int64 = 0;
	pDown2::Int64 = 0;
	
	pUp1::Int64 = 0;
	pUp2::Int64 = 0;
	
	
	if (ek >=1 && ek<=testMesh.nCells)
								   
								   
		if (ek_type == 3) ## tri element 
		
			pDown1 = testMesh.node2cellsL2down[i,index];
			pUp1 = testMesh.node2cellsL2up[i,index];		
					
			uUpp[1] = testFields.densityNodes[pUp1];
			uUpp[2] = testFields.UxNodes[pUp1];
			uUpp[3] = testFields.UyNodes[pUp1];
			uUpp[4] = testFields.pressureNodes[pUp1];
					
			uDownp[1] = testFields.densityNodes[pDown1];
			uDownp[2] = testFields.UxNodes[pDown1];
			uDownp[3] = testFields.UyNodes[pDown1];
			uDownp[4] = testFields.pressureNodes[pDown1];
		
		elseif (ek_type == 2) ## quad element 
		
			pDown1 = testMesh.node2cellsL2down[i,index];
			pDown2 = testMesh.node2cellsL2down[i,index+1];
			
			pUp1 = testMesh.node2cellsL2up[i,index];		
			pUp2 = testMesh.node2cellsL2up[i,index+1];		
		
			uUpp[1] = 0.5*(testFields.densityNodes[pUp1]  + testFields.densityNodes[pUp2]);
			uUpp[2] = 0.5*(testFields.UxNodes[pUp1]       + testFields.UxNodes[pUp2]);
			uUpp[3] = 0.5*(testFields.UyNodes[pUp1]       + testFields.UyNodes[pUp2]);
			uUpp[4] = 0.5*(testFields.pressureNodes[pUp1] + testFields.pressureNodes[pUp2]);
					
			uDownp[1] = 0.5*(testFields.densityNodes[pDown1]  + testFields.densityNodes[pDown2]);
			uDownp[2] = 0.5*(testFields.UxNodes[pDown1]       + testFields.UxNodes[pDown2]);
			uDownp[3] = 0.5*(testFields.UyNodes[pDown1]       + testFields.UyNodes[pDown2]);
			uDownp[4] = 0.5*(testFields.pressureNodes[pDown1] + testFields.pressureNodes[pDown2]);
		
		
		end
		

		uRightp[1] = testFields.densityCells[ek];
		uRightp[2] = testFields.UxCells[ek];
		uRightp[3] = testFields.UyCells[ek];
		uRightp[4] = testFields.pressureCells[ek];					
					
	else
					
		##yc::Float64 = testMesh.cell_mid_points[i,2]; 
		##uRightp = ComputeUPhysFromBoundaries(i,k, ek, uLeftp, nx,ny, yc, thermo.Gamma, flowTime );
		
		ComputeUPhysFromBoundaries(i,k, ek, uLeftp, nx,ny, testMesh.cell_mid_points[i,2], thermo.Gamma, flowTime ,uRightp);
					
		
		uDownp[1] = uLeftp[1];
		uDownp[2] = uLeftp[2];
		uDownp[3] = uLeftp[3];
		uDownp[4] = uLeftp[4];
		
		uUpp[1] = uRightp[1];
		uUpp[2] = uRightp[2];
		uUpp[3] = uRightp[3];
		uUpp[4] = uRightp[4];
	
	
	end
				
				
	#ksi::Float64 = 1.0e-6;			
	#UpRight = zeros(Float64,4);
	#UpLeft = zeros(Float64,4);

	
	UpLeft[i,1,k]  = uLeftp[1] + 0.5*Minmod_Limiter( uLeftp[1]  - uDownp[1], uRightp[1] - uLeftp[1], 1.0e-6);
	UpLeft[i,2,k]  = uLeftp[2] + 0.5*Minmod_Limiter( uLeftp[2]  - uDownp[2], uRightp[2] - uLeftp[2], 1.0e-6);
	UpLeft[i,3,k]  = uLeftp[3] + 0.5*Minmod_Limiter( uLeftp[3]  - uDownp[3], uRightp[3] - uLeftp[3], 1.0e-6);
	UpLeft[i,4,k]  = uLeftp[4] + 0.5*Minmod_Limiter( uLeftp[4]  - uDownp[4], uRightp[4] - uLeftp[4], 1.0e-6);
					
	UpRight[i,1,k] = uRightp[1] - 0.5*Minmod_Limiter( uRightp[1] - uLeftp[1], uUpp[1]  - uRightp[1],  1.0e-6);
	UpRight[i,2,k] = uRightp[2] - 0.5*Minmod_Limiter( uRightp[2] - uLeftp[2], uUpp[2]  - uRightp[2],  1.0e-6);	
	UpRight[i,3,k] = uRightp[3] - 0.5*Minmod_Limiter( uRightp[3] - uLeftp[3], uUpp[3]  - uRightp[3],  1.0e-6);
	UpRight[i,4,k] = uRightp[4] - 0.5*Minmod_Limiter( uRightp[4] - uLeftp[4], uUpp[4]  - uRightp[4],  1.0e-6);
	
						
	#AUSMplusFlux2d(UpRight,UpLeft, nx,ny,side,thermo.Gamma);	
	#AUSMplusFlux2dFast(UpRight[1],UpRight[2],UpRight[3],UpRight[4],UpLeft[1],UpLeft[2],UpLeft[3],UpLeft[4], nx,ny,side,thermo.Gamma, flux);	
	
	
	


end




@everywhere function c_computeInterfaceData(
	beginCell::Int32,endCell::Int32, bettaKJ::Float64, dt::Float64, flowTime::Float64, 
	testMesh::mesh2d_Int32, testFields::fields2d, thermo::THERMOPHYSICS, UpLeft::Array{Float64,3}, UpRight::Array{Float64,3})

	
	uLeftp = zeros(Float64,4);
	uUpp = zeros(Float64,4);
	uDownp = zeros(Float64,4);
	uRightp = zeros(Float64,4);
		
	
	for i = beginCell:endCell
    		
		
		uLeftp[1] = testFields.densityCells[i];
		uLeftp[2] = testFields.UxCells[i];
		uLeftp[3] = testFields.UyCells[i];
		uLeftp[4] = testFields.pressureCells[i];
			   
		
		if (testMesh.mesh_connectivity[i,3] == 3)
		
		
			
			computeInterfaceSlopeIndex(i, Int32(1), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			computeInterfaceSlopeIndex(i, Int32(2), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			computeInterfaceSlopeIndex(i, Int32(3), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
	
						

		elseif (testMesh.mesh_connectivity[i,3] == 4)
			
			
			computeInterfaceSlopeIndex(i, Int32(1), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			computeInterfaceSlopeIndex(i, Int32(2), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			computeInterfaceSlopeIndex(i, Int32(3), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			computeInterfaceSlopeIndex(i, Int32(4), testMesh, testFields, thermo, uLeftp, uUpp, uDownp, uRightp, flowTime, UpLeft, UpRight) ;
			
			

		else
			
			display("something wrong in flux calculations ... ")
			
		end
		  
   
	end # i - loop for all cells


end

