
@everywhere function distibuteCellsInThreadsSA(nThreads::Int64, nCells::Int64 )::Array{Int32,2}

	cellsThreads = zeros(Int32,nThreads,2);
 
 	if (nThreads>1)
		
  		#cellsThreads = zeros(Int64,nThreads,2);
		#cellsThreads = SharedArray{Int64}(nThreads,2);

    	#cout << "nThreads: " <<  nThreads << endl;
    	#cout << "nCells: " <<  get_num_cells() << endl;
    	nParts = floor(nCells/nThreads);
    	#cout << "nParts: " <<  nParts << endl;


	    for i=1:nThreads
    		cellsThreads[i,2] =  nCells - nParts*(nThreads-i );
		end
	

    	for i=1:nThreads
      		cellsThreads[i,1] =  cellsThreads[i,2] - nParts + 1;
		end

    	cellsThreads[1,1] = 1;

		#display(cellsThreads);	
		
	end
	
	return cellsThreads;

end


@everywhere function distibuteNodesInThreadsSA(nThreads::Int64, nNodes::Int64 )::Array{Int32,2}

	nodesThreads = zeros(Int32,nThreads,2);
 
 	if (nThreads>1)
		
    	nParts = floor(nNodes/nThreads);

	    for i=1:nThreads
    		nodesThreads[i,2] =  nNodes - nParts*(nThreads-i );
		end
	

    	for i=1:nThreads
      		nodesThreads[i,1] =  nodesThreads[i,2] - nParts + 1;
		end

    	nodesThreads[1,1] = 1;
		
	end
	
	return nodesThreads;

end
