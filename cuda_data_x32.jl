
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