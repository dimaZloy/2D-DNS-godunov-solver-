
using CUDA

include("RoeFlux2dCUDA.jl")
include("AUSMflux2dCUDA.jl")


function testFluxes2d()

    rhoLeft::Float64 = 1.0;
    ULeft::Float64 = 290.0;
    VLeft::Float64 = 0.0;
    PLeft::Float64 = 7143;

    rhoRight::Float64 = 1.7;
    URight::Float64 = 263.72;
    VRight::Float64 = -51.62;
    PRight::Float64 = 15282.0;

    nx::Float64  = 1.0;
    ny::Float64  = 0.0;
    side::Float64 = 1.0;

    gamma::Float64 = 1.4; 

    num_blocks::Int64 = 256;
    num_threads::Int64 = 10;

    num_elements::Int64  = num_blocks*num_threads;

    gammaV = zeros(Float64,num_elements );

    rhoLeftV = zeros(Float64,num_elements );
    ULeftV = zeros(Float64,num_elements );
    VLeftV = zeros(Float64,num_elements );
    PLeftV = zeros(Float64,num_elements );

    rhoRightV = zeros(Float64,num_elements );
    URightV = zeros(Float64,num_elements );
    VRightV = zeros(Float64,num_elements );
    PRightV = zeros(Float64,num_elements );

    nxV = zeros(Float64,num_elements );
    nyV = zeros(Float64,num_elements );
    sideV = zeros(Float64,num_elements );
   
    flux1V = zeros(Float64,num_elements );
    flux2V = zeros(Float64,num_elements );
    flux3V = zeros(Float64,num_elements );
    flux4V = zeros(Float64,num_elements );

    for i = 1:num_elements

        rhoLeftV[i] = rhoLeft;
        ULeftV[i] = ULeft;
        VLeftV[i] = VLeft;
        PLeftV[i] = PLeft;

        rhoRightV[i] = rhoRight;
        URightV[i] = URight;
        VRightV[i] = VRight;
        PRightV[i] = PRight;

        nxV[i] = nx;
        nyV[i] = ny;
        sideV[i] = side;
        gammaV[i] = gamma;

    end

    cuRhoLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuULeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuVLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuPLeftV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)

    cuRhoRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuURightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuVRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuPRightV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)

    cuNxV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuNyV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuSideV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)

    cuFlux1V = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuFlux2V = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuFlux3V = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    cuFlux4V = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)

    cuGammaV = CuVector{Float64, Mem.UnifiedBuffer}(undef,num_elements)
    copyto!(cuGammaV, gammaV)

    copyto!(cuRhoLeftV, rhoLeftV)
    copyto!(cuULeftV, ULeftV)
    copyto!(cuVLeftV, VLeftV)
    copyto!(cuPLeftV, PLeftV)

    copyto!(cuRhoRightV, rhoRightV)
    copyto!(cuURightV, URightV)
    copyto!(cuVRightV, VRightV)
    copyto!(cuPRightV, PRightV)

    copyto!(cuNxV, nxV)
    copyto!(cuNyV, nyV)
    copyto!(cuSideV, sideV)



    @cuda blocks = num_blocks threads = num_threads kernel_testRoe(cuRhoLeftV, cuULeftV, cuVLeftV, cuPLeftV, cuRhoRightV, cuURightV, cuVRightV, cuPRightV, 
         cuNxV, cuNyV, cuSideV, cuFlux1V, cuFlux2V, cuFlux3V,cuFlux4V, cuGammaV);

    display(cuFlux1V[1])
    display(cuFlux2V[1])
    display(cuFlux3V[1])
    display(cuFlux4V[1])
          

    @cuda blocks = num_blocks threads = num_threads kernel_test(cuRhoLeftV, cuULeftV, cuVLeftV, cuPLeftV, cuRhoRightV, cuURightV, cuVRightV, cuPRightV, 
         cuNxV, cuNyV, cuSideV, cuFlux1V, cuFlux2V, cuFlux3V,cuFlux4V, cuGammaV);

    
    display(cuFlux1V[1])
    display(cuFlux2V[1])
    display(cuFlux3V[1])
    display(cuFlux4V[1])

end


@time testFluxes2d();
@time testFluxes2d();
@time testFluxes2d();

