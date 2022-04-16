2D-DNS-parallel-godunov-solver
Direct Numerical simulations (DNS) of the supersonic flow over a circular cylinder at Mach number, M = 3.5.  2D Navier-Stokes equations are computed using the Godunov-type approach based on the 2nd order finite-volume method coupled with the AUSM+ flux splitting method. First order explicit Euler scheme is applied for time integration (it seems it works fine and comparable with alternative RK methods and BDF-2).  High performance parallelism is achieved suing a simple threads mechanism implemented in the Julia HPC language.  The latest tested mesh contains 1.04M nodes and 2.08M triangular cells. The Numerical Schlieren visualization of the flow is provided below:
<img src="https://github.com/dimaZloy/2D-DNS-parallel-godunov-solver/blob/main/final.png" alt="visualization"/>
[![Watch the video](https://i.ytimg.com/an_webp/bRAQS27o_GU/mqdefault_6s.webp?du=3000&sqp=CIWX6pIG&rs=AOn4CLDwxWGG7cIaGeByZQ_biGw2adBp8A)](https://youtu.be/bRAQS27o_GU)
