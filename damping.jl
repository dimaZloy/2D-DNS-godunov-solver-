
using PyPlot


r = 2.0:0.05:4.0;
a = 0.01;
b = 10.0;
rs = 2.0;
re = 4.0

rb = (r.-rs)./(re.-rs)
ksi = (1.0 .- a.*rb.*rb).* (1.0 .- (1.0 .- exp.(b.*rb.*rb))./(1.0 .- exp.(b)))
U = 1.0 .- ksi.*(rand() .- 1.0)

figure(1)
clf()
plot(r,ksi)
