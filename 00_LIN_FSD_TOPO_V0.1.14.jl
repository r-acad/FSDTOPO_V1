### A Pluto.jl notebook ###
# v0.14.6

using Markdown
using InteractiveUtils

# ╔═╡ 894130b0-3038-44fe-9d1f-46afe8734b98
begin
		using Plots, OffsetArrays, SparseArrays
		using StaticArrays, Dates, CUDA
		using Statistics, LinearAlgebra  # standard libraries
end;#begin

# ╔═╡ bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
md""" ## LINEAR FSDTOPO"""

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### Version  v 0.1.11 
- 0.0.8 Back to original formulation in 88 lines after attempt to reorder elements in v 0.0.6
- 0.0.8 OK works in obtaining a meaningful internal loads field

- 0.0.9 Clean up of 0.0.8OK   5 APR 21. It works with canonical problem

- 0.0.10 Added animation and additional clean-up. GAUSS FILTER ADDED, IT WORKS FINE
- V0.0.28 Element Matrices made static, general code clean up in linear section
- V0.0.30 Last Pluto version of the Linear Solver and FSD Topo

- LIN V0.1.0 First isolated Linear Solver version in Pluto
- LIN v0.1.3 Code refactored to remove function calls, all OK
- LIN v0.1.8 All OK
- LIN v0.1.10  Attempt to use convolution in threads with explitic for loops
"""

# ╔═╡ b23125f6-7118-4ce9-a10f-9c3d3061f8ce
md"""
### Setup model
"""

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin # Setup models

println(">>> START FSD-TOPO: "  * string(Dates.now()))
	
# Set global parameters
const sigma_all	= 7.0
const max_all_t = 5.0
const max_penalty = 5
		
scale = 40
	
nelx = 6*scale ; nely = 2*scale  #mesh size

Niter = 25

full_penalty_iter = Niter*1.1

ngauss = min(10, Int(floor(scale / 10)))

Gauss_kernel = @MArray ones(2*ngauss+1,2*ngauss+1)			
Gauss_kernel = (collect([exp(- (i^2+j^2) / (2*ngauss^2)) for i in -ngauss:ngauss, j in -ngauss:ngauss])	)	
			
println("       Set Forces: " * string(Dates.now()))		
F = zeros(Float64,  2*(nely+1)*(nelx+1))	# Initialize external forces vector
# Loads
F[2] = -1.0	   # Set applied external force	
	
fixeddofs = [(1:2:2*(nely+1));  2*(nely+1)*(nelx+1) ]  # Set boundary conditions
	
U = zeros(Float64,  2*(nely+1)*(nelx+1))	# Initialize global displacements
	
freedofs  = setdiff([1:( 2*(nely+1)*(nelx+1))...],fixeddofs)	# Free DoFs for solving the disp.

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx) # Array with node numbers
	
edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:]	

edofMat = repeat(edofVec,1,8) + repeat([-1 -2 1 0 2*nely.+[3 2 1 0]],nelx*nely)	# order changed to match a K of a 2x1 mesh ftom 99 lines
	
iK = kron(edofMat,ones(Int64,8,1))'[:]
jK = kron(edofMat,ones(Int64,1,8))'[:]
 	
t_res = []	# Array of arrays with iteration history of thickness	

println("       Set Canvas: " * string(Dates.now()))		
canvas = OffsetArray(zeros(Float64, 1:nely+2*ngauss, 1:nelx+2*ngauss), (-ngauss+1):nely+ngauss,(-ngauss+1):nelx+ngauss)	
	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)	

end;#begin

# ╔═╡ 7ae886d4-990a-4b14-89d5-5708f805ef93
md"""
#### Call FSDTOPO with Niter
"""

# ╔═╡ 95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
heatmap(reverse(abs.(S).*t_res[end] ./5 , dims = 1), clim = (0, 12), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
begin
	
# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
const AK4 = -5.766129E-01; const BK4 = -6.330645E-01 
const CK4 =  2.096774E-01 ; const DK4 = 3.931452E-01;  const GK4 = 3.024194E-02	
KE_CQUAD4 = @SMatrix [ 	 1   DK4  AK4 -GK4   BK4 -DK4  CK4  GK4;
		 				DK4    1  GK4  CK4  -DK4  BK4 -GK4  AK4;
		 				AK4  GK4  1   -DK4   CK4 -GK4  BK4  DK4;
			   	       -GK4  CK4 -DK4  1     GK4  AK4  DK4  BK4;
			 			BK4 -DK4  CK4  GK4   1    DK4  AK4 -GK4;
				   	   -DK4  BK4 -GK4  AK4   DK4  1    GK4  CK4;
			 			CK4 -GK4  BK4  DK4   AK4  GK4  1   -DK4;
			 			GK4  AK4  DK4  BK4  -GK4  CK4 -DK4   1  ]	
	
# Matrix relating cartesian stress components (sxx, syy, sxy) with nodal displacements in CQUAD4 element, reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
const AS4 = -1.209677E+00; const BS4 = -3.629032E-01; const CS4 = -4.233871E-01  	
SU_CQUAD4 = @SMatrix [ 	 AS4  BS4  -AS4 BS4 -AS4 -BS4  AS4 -BS4;
						 BS4  AS4  -BS4 AS4 -BS4 -AS4  BS4 -AS4;
						 CS4  CS4  CS4 -CS4 -CS4 -CS4 -CS4  CS4]
	
end;#begin

# ╔═╡ 87be1f09-c729-4b1a-b05c-48c79039390d
begin

	
# Initialize iterated thickness in domain
t_iter = ones(Float64,1:nely,1:nelx).*max_all_t 	
	
# Loop niter times the FSD-TOPO algorithm		
for iter in 1:Niter

println("TOPO ITER : " * string(iter) * " " * string(Dates.now()))	
		
sK = reshape(KE_CQUAD4[:]*t_iter[:]', 64*nelx*nely) 

println("       Solve linear system: " * string(Dates.now()))			
# Build global stiffness matrix		
K = sparse(iK,jK,sK)
		
# Obtain global displacements	
U[freedofs] = K[freedofs,freedofs]\F[freedofs]
	
println(" Calculate internal loads "  * string(Dates.now()))	
		
@inbounds Threads.@threads 	for x = 1:nelx
@inbounds Threads.@threads  for y = 1:nely # Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
			
n2 = (nely+1)* x +y	; 	n1 = n2	- (nely+1)
Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
		
sxx, syy, sxy = (SU_CQUAD4 * Ue) .* nelx  # Element stress vector in x, y coordinates. Scaled by mesh size	
# Principal stresses
s1 = 0.5 * (sxx + syy + √((sxx - syy) ^ 2 + 4 * sxy ^ 2))
s2 = 0.5 * (sxx + syy - √((sxx - syy) ^ 2 + 4 * sxy ^ 2))				
ssign = abs(s1) < abs(s2) ? 1 : -1							
S[y, x] = ssign * √(s1 ^ 2 + s2 ^ 2 - s1 * s2)  # Von Mises stress in plane stress case
				
#S[y, x] = ssign * (abs(s1) + abs(s2))

			
				
				
				
end # for x
end # for y

		
		
		
# Obtain new thickness by FSD algorithm	and normalize		
t_iter .*= ((((abs.(S) ./ sigma_all ) .- t_iter) .* 1) .+ t_iter)	./ max_all_t
	
		
#*************************************************************************		
# apply spatial filter a decreasing number of times function of the iteration number, but proportional to the scale, in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)			
if iter < 0 #Niter	-120
println("       GAUSS Start: " * string(Dates.now()))								
# matr is convolved with kern. The key assumption is that the padding elements are 0 and no element in the "interior" of matr is = 0 (THIS IS A STRONG ASSUMPTION IN THE GENERAL CASE BUT VALID IN FSD-TOPO AS THERE IS A MINIMUM ELEMENT THICKNESS > 0)	
canvas[1:size(t_iter,1), 1:size(t_iter,2)] .= t_iter
# Return the sum the product of a subarray centered in the cartesian indices corresponding to i, of the interior matrix, and the kernel elements, centered in CartInd i. Then .divide by the sum of the weights multiplied by a 1 or a 0 depending on whether the base element is >0 or not. Note: the lines below are a single expression
			
t_iter .= [sum( canvas[i .+ CartesianIndices((-ngauss:ngauss, -ngauss:ngauss))] .* Gauss_kernel) / 
 sum((canvas[i .+ CartesianIndices((-ngauss:ngauss, -ngauss:ngauss))] .!== 0.0)  .* Gauss_kernel) for i in CartesianIndices(t_iter)]			
			
println("       GAUSS End: " * string(Dates.now()))	
end # if iter do Gauss
#*************************************************************************	
		
		
# Limit thickness to maximum			
t_iter .= [min(nt, 1) for nt in t_iter] 		
		
# Calculate penalty at this iteration			
penalty = min(1 + iter / full_penalty_iter, max_penalty) 
t_iter .= max_all_t .* [max((nt^penalty),  1e-8) for nt in t_iter]				
			
		
push!(t_res, copy(t_iter))			

curr_thick_plot = heatmap(reverse(t_iter, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " Iter: "*string(iter) *" Ngauss: "*string(ngauss) *" Scale: "*string(scale))			
png(curr_thick_plot, "z:\\thick"*string(iter))		
		

end	# for topo iter
	
println("<<< END FSD-TOPO: "  * string(Dates.now()))		
		
end;#begin

# ╔═╡ c72f9b42-94c7-4377-85cd-5afebbe1d271
md"""
### NOTEBOOK SETUP
"""

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
function plot_animation()
	
	anim_evolution = @animate for i in 1:Niter	
		heatmap([ reverse(t_res[i], dims=(1,2)) reverse(t_res[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), fps=3)
	end
	
	gif(anim_evolution, "z:\\latest.gif", fps = 6)
	
end

# ╔═╡ b0de4ff7-5004-43f2-9c56-f8a27485754a
plot_animation()

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
function show_final_design()
	heatmap(reverse(t_res[end].* (sign.(S)), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" Ngauss: "*string(ngauss) *" Scale: "*string(scale) )
end	

# ╔═╡ 4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
show_final_design()

# ╔═╡ Cell order:
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╠═4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
# ╠═95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
# ╠═b0de4ff7-5004-43f2-9c56-f8a27485754a
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╟─894130b0-3038-44fe-9d1f-46afe8734b98
# ╟─7f47d8ef-98be-416d-852f-97fbaa287eec
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
