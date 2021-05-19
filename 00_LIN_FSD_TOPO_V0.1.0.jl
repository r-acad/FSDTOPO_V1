### A Pluto.jl notebook ###
# v0.14.5

using Markdown
using InteractiveUtils

# ╔═╡ 894130b0-3038-44fe-9d1f-46afe8734b98
begin
		using Plots, OffsetArrays, SparseArrays
		using StaticArrays
		using Statistics, LinearAlgebra  # standard libraries
end


Aa = rand(100, 100)
heatmap(Aa)


# ╔═╡ bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
md""" ## LINEAR FSDTOPO"""

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### Version  v 0.0.10 OK
- 0.0.8 Back to original formulation in 88 lines after attempt to reorder elements in v 0.0.6
- 0.0.8 OK works in obtaining a meaningful internal loads field

- 0.0.9 Clean up of 0.0.8OK   5 APR 21. It works with canonical problem

- 0.0.10 Added animation and additional clean-up. GAUSS FILTER ADDED, IT WORKS FINE
- V0.0.28 Element Matrices made static, general code clean up in linear section
- V0.0.30 Last Pluto version of the Linear Solver and FSD Topo

- LIN V0.1.0 First isolated Linear Solver version in Pluto
"""

# ╔═╡ 965946ba-8217-4202-8870-73d89c0c7340
md"""
### Global Parameters
"""

# ╔═╡ 6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
begin 
	
# Set global parameters
const sigma_all	= 3
const max_all_t = 5

const max_penalty = 5
const thick_ini = 1.0		
min_thick = max_all_t * 0.000001
		
scale = 10
nelx = 6*scale ; nely = 2*scale  #mesh size

Niter = 25

full_penalty_iter = Niter*.1
end;#begin

# ╔═╡ b23125f6-7118-4ce9-a10f-9c3d3061f8ce
md"""
### Setup model
"""

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin # Setup models

nDoF = 2*(nely+1)*(nelx+1)  # Total number of degrees of freedom
	
F = zeros(Float64, nDoF)	# Initialize external forces vector
F[2] = -1.0	   # Set applied external force
	
U = zeros(Float64, nDoF)	# Initialize global displacements
	
fixeddofs = [(1:2:2*(nely+1)); nDoF ]  # Set boundary conditions
	
freedofs  = setdiff([1:nDoF...],fixeddofs)	# Free DoFs for solving the disp.

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx) # Array with node numbers
	
edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:]	

edofMat = repeat(edofVec,1,8) + repeat([-1 -2 1 0 2*nely.+[3 2 1 0]],nelx*nely)	# order changed to match a K of a 2x1 mesh ftom 99 lines
	
iK = kron(edofMat,ones(Int64,8,1))'[:]
jK = kron(edofMat,ones(Int64,1,8))'[:]
	
end;#begin

# ╔═╡ 7ae886d4-990a-4b14-89d5-5708f805ef93
md"""
#### Call FSDTOPO with Niter
"""

# ╔═╡ 200a3956-d229-434b-bf25-105e34acb35b
#gif(anim_evolution, "scale_120_44Ksec_21_05_18.gif", fps = 6)

# ╔═╡ 6bd11d90-93c1-11eb-1368-c9484c1302ee
md""" ### FE SOLVER FUNCTIONS  """

# ╔═╡ d108d820-920d-11eb-2eee-bb6470fb4a56
md"""
### AUXILIARY FUNCTIONS and MATRICES
"""

# ╔═╡ b9ec0cbf-f9a2-4980-b7cd-1ecda0566631
@inline function convolution3x3(matr, kern)

# matr is convolved with kern. The key assumption is that the padding elements are 0 and no element in the "interior" of matr is = 0 (THIS IS A STRONG ASSUMPTION IN THE GENERAL CASE BUT VALID IN FSD-TOPO AS THERE IS A MINIMUM ELEMENT THICKNESS > 0)	
	
canvas = OffsetArray(zeros(Float64, 1:size(matr,1)+2, 1:size(matr,2)+2), 0:size(matr,1)+1,0:size(matr,2)+1)
	
canvas[1:size(matr,1), 1:size(matr,2)] = matr

# Return the sum the product of a subarray centered in the cartesian indices corresponding to i, of the interior matrix, and the kernel elements, centered in CartInd i. Then .divide by the sum of the weights multiplied by a 1 or a 0 depending on whether the base element is >0 or not. Note: the lines below are a single expression

[sum( canvas[i .+ CartesianIndices((-1:1, -1:1))]         .* kern)	/ 
 sum((canvas[i .+ CartesianIndices((-1:1, -1:1))] .> 0.0) .* kern) 
		 for i in CartesianIndices(matr)]
end

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

Gauss_3x3_kernel = @SMatrix [1.0 2.0 1.0 ;
				    		 2.0 4.0 2.0 ;
				    		 1.0 2.0 1.0 ] 	
	
end;#begin

# ╔═╡ 2c768930-9210-11eb-26f8-0dc24f22afaf
function SOLVE_INTERNAL_LOADS(thick)

	
sK = reshape(KE_CQUAD4[:]*thick[:]', 64*nelx*nely) 

# Build global stiffness matrix		
K = Symmetric(sparse(iK,jK,sK))
# Obtain global displacements	
U[freedofs] = K[freedofs,freedofs]\F[freedofs]		
	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)
				
@inbounds Threads.@threads 	for y = 1:nely
@inbounds Threads.@threads  for x = 1:nelx # Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
			
n2 = (nely+1)* x +y	; 	n1 = n2	- (nely+1)
			
Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
		
sxx, syy, sxy = (SU_CQUAD4 * Ue) .* nelx  # Element stress vector in x, y coordinates. Scaled by mesh size
	
# Principal stresses
s1 = 0.5 * (sxx + syy + √((sxx - syy) ^ 2 + 4 * sxy ^ 2))
s2 = 0.5 * (sxx + syy - √((sxx - syy) ^ 2 + 4 * sxy ^ 2))
S[y, x] = √(s1 ^ 2 + s2 ^ 2 - 2 * 0.3 * s1 * s2)  # elastic strain energy??

end # for x
end # for y
	
return S	
end # function	

# ╔═╡ 87be1f09-c729-4b1a-b05c-48c79039390d
function FSDTOPO(niter)	
	
# Initialize iterated thickness in domain
t_iter = ones(Float64,1:nely,1:nelx).*thick_ini  	
t_res = []	# Array of arrays with iteration history of thickness

# Loop niter times the FSD-TOPO algorithm		
for iter in 1:niter

# Obtain new thickness by FSD algorithm			
t_iter .*= SOLVE_INTERNAL_LOADS(t_iter) / sigma_all 

# Limit thickness to maximum			
t_iter .= [min(nt, max_all_t) for nt in t_iter] 
		
# Calculate penalty at this iteration			
penalty = min(1 + iter / full_penalty_iter, max_penalty) 
			
# apply spatial filter a decreasing number of times function of the iteration number, but proportional to the scale, in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)	
for gauss in 1:max(0,(ceil(scale*(iter-.7*Niter)/(2*-.7*Niter)))) 
	t_iter .= convolution3x3(t_iter, Gauss_3x3_kernel)					
end # for gauss								

t_iter .= [max((max_all_t*(min(nt,max_all_t)/max_all_t)^penalty), min_thick) for nt in t_iter]
			
push!(t_res, copy(t_iter))			
	
end	# for topo iter
		
return t_res # returns an array of the views of the canvas containing only the thickness domain for each iteration
		
end # end function

# ╔═╡ d007f530-9255-11eb-2329-9502dc270b0d
newt = FSDTOPO(Niter); # Call topology optimization problem and store results in array of arrays

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
function show_final_design()
heatmap(reverse(newt[end], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))
end	

# ╔═╡ 4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
show_final_design()

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
function plot_animation()
	
	anim_evolution = @animate for i in 1:Niter	
		heatmap([ reverse(newt[i], dims=(1,2)) reverse(newt[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), fps=3)
	end
	
	gif(anim_evolution, "scale_120_44Ksec_21_05_18.gif", fps = 6)
	
end

# ╔═╡ b0de4ff7-5004-43f2-9c56-f8a27485754a
plot_animation()

# ╔═╡ 0342ef5f-484d-4ed7-8260-b2b1330fead0
#heatmap( reverse(Array(Gauss_3x3_kernel), dims=1) , aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ c652e5c0-9207-11eb-3310-ddef16cdb1ac
#heatmap(reverse(Array(KE_CQUAD4), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ c58a7360-920c-11eb-2a15-bda7ed075812
#heatmap( reverse(Array(SU_CQUAD4), dims=1) , aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ c72f9b42-94c7-4377-85cd-5afebbe1d271
md"""
### NOTEBOOK SETUP
"""

# ╔═╡ fc7e00a0-9205-11eb-039c-23469b96de19
begin
	"""
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add([
			Pkg.PackageSpec(name="Plots", version="1.10"), 
			Pkg.PackageSpec(name="OffsetArrays"),
			Pkg.PackageSpec(name="SparseArrays"),
			Pkg.PackageSpec(name="StaticArrays")
			])
	using PlutoUI
	using Plots, OffsetArrays, SparseArrays
	using StaticArrays
	using Statistics, LinearAlgebra  # standard libraries
	
	TableOfContents(aside=true)
	"""
end;#begin

# ╔═╡ Cell order:
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─965946ba-8217-4202-8870-73d89c0c7340
# ╠═6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╠═4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
# ╠═b0de4ff7-5004-43f2-9c56-f8a27485754a
# ╠═7f47d8ef-98be-416d-852f-97fbaa287eec
# ╠═200a3956-d229-434b-bf25-105e34acb35b
# ╟─6bd11d90-93c1-11eb-1368-c9484c1302ee
# ╠═2c768930-9210-11eb-26f8-0dc24f22afaf
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╠═b9ec0cbf-f9a2-4980-b7cd-1ecda0566631
# ╠═cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╠═0342ef5f-484d-4ed7-8260-b2b1330fead0
# ╠═c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╠═c58a7360-920c-11eb-2a15-bda7ed075812
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╠═fc7e00a0-9205-11eb-039c-23469b96de19
# ╠═894130b0-3038-44fe-9d1f-46afe8734b98
