### A Pluto.jl notebook ###
# v0.14.7

using Markdown
using InteractiveUtils

# ╔═╡ 894130b0-3038-44fe-9d1f-46afe8734b98
begin
		using Plots, OffsetArrays, SparseArrays
		using StaticArrays, Dates, CUDA, JLD2
		using Statistics, LinearAlgebra  # standard libraries
end;#begin

# ╔═╡ bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
md""" ## LINEAR FSDTOPO"""

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### Version  v 0.1.15 
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
- LIN v0.1.15 All OK, clean up of matrices (back to plain numbers) and general code
"""

# ╔═╡ b23125f6-7118-4ce9-a10f-9c3d3061f8ce
md"""
### Setup model
"""

# ╔═╡ 179caf2b-67e9-417a-8c30-6d370af12182
plot([Int(ceil( (i/1000 )^.25 * (500 / 20) ))  for i in 0:1:1000   ])

# ╔═╡ 02a73612-ad33-4d5b-a4c6-c3e8b7f5c477
A = [1 0 ; 0 1 ]

# ╔═╡ 5e2dc37c-c07c-45da-a984-f1f8212fff21
sum(abs.(eigvals(A)))

# ╔═╡ 7ae886d4-990a-4b14-89d5-5708f805ef93
md"""
#### Call FSDTOPO with Niter
"""

# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
begin


# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
 KE_CQUAD4 = @SMatrix [
 1.0 0.3931452 -0.5766129 -0.03024194 -0.6330645 -0.3931452 0.2096774 0.03024194;  0.3931452 1.0 0.03024194 0.2096774 -0.3931452 -0.6330645 -0.03024194 -0.5766129; -0.5766129 0.03024194 1.0 -0.3931452 0.2096774 -0.03024194 -0.6330645 0.3931452; -0.03024194 0.2096774 -0.3931452 1.0 0.03024194 -0.5766129 0.3931452 -0.6330645; -0.6330645 -0.3931452 0.2096774 0.03024194 1.0 0.3931452 -0.5766129 -0.03024194; -0.3931452 -0.6330645 -0.03024194 -0.5766129 0.3931452 1.0 0.03024194 0.2096774;  0.2096774 -0.03024194 -0.6330645 0.3931452 -0.5766129 0.03024194 1.0 -0.3931452;  0.03024194 -0.5766129 0.3931452 -0.6330645 -0.03024194 0.2096774 -0.3931452 1.0]

# Matrix relating cartesian stress components (sxx, syy, sxy) with nodal displacement
SU_CQUAD4 = @SMatrix [
-1.209677 -0.3629032 1.209677 -0.3629032 1.209677 0.3629032 -1.209677 0.3629032; -0.3629032 -1.209677 0.3629032 -1.209677 0.3629032 1.209677 -0.3629032 1.209677; -0.4233871 -0.4233871 -0.4233871 0.4233871 0.4233871 0.4233871 0.4233871 -0.4233871]

	
end;#begin

# ╔═╡ c72f9b42-94c7-4377-85cd-5afebbe1d271
md"""
### NOTEBOOK SETUP
"""

# ╔═╡ 5c4c2c37-9873-4471-abd9-3b9c72ba8492
dpi_quality = 120

# ╔═╡ 87be1f09-c729-4b1a-b05c-48c79039390d
begin

println(">>> START FSD-TOPO: "  * string(Dates.now()))
	
# Set global parameters
const sigma_all	= 6.0
const max_all_t = 5.0
const max_penalty = 1.0
		
scale = 800
	
nelx = 6*scale ; nely = 2*scale  #mesh size

Niter = 1000

full_penalty_iter = Niter*0.1

conv_scale = 10	

			
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


	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)	
	
	
# Initialize iterated thickness in domain
t_iter = ones(Float64,1:nely,1:nelx).*max_all_t 	
	
	
	
	
# Loop niter times the FSD-TOPO algorithm		
for iter in 1:Niter
		
		
ngauss = Int(ceil( (iter/Niter ) ^1.0 * (scale / conv_scale) ))

Gauss_kernel = @MArray ones(2*ngauss+1,2*ngauss+1)			
Gauss_kernel = (collect([exp(- (i^2+j^2) / (2*ngauss^2)) for i in -ngauss:ngauss, j in -ngauss:ngauss])	)	
		
println("       Set Canvas: " * string(Dates.now()))		
canvas = OffsetArray(zeros(Float64, 1:nely+2*ngauss, 1:nelx+2*ngauss), (-ngauss+1):nely+ngauss,(-ngauss+1):nelx+ngauss)	
		
		

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
#s1 = 0.5 * (sxx + syy + √((sxx - syy) ^ 2 + 4 * sxy ^ 2))
#s2 = 0.5 * (sxx + syy - √((sxx - syy) ^ 2 + 4 * sxy ^ 2))				
#ssign = abs(s1) < abs(s2) ? 1 : -1							
#S[y, x] = ssign * √(s1 ^ 2 + s2 ^ 2 - s1 * s2)  # Von Mises stress in plane stress case
				
#S[y, x] = ssign * (abs(s1) + abs(s2))
				
				
S[y, x] =  sum(abs.(eigvals([sxx sxy ; sxy syy  ]))) # Michell criterion, sum of absolute values of principal stresses, eigenvalues of 2D stress tensor
				
				
end # for x
end # for y

		
		
# Obtain new thickness by FSD algorithm	and normalize		
#t_iter .*= ((((abs.(S) ./ sigma_all ) .- t_iter) .* 1) .+ t_iter)	./ max_all_t

#t_iter .*= (abs.(S) ./ (sigma_all * max_all_t) ) # if using signs
		
t_iter .*= S ./ (sigma_all * max_all_t) # Stress value is always positive
		
		
		
#*************************************************************************		
# apply spatial filter a decreasing number of times function of the iteration number, but proportional to the scale, in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)			
#if iter <  11111110 #Niter # / 2
println("       GAUSS Start: " * string(Dates.now()))								
# matr is convolved with kern. The key assumption is that the padding elements are 0 and no element in the "interior" of matr is = 0 (THIS IS A STRONG ASSUMPTION IN THE GENERAL CASE BUT VALID IN FSD-TOPO AS THERE IS A MINIMUM ELEMENT THICKNESS > 0)	
canvas[1:size(t_iter,1), 1:size(t_iter,2)] .= t_iter
# Return the sum the product of a subarray centered in the cartesian indices corresponding to i, of the interior matrix, and the kernel elements, centered in CartInd i. Then .divide by the sum of the weights multiplied by a 1 or a 0 depending on whether the base element is >0 or not. Note: the lines below are a single expression
			
t_iter .= [sum( canvas[i .+ CartesianIndices((-ngauss:ngauss, -ngauss:ngauss))] .* Gauss_kernel) / 
 sum((canvas[i .+ CartesianIndices((-ngauss:ngauss, -ngauss:ngauss))] .!== 0.0)  .* Gauss_kernel) for i in CartesianIndices(t_iter)]			
			
println("       GAUSS End: " * string(Dates.now()))	
#end # if iter do Gauss
#*************************************************************************	
		
		
# Limit thickness to maximum			
t_iter .= [min(nt, 1.0) for nt in t_iter] 		
		
# Calculate penalty at this iteration			
#penalty = min(1 + iter / full_penalty_iter, max_penalty) 
#t_iter .= max_all_t .* [nt^penalty + 1e-8 for nt in t_iter]				
		
#t_iter .= +1e-8 .+ max_all_t .* [((1.0 - cos(pi*i))/2)^penalty for i in t_iter]	

		
t_iter .= [(i > ((iter / Niter) * .95)) * i * max_all_t + 1.e-9 for i in t_iter]		
		
push!(t_res, copy(t_iter))			

curr_thick_plot = heatmap(reverse(t_iter, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " Iter: "*string(iter) *" Ngauss: "*string(ngauss) *" Scale: "*string(scale), dpi=dpi_quality, grids=false, tickfontsize=4, titlefontsize = 4)			

png(curr_thick_plot, "z:\\thick"*string(iter))		
		

end	# for topo iter
	
println("<<< END FSD-TOPO: "  * string(Dates.now()))		
		
end;#begin

# ╔═╡ a3592c76-5fe7-4936-9d02-5ad2ce25a504
Vol_Frac_pct = sum(t_res[end])/(nelx*nely*max_all_t)*100

# ╔═╡ 95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
heatmap(reverse(abs.(S).*t_res[end] ./5 , dims = 1), clim = (0, 15), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), dpi=dpi_quality, grids=false, showaxis=:y, tickfontsize=4)

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
function plot_animation()
	
	anim_evolution = @animate for i in 1:Niter	
		heatmap([ reverse(t_res[i], dims=(1,2)) reverse(t_res[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), fps=3, dpi=dpi_quality, grids=false, tickfontsize=4 , titlefontsize = 4)
	end
	
	
	gif(anim_evolution, "z:\\00_latest_thickness.gif", fps = 12)
	
end

# ╔═╡ b0de4ff7-5004-43f2-9c56-f8a27485754a
plot_animation()

# ╔═╡ c8ac6bd4-1315-4c85-980d-ad5b2a3141b1
function plot_animation_stress()
	
	anim_evolution = @animate for i in 1:Niter	
			
	heatmap(reverse(t_res[i] , dims = 1), clim = (0, 5), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), fps=10, dpi=dpi_quality, grids=false, tickfontsize=4 , titlefontsize = 4)
		
	end
	
	
	gif(anim_evolution, "z:\\00_latest_stress.gif", fps = 8)
	
end

# ╔═╡ f597b95c-4a65-4a8f-81cb-79d98b25e209
plot_animation_stress( )


# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
function show_final_design()
	heatmap(reverse(t_res[end].* (sign.(S)), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale) , dpi=dpi_quality, grids=false, tickfontsize=4, titlefontsize = 4)
end	

# ╔═╡ 4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
show_final_design()

# ╔═╡ Cell order:
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═179caf2b-67e9-417a-8c30-6d370af12182
# ╠═02a73612-ad33-4d5b-a4c6-c3e8b7f5c477
# ╠═5e2dc37c-c07c-45da-a984-f1f8212fff21
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╠═a3592c76-5fe7-4936-9d02-5ad2ce25a504
# ╠═4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
# ╠═95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
# ╠═b0de4ff7-5004-43f2-9c56-f8a27485754a
# ╠═f597b95c-4a65-4a8f-81cb-79d98b25e209
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╠═5c4c2c37-9873-4471-abd9-3b9c72ba8492
# ╠═894130b0-3038-44fe-9d1f-46afe8734b98
# ╠═7f47d8ef-98be-416d-852f-97fbaa287eec
# ╠═c8ac6bd4-1315-4c85-980d-ad5b2a3141b1
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
