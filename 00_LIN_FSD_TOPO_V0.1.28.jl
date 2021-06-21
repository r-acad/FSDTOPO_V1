### A Pluto.jl notebook ###
# v0.14.8

using Markdown
using InteractiveUtils

# ╔═╡ 894130b0-3038-44fe-9d1f-46afe8734b98
begin
		using Plots, OffsetArrays, SparseArrays
		using StaticArrays, Dates, CUDA, HDF5
		using Statistics, LinearAlgebra  # standard libraries
end;#begin

# ╔═╡ bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
md""" ## LINEAR FSDTOPO"""

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
- 0.0.8 Back to original formulation in 88 lines after attempt to reorder elements in v 0.0.6
- 0.0.8 OK works in obtaining a meaningful internal loads field

- 0.0.9 Clean up of 0.0.8OK   5 APR 21. It works with canonical problem

- 0.0.10 Added animation and additional clean-up. GAUSS FILTER ADDED, IT WORKS FINE
- V0.0.28 Element Matrices made static, general code clean up in linear section
- V0.0.30 Last Pluto version of the Linear Solver and FSD Topo

- LIN V0.1.0 First isolated Linear Solver version in Pluto
- LIN v0.1.3 Code refactored to remove function calls, all OK
- LIN v0.1.8 All OK
- LIN v0.1.10  Attempt to use convolution in threads with explicit for-loops
- LIN v0.1.15 All OK, clean up of matrices (back to plain numbers) and general code
- LIN v0.1.21 Calculation of element stresses now done directly using eigenvalues of element stress tensor. Last version containing code remains of classical penalty formulation, from this version onwards only l0 threshold is kept
"""

# ╔═╡ 635d9298-a222-4088-bba4-df2d7e2b6776
plot([
		min(40,Int(max(ceil((((iter-1000*.4)/ ((1-.4)*1000)))* 300*6 * 1.5/100),1))) 		
		for iter in 1:1000] )

# ╔═╡ c64b6cc8-abd2-4cfe-8504-8ce676a80a59


# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
begin


# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
 KE_CQUAD4 = @SMatrix [
 1.0 0.3931452 -0.5766129 -0.03024194 -0.6330645 -0.3931452 0.2096774 0.03024194;  0.3931452 1.0 0.03024194 0.2096774 -0.3931452 -0.6330645 -0.03024194 -0.5766129; -0.5766129 0.03024194 1.0 -0.3931452 0.2096774 -0.03024194 -0.6330645 0.3931452; -0.03024194 0.2096774 -0.3931452 1.0 0.03024194 -0.5766129 0.3931452 -0.6330645; -0.6330645 -0.3931452 0.2096774 0.03024194 1.0 0.3931452 -0.5766129 -0.03024194; -0.3931452 -0.6330645 -0.03024194 -0.5766129 0.3931452 1.0 0.03024194 0.2096774;  0.2096774 -0.03024194 -0.6330645 0.3931452 -0.5766129 0.03024194 1.0 -0.3931452;  0.03024194 -0.5766129 0.3931452 -0.6330645 -0.03024194 0.2096774 -0.3931452 1.0]


	
# Matrix relating cartesian stress components [sxx, sxy; syy, sxy] with nodal displacements
SU_CQUAD4 = @SMatrix [
-1.209677 -0.3629032 1.209677 -0.3629032 1.209677 0.3629032 -1.209677 0.3629032; 
-0.4233871 -0.4233871 -0.4233871 0.4233871 0.4233871 0.4233871 0.4233871 -0.4233871;
-0.4233871 -0.4233871 -0.4233871 0.4233871 0.4233871 0.4233871 0.4233871 -0.4233871;
-0.3629032 -1.209677 0.3629032 -1.209677 0.3629032 1.209677 -0.3629032 1.209677	]
	
end;#begin

# ╔═╡ c72f9b42-94c7-4377-85cd-5afebbe1d271
md"""
#### NOTEBOOK SETUP
"""

# ╔═╡ 5c4c2c37-9873-4471-abd9-3b9c72ba8492
dpi_quality = 120

# ╔═╡ 87be1f09-c729-4b1a-b05c-48c79039390d
begin

println(">>> START FSD-TOPO: "  * string(Dates.now()))
	
# Set global parameters
scale = 40
Niter = 20
	
const sigma_all	= 6.0 # allowable sigma 
const max_all_t = 5.0 # max. thickness

conv_scale = 2.5  # % of nelx

nelx = 6*scale ; nely = 2*scale  #mesh size		
	

	
# Initialize external forces vector	
F = zeros(Float64,  2*(nely+1)*(nelx+1)) 
F[2] = -1.0	   # Set applied external force	

# Set boundary conditions	
fixeddofs = [(1:2:2*(nely+1));  2*(nely+1)*(nelx+1) ]  
	
U = zeros(Float64,  2*(nely+1)*(nelx+1)) # Initialize global displacements

# Free DoFs for solving the displacements	
freedofs  = setdiff([1:( 2*(nely+1)*(nelx+1))...],fixeddofs) 
	
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx) # Array with node numbers	
edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:] # Vector of element DoFs

# order changed to match a K of a 2x1 mesh from 99 lines	
edofMat = repeat(edofVec,1,8) + repeat([-1 -2 1 0 2*nely.+[3 2 1 0]],nelx*nely)	
	
iK = kron(edofMat,ones(Int64,8,1))'[:]
jK = kron(edofMat,ones(Int64,1,8))'[:]
	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing element stresses	
	
# Initialize iterated thickness in domain and set to maximum thickness
t_iter = ones(Float64,1:nely,1:nelx).*max_all_t 	

t_res = []	# Array of arrays with iteration history of thickness	
S_res = []	# Array of arrays with iteration history of stress	
vol_frac_array = [] # Array of volume fraction evolution	
compliance_array = [] # Array of compliance evolution	
ngauss_array = [] # Array of ngauss evolution	
	
	
# ********* Loop niter times the FSD-TOPO algorithm *********
for iter in 1:Niter

# 40 is a safe limit for a static array in a Ryzen9, reduce or remove mutable static array when publishing		
ngauss = min(40,Int(max(ceil((((iter-Niter*.4)/ (Niter*(1-.4))))* nelx * conv_scale/100),1)))   # -1

	
		
# Initialize "canvas" (domain surrounded by a frame of 0.0 values)		
canvas = OffsetArray(
			zeros(Float64, 1:nely+2*ngauss, 1:nelx+2*ngauss), 
			(-ngauss+1):nely+ngauss,(-ngauss+1):nelx+ngauss )	
		
domain = copy(canvas)		
domain[1:size(t_iter,1), 1:size(t_iter,2)] .= 1.0 # array with 1.0 in the domain
		
	

println("TOPO ITER : " * string(iter) * " " * string(Dates.now()))	
		
sK = reshape(KE_CQUAD4[:]*t_iter[:]', 64*nelx*nely) 



# Build global stiffness matrix		
K = sparse(iK,jK,sK)
		
# Solve for global displacements	
U[freedofs] = K[freedofs,freedofs]\F[freedofs]
	
println("    Calculate internal loads "  * string(Dates.now()))	

		
# Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
@inbounds Threads.@threads for x = 1:nelx; @inbounds Threads.@threads for y = 1:nely n2 = (nely+1)* x +y	; n1 = n2 - (nely+1)
Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
# Vector of element principal stresses, eigenvalues of the stress tensor. Scaled by mesh size		
elm_princ_stress_vec = eigvals(reshape((SU_CQUAD4 * Ue) .* nelx , 2, 2)) 	
S[y, x] = sign(sum(elm_princ_stress_vec)) * sum(abs.(elm_princ_stress_vec))			end; end # for y, x
		
# FSD algorithm, using absolute value of S which has the tension-compression state encoded in the sign of the elms of S.		
t_iter .*= (abs.(S) ./ (sigma_all * max_all_t) ) 


		
if ngauss > 0		
#*************************************************************************		
# apply spatial filter 
println("       GAUSS Start: " * string(Dates.now()) * " Ngauss = " * string(ngauss))
		
Gauss_kernel = SMatrix{2ngauss+1, 2ngauss+1}(collect([exp(- (i^2+j^2) / (1*ngauss^2)) for i in -ngauss:ngauss, j in -ngauss:ngauss]) )			

canvas[1:size(t_iter,1), 1:size(t_iter,2)] .= t_iter
# Return the sum the product of a subarray centered in the cartesian indices corresponding to i, of the interior matrix, and the kernel elements, centered in CartInd i. Then .divide by the sum of the weights multiplied by a 1 or a 0 depending on whether the base element is >0 or not. Note: the lines below are a single logical expression
		
# check this https://discourse.julialang.org/t/unpacking-cartesianindex/27374/4		
		
t_iter .= [
sum(canvas[i .+ CartesianIndices((-ngauss:ngauss,-ngauss:ngauss))] .* Gauss_kernel) / 
sum(domain[i .+ CartesianIndices((-ngauss:ngauss,-ngauss:ngauss))] .* Gauss_kernel) for i in CartesianIndices(t_iter)]		
		
println("       GAUSS End: " * string(Dates.now()))	
#*************************************************************************	
end		


			
		
# Limit thickness to maximum and force progressive l0 threshold		
"""
t_iter .= [((t > (min((iter / Niter), .95))) * min(t, 1.0) + 1.e-5)* max_all_t
			for t in t_iter]	
"""

t_iter .= [((t > (min((iter / Niter), .95))) * min(t, 1.0) + 1.e-5)* max_all_t
			for t in t_iter]			
		
		

		
# Store intermediate results		
Vol_Frac_pct = sum(t_iter)/(nelx*nely*max_all_t)*100
push!(vol_frac_array, 	Vol_Frac_pct)	
		
push!(compliance_array, abs(U[2]))
push!(ngauss_array, ngauss)		
		
push!(t_res, copy(t_iter))	
push!(S_res, copy(S))			

# Save current thickness plot
curr_thick_plot = heatmap(reverse(t_iter, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " Iter: "*string(iter) *" ngauss: "*string(ngauss) *" Scale: "*string(scale), dpi=dpi_quality, grids=false, tickfontsize=4, titlefontsize = 4)			

png(curr_thick_plot, "z:\\thick"*string(iter))		
		

end	# for topo iter

println("<<< END FSD-TOPO: "  * string(Dates.now()))		
		
end;#begin

# ╔═╡ a448b803-3925-4d5b-856b-b62dfaa3c3a9
function plot_volfrac()

metrics = plot([normalize(vol_frac_array), normalize(abs.(compliance_array))], 
		label=["Volume fraction"  "Normalized compliance"  ], 
		legend=:topleft, 
		background_colour = :black,
		xlabel="Iteration")
	

png(metrics, "z:\\01_metrics")			
	
end	

# ╔═╡ e7ffd9ab-6d00-4032-b76b-7d46beea3bce
plot_volfrac()	

# ╔═╡ 997b6acc-67b7-4c68-8a7a-ee07adabede6
begin
	
		conv_plot = plot(ngauss_array./nelx.*100 , label= "effect. conv. scale", legend=:topleft, xlabel="Iteration", ylabel = "ngauss, % effect. conv. scale",background_colour = :black)
		
		
		conv_plot = plot!(ngauss_array, label="ngauss", legend=:topleft, xlabel="Iteration", ylabel = "ngauss, % effect. conv. scale")
			
png(conv_plot, "z:\\02_ngauss_and_conv_scale")	
		
end

# ╔═╡ f955a0d5-c47e-44a7-832c-e9c75a1d0b41
function plot_animation_s_histogram()
	
	anim_evolution = @animate for i in 1:Niter	
			
	histogram(abs.(S_res[i][:]) , 
			bins = 0:1:80, 
			title= string(Dates.now())* " niter: "*string(i) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), 
			fps=10, 
			dpi=dpi_quality, 
			grids=false, 
			tickfontsize=4 , 
			titlefontsize = 4, 
			yaxis=nothing,
			background_colour = :black,
		    label= "Σ|λ|")
		
	end
	
	
	gif(anim_evolution, "z:\\00_s_histogram.gif", fps = 8)
	
end

# ╔═╡ 55ae0872-7811-4c20-bb8b-8c6fa773ebaa
aa = plot_animation_s_histogram()

# ╔═╡ c84563d9-ca89-4035-b858-fb0dc96c14b6
function stress_histogram(i)
		histogram(abs.(S_res[i][:]) , 
			bins = 0:80, 
			title= string(Dates.now())* " niter: "*string(i) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), 
			fps=10, 
			dpi=dpi_quality, 
			grids=false, 
			tickfontsize=4 , 
			titlefontsize = 4, 
			yaxis=nothing,
		background_colour = :black,
		
			label = "S")
end	

# ╔═╡ a4aa852b-ee1d-4fb8-8aab-8b652b1bebfd
function th_histogram(i)
		histogram(abs.(t_res[i][:]) , 
			bins = 0.2:0.05:5, 
			title= string(Dates.now())* " niter: "*string(i) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), 
			fps=10, 
			dpi=dpi_quality, 
			grids=false, 
			tickfontsize=4 , 
			titlefontsize = 4, 
			yaxis=nothing,
		background_colour = :black,
		
			label = "th")
end	

# ╔═╡ 07197159-15f2-446e-8307-9150a7ecae7c
function plot_animation_th_histogram()
	
	anim_evolution = @animate for i in 1:Niter	
			
	th_histogram(i)
		
	end
	
	
	gif(anim_evolution, "z:\\00_th_histogram.gif", fps = 8)
	
end

# ╔═╡ 3de24285-6de1-4891-abd5-571d2232e28b
plot_animation_th_histogram()

# ╔═╡ 95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
heatmap(reverse(abs.(S).*t_res[end] , dims = 1), 
	clim = (0, 80), 
	aspect_ratio = 1, 
	c=cgrad([:black, :blue, :cyan, :green, :yellow, :orange, :red, :white], 13),   dpi=dpi_quality, 
	grids=false, 
	showaxis=:y, 
	tickfontsize=4, 
	background_colour = :black)

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
function plot_symm_thickness_animation()
	
	anim_evolution = @animate for i in 1:Niter	
		
		heatmap([ reverse(t_res[i], dims=(1,2)) reverse(t_res[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), fps=3, dpi=dpi_quality, grids=false, tickfontsize=4 , titlefontsize = 4)
	
	
	end
	
	
	gif(anim_evolution, "z:\\00_latest_symm_thickness.gif", fps = 12)
	
end

# ╔═╡ b0de4ff7-5004-43f2-9c56-f8a27485754a
plot_symm_thickness_animation()

# ╔═╡ 4048f21a-b68e-4afc-a473-f36d6aa7baa2
function plot_thickness_animation_black()
	
anim_evolution = @animate for i in 1:Niter	
		
heatmap([ reverse(t_res[i], dims=(1,2)) reverse(t_res[i], dims=1)], 
aspect_ratio = 1, 
#c=cgrad([:black, :blue, :cyan, :green, :yellow, :orange, :red, :white], 13), 	
c=cgrad([:black, :white], 13), 				
title= string(Dates.now())* " Iter: "*string(i) *" /" *string(Niter) *" conv_scale: "*string(conv_scale) *" nelx: "*string(nelx) *" nely: "*string(nely)           , 
			
fps=6, 
dpi=dpi_quality, 
grids=false, 
tickfontsize=10, 
titlefontsize = 14, 
background_colour = :black, 
size =(1920, 1080), 
legend=false, 
showaxis=:y, 
yaxis=nothing)

end
	
	
	gif(anim_evolution, "z:\\00_symm_thickness_anim_black.gif", fps = 6)
	
end

# ╔═╡ cb03641d-22fb-4da7-97cb-f676a5e6b386
plot_thickness_animation_black();

# ╔═╡ 6cc8611b-a04d-43d4-8451-d6512f9719d6
function plot_signed_stress_animation_black()
	
anim_evolution = @animate for i in 1:Niter	
		
heatmap(reverse(S_res[i].*t_res[i], dims=1), 
aspect_ratio = 1, 
c=cgrad([:blue, :black, :red], 13), 	
	
clim = (-20, 20),			
title= string(Dates.now())* " Iter: "*string(i) *" /" *string(Niter) *" conv_scale: "*string(conv_scale) *" nelx: "*string(nelx) *" nely: "*string(nely)           , 
legend=false,			
fps=6, 
dpi=dpi_quality, 
grids=false, 
tickfontsize=10, 
titlefontsize = 14, 
background_colour = :black, 
size =(1920, 1080),  
showaxis=:y, 
yaxis=nothing)

end
	
	
	gif(anim_evolution, "z:\\00_signed_stress_anim_black.gif", fps = 6)
	
end

# ╔═╡ 27e8587d-4e82-48ce-aa7a-38a30c2780a3
plot_signed_stress_animation_black();

# ╔═╡ f8d64921-0594-4b69-b7d5-c5d42641c504
function plot_abs_stress_animation_black()
	
anim_evolution = @animate for i in 1:Niter	
		
heatmap(reverse(abs.(S_res[i].*t_res[i]), dims=1), 
aspect_ratio = 1, 
c=cgrad([:black, :blue, :cyan, :green, :yellow, :orange, :red, :white], 13), 	
clim = (0, 50),			
title= string(Dates.now())* " Iter: "*string(i) *" /" *string(Niter) *" conv_scale: "*string(conv_scale) *" nelx: "*string(nelx) *" nely: "*string(nely)           , 
			
fps=6, 
dpi=dpi_quality, 
grids=false, 
tickfontsize=10, 
titlefontsize = 14, 
background_colour = :black, 
size =(1920, 1080),  
showaxis=:y, 
yaxis=nothing)

end
	gif(anim_evolution, "z:\\00_abs_stress_anim_black.gif", fps = 6)
end

# ╔═╡ 5d75ae57-4e3e-46c6-8b3a-42160ebf0d71
plot_abs_stress_animation_black();

# ╔═╡ c8ac6bd4-1315-4c85-980d-ad5b2a3141b1
function plot_animation_half_thickness()
	
	anim_evolution = @animate for i in 1:Niter	
			
	heatmap(reverse(t_res[i] , dims = 1), clim = (0, 5), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale), fps=10, dpi=dpi_quality, grids=false, tickfontsize=4 , titlefontsize = 4)
		
	end
	
	
	gif(anim_evolution, "z:\\00_latest_stress.gif", fps = 8)
	
end

# ╔═╡ f597b95c-4a65-4a8f-81cb-79d98b25e209
plot_animation_half_thickness()

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
function show_signed_stress(n)
	heatmap(reverse(t_res[n].* (sign.(S)), dims=1), 
	aspect_ratio = 1, 
	c=cgrad(:jet1, 10, categorical = true), 
	title= string(Dates.now())* " NIter: "*string(Niter) *" conv_scale: "*string(conv_scale) *" Scale: "*string(scale) , 
	dpi=dpi_quality, 
	grids=false, 
	tickfontsize=4, 
	titlefontsize = 4)
end	

# ╔═╡ 95bc4b85-bc89-4375-9466-e4491235c716
begin
	
	
	anim_evolution = @animate for i in 1:Niter	
	
	
	l1 = @layout [a{0.6h}; b{0.5w} c ]
	
	z1 = show_signed_stress(i)
	z2 = th_histogram(i)
	z3 = stress_histogram(i)
	
	plot( z1, z2, z3, layout = l1 ,
	size =(1920, 1080),  )
		
	end	
	
	gif(anim_evolution, "z:\\00_signed_s_and_histogram.gif", fps = 8)
		
end

# ╔═╡ 4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
bb = show_signed_stress(Niter)

# ╔═╡ Cell order:
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─635d9298-a222-4088-bba4-df2d7e2b6776
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╠═c64b6cc8-abd2-4cfe-8504-8ce676a80a59
# ╠═e7ffd9ab-6d00-4032-b76b-7d46beea3bce
# ╠═a448b803-3925-4d5b-856b-b62dfaa3c3a9
# ╠═997b6acc-67b7-4c68-8a7a-ee07adabede6
# ╠═95bc4b85-bc89-4375-9466-e4491235c716
# ╠═55ae0872-7811-4c20-bb8b-8c6fa773ebaa
# ╠═f955a0d5-c47e-44a7-832c-e9c75a1d0b41
# ╠═3de24285-6de1-4891-abd5-571d2232e28b
# ╠═c84563d9-ca89-4035-b858-fb0dc96c14b6
# ╠═a4aa852b-ee1d-4fb8-8aab-8b652b1bebfd
# ╠═07197159-15f2-446e-8307-9150a7ecae7c
# ╠═4c4e1eaa-d605-47b0-bce9-240f15c6f0aa
# ╠═95b78a43-1caa-4840-ba5c-a0dbd6c78d0d
# ╠═b0de4ff7-5004-43f2-9c56-f8a27485754a
# ╠═f597b95c-4a65-4a8f-81cb-79d98b25e209
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╟─5c4c2c37-9873-4471-abd9-3b9c72ba8492
# ╟─894130b0-3038-44fe-9d1f-46afe8734b98
# ╟─7f47d8ef-98be-416d-852f-97fbaa287eec
# ╠═cb03641d-22fb-4da7-97cb-f676a5e6b386
# ╟─4048f21a-b68e-4afc-a473-f36d6aa7baa2
# ╠═27e8587d-4e82-48ce-aa7a-38a30c2780a3
# ╟─6cc8611b-a04d-43d4-8451-d6512f9719d6
# ╠═5d75ae57-4e3e-46c6-8b3a-42160ebf0d71
# ╠═f8d64921-0594-4b69-b7d5-c5d42641c504
# ╟─c8ac6bd4-1315-4c85-980d-ad5b2a3141b1
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
