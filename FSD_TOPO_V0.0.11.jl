### A Pluto.jl notebook ###
# v0.14.2

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : missing
        el
    end
end

# ╔═╡ fc7e00a0-9205-11eb-039c-23469b96de19
begin
	import Pkg
	Pkg.activate(mktempdir())
	Pkg.add([
			Pkg.PackageSpec(name="Images", version="0.22.4"), 
			Pkg.PackageSpec(name="ImageMagick", version="0.7"), 
			Pkg.PackageSpec(name="PlutoUI", version="0.7"), 
			Pkg.PackageSpec(name="Plots", version="1.10"), 
			Pkg.PackageSpec(name="Colors", version="0.12"),
			Pkg.PackageSpec(name="ColorSchemes", version="3.10"),
			Pkg.PackageSpec(name="ForwardDiff"),
			Pkg.PackageSpec(name="LaTeXStrings"),
			Pkg.PackageSpec(name="OffsetArrays"),
			Pkg.PackageSpec(name="SparseArrays"),
			Pkg.PackageSpec(name="StaticArrays")
			])

	using PlutoUI
	using Colors, ColorSchemes, Images
	using Plots, OffsetArrays, SparseArrays
	using StaticArrays
	using LaTeXStrings
	
	using Statistics, LinearAlgebra  # standard libraries
end

# ╔═╡ 454494b5-aca5-43d9-8f48-d5ce14fbd5a9
md"### Soft body section"

# ╔═╡ 58e8e7c4-7122-47e2-8529-1b761ebb8177
mutable struct Atom
	x::SVector{3,Int64}
	v::SVector{3,Int64}
	F::SVector{3,Int64}
	μ::Float64
	m::Float64
	I::Float64
end

# ╔═╡ bb0c8c15-efcc-44da-ad4a-4ca1c7ef9583
function makeAtom()
	a = Atom([1. 0 0], [1. 0 0], [1. 0 0], 0.0, 0,0)
	
	a.x = [1. 0 0]
	a.v = [0 2. 0]
	a.F = [0 3. 0]
	a.μ = .3
	a.m = 10.0
	a.I = 1.2
	
	return a
end

# ╔═╡ c08a6bf4-1b23-4fa6-b013-a8f8400b9cae
begin
	natoms_c = 70
	natoms_r = 9
	Δa = 1 #  interatomic distance on same axis
	Δt = .001
	
	Default_Atom_Intensity = 800.0
	
	Niter_euler = 12001
	
	m = 10
	mu = .11
	natoms = natoms_c * natoms_r

		apx = OffsetArray(zeros(natoms_r+2, natoms_c+2, Niter_euler+1 ), 0:natoms_r+1,0:natoms_c+1, 0:Niter_euler)
	
		apy = similar(apx)
	
		avx = similar(apx)
		avy = similar(apx)		
	
	
		aI  = similar(apx)  # Atom "intensity" (makes Klink as product of intensities divided by rest-length)
		aE  = similar(apx)  # Atom "energy level" (sum abs(forces))
	
	
		aFx = similar(apx)
		aFy = similar(apx)	
	
end;

# ╔═╡ 61f3e46a-2f2f-4028-b59a-4fd939b13eea
begin
function draw_scatter()	

#scatter(apx[1:natoms_r, 1:natoms_c][:], apy[1:natoms_r, 1:natoms_c][:], m = (:heat, 0.8, Plots.stroke(1, :green)), ms = 9 .* aI.+ 0, lab = false)

		ys = apy[1:natoms_r, 1:natoms_c, end-1][:]
		xs = apx[1:natoms_r, 1:natoms_c, end-1][:]
	
plot(xs, ys, color = [:black :orange], line = (1), marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1   )		
		
end
end	

# ╔═╡ 6cf8315f-8fd5-4277-854e-d5aa1e9adbfb
function creategrid()

for t = 0:1
	
for i = 0:natoms_r+1, j = 0:natoms_c+1  # create grid
	
	apx[i,j, t] = j * Δa
	apy[i,j, t] = i * Δa

	avx[i,j, t] = 0.0
	avy[i,j, t] = 0.0		
		
	aFx[i,j, t] = 0.0
	aFy[i,j, t] = 0.0
			
	aE[i,j, t] = 0.0		# Reset atom energy level	
					
end #for i,j
		
end # next time		

	
aI[1:natoms_r, 1:natoms_c, 0:Niter_euler] .= Default_Atom_Intensity		
	
	
draw_scatter()		
	
end

# ╔═╡ b07a27ea-b155-42b3-a1e6-0a31ef2d9028
creategrid();

# ╔═╡ 5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
begin
	#function apply_forces()
	creategrid()
	
	for t in 1:Niter_euler-1
	
	for i = 1:natoms_r, j = 1:natoms_c  
		
		aFy[i,j, t] = -m * 9.8
		aFx[i,j, t] = 0
		
		indices_hv =   [(-1,0)  (0,-1) (0,1)  (1,0)]
	    indices_diag = [(-1,-1) (-1,1) (1,-1) (1,1)]		
			
		indices = [indices_hv indices_diag]
			
		for ind in indices
				
			# calculate local stiffness
			
			rest_length = abs(ind[1]) + abs(ind[2]) == 1 ? Δa : Δa * √2	
				
			Klink = aI[i,j, t] * aI[i+ind[1],j+ind[2], t] / rest_length

			rest_length = ind[1] * ind[2] == 0 ? Δa : Δa * √2

			Δx = (apx[i,j, t] - apx[i+ind[1],j+ind[2], t]) 
			Δy = (apy[i,j, t] - apy[i+ind[1],j+ind[2], t]) 
				
			distance = √(  Δx^2 + Δy^2 )
				
			extension = distance - rest_length
				
			force = extension * Klink
				

			aFx[i,j, t] += -1* force * Δx/distance
			aFy[i,j, t] += -1* force * Δy/distance		
				
			aE[i,j, t] += sum(abs.([ aFx[i,j, t], aFx[i,j, t] ]))
				
			end # for indices
		
	end # for i, j
		
	
# F = m * Δv / Δt   ->   Δv = F/m * Δt
# v = Δx / Δt    ->   Δx = v * Δt  ->    Δx = F/m * Δt ^2
		
		
 
# Strönberg
for i = 1:natoms_r, j = 1:natoms_c  					
		apx[i,j, t+1] = 2*apx[i,j, t] - apx[i,j, t-1] + aFx[i,j, t] * Δt^2 /m 
		apy[i,j, t+1] = 2*apy[i,j, t] - apy[i,j, t-1] + aFy[i,j, t] * Δt^2 /m	
end		


"""		
# ñapa				
for i = 1:natoms_r, j = 1:natoms_c  					
		apx[i,j, t+1] = apx[i,j, t]  + aFx[i,j, t] * Δt^2 /m 
		apy[i,j, t+1] = apy[i,j, t]  + aFy[i,j, t] * Δt^2 /m	
end		
"""
				
		
		
apx[1,1, t+1] = 1
apy[1,1, t+1] = 1
		
apy[1,natoms_c, t+1] = 1		
		
		
end	# next time    -- function
	
draw_scatter()	
end

# ╔═╡ 30d5a924-7bcd-4eee-91fe-7b10004a4139
begin
	
	@gif for t in 1:(Int64(floor(Niter_euler/100))):Niter_euler-1

		ys = apy[1:natoms_r, 1:natoms_c, t][:]
		xs = apx[1:natoms_r, 1:natoms_c, t][:]
		Es = aE[1:natoms_r, 1:natoms_c, t][:]
		
	
plot(xs, ys, color = [:black :orange], line = (1), marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, zcolor = Es   )		
	
	end
	
end

# ╔═╡ e95fe0fb-c52c-4636-ad2c-fa6d3bbef926
apy

# ╔═╡ 40b439ff-0780-410f-b4f5-b59864f4f7f8
apx[:]

# ╔═╡ 0b4e83a6-d8e2-48ea-a9bd-1f15c7b3580e
apy[:]

# ╔═╡ 1c090881-4de5-4a5b-b4fc-4f3abf69e1d6


# ╔═╡ c0fd4cf5-99dd-4cb6-87e6-62fc5e3a553e
aFy

# ╔═╡ b7c8d956-f723-4a8d-9195-88ffb67f5774


# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### Version  v 0.0.10 OK
- 0.0.8 Back to original formulation in 88 lines after attempt to reorder elements in v 0.0.6
- 0.0.8 OK works in obtaining a meaningful internal loads field

- 0.0.9 Clean up of 0.0.8OK   5 APR 21. It works with canonical problem

- 0.0.10 Added animation and additional clean-up. GAUSS FILTER ADDED, IT WORKS FINE

"""

# ╔═╡ 965946ba-8217-4202-8870-73d89c0c7340
md"""
### Global Parameters
"""

# ╔═╡ 6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
begin 
	
# Set global parameters
	
sigma_all	= 3.5
max_all_t = 5
full_penalty_iter = 5
max_penalty = 5
thick_ini = 1.0		
min_thick = 0.00001
		
scale = 1
nelx = 60*scale ; nely = 20*scale  #mesh size

Niter = 35

end;

# ╔═╡ b23125f6-7118-4ce9-a10f-9c3d3061f8ce
md"""
### Setup model
"""

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin

# Setup models

nDoF = 2*(nely+1)*(nelx+1)  # Total number of degrees of freedom
	
F = zeros(Float64, nDoF)	# Initialize external forces vector
F[2] = -1.0	   # Set applied external force
	
U = zeros(Float64, nDoF)	# Initialize global displacements
	
fixeddofs = [(1:2:2*(nely+1)); nDoF ]  # Set boundary conditions
	
freedofs  = setdiff([1:nDoF...],fixeddofs)	

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx)
	
edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:]	

# edofMat = repeat(edofVec,1,8) + repeat([0 1 2*nely.+[2 3 0 1] -2 -1],nelx*nely) ORIG 88 lines

edofMat = repeat(edofVec,1,8) + repeat([-1 -2 1 0 2*nely.+[3 2 1 0]],nelx*nely)	# order changed to match a K of a 2x1 mesh ftom 99 lines
	
#edofMat =  [ 2 1 4 3 8 7 6 5]	# corresponding to a single element
	
iK = kron(edofMat,ones(Int64,8,1))'[:]
jK = kron(edofMat,ones(Int64,1,8))'[:]
	

end;

# ╔═╡ 7ae886d4-990a-4b14-89d5-5708f805ef93
md"""
#### Call FSDTOPO with Niter
"""

# ╔═╡ 52b66b2c-c68e-41eb-aff6-46234e23debf
(a, b) = 5 > 13 ? (3, 4) : (8,9)

# ╔═╡ 5124345d-2b5f-41fd-b2b3-a2c7cbb832bb
b

# ╔═╡ 2bfb23d9-b434-4f8e-ab3a-b598701aa0e6
md"""
N = $(@bind N Slider(1:25, show_value=true, default=1))
"""

# ╔═╡ 6bd11d90-93c1-11eb-1368-c9484c1302ee
md""" ### FE SOLVER FUNCTIONS  """

# ╔═╡ d108d820-920d-11eb-2eee-bb6470fb4a56
md"""
### AUXILIARY FUNCTIONS
"""

# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
function KE_CQUAD4()
# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
		
A = -5.766129E-01; B = -6.330645E-01 ; C =  2.096774E-01 ; D = 3.931452E-01	; G = 3.024194E-02	

KE = [ 	[ 1  D  A -G   B -D  C  G];
		[ D  1  G  C  -D  B -G  A];
		[ A  G  1 -D   C -G  B  D];
		[-G  C -D  1   G  A  D  B];
		[ B -D  C  G   1  D  A -G];
		[-D  B -G  A   D  1  G  C];
		[ C -G  B  D   A  G  1 -D];
		[ G  A  D  B  -G  C -D  1]		
		]'	
end	
	

# ╔═╡ a8c96d92-aee1-4a91-baf0-2a585c2fa51f
begin

function NODAL_DISPLACEMENTS(thick)

KE = KE_CQUAD4() # Local element stiffness matrix
		
sK = reshape(KE[:]*thick[:]', 64*nelx*nely)
		
K = Symmetric(sparse(iK,jK,sK));
		
U[freedofs] = K[freedofs,freedofs]\F[freedofs]		
				
end # function
	
end

# ╔═╡ c652e5c0-9207-11eb-3310-ddef16cdb1ac
#heatmap(reverse(KE_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ c1711000-920b-11eb-14ba-eb5ce08f3941
function SU_CQUAD4()
# Matrix relating cartesian stress components (sxx, syy, sxy) with nodal displacements in CQUAD4 element, reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
		
A = -1.209677E+00 ; B = -3.629032E-01 ; C = -4.233871E-01   	

KE = [ 	[ A  B  C ];
		[ B  A  C ];
		[-A -B  C ];
		[ B  A -C ];
		[-A -B -C ];
		[-B -A -C ];
		[ A  B -C ];
		[-B -A  C ];
		]'	

end	

# ╔═╡ 2c768930-9210-11eb-26f8-0dc24f22afaf
begin

function INTERNAL_LOADS(thick)
		
NODAL_DISPLACEMENTS(thick)	# First solve for global displacements
		
		
SUe = SU_CQUAD4() # Matrix that relates element stresses to nodal displacements
		
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)
				
@inbounds for y = 1:nely, x = 1:nelx # Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
			
	n2 = (nely+1)* x +y	; 	n1 = n2	- (nely+1)
			
	Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
		
	Te = (SUe * Ue) .* nelx  # Element stress vector in x, y coordinates. Scaled by mesh size
	sxx = Te[1] ; syy = Te[2] ; sxy = Te[3]
	
	# Principal stresses
	s1 = 0.5 * (sxx + syy + ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
	s2 = 0.5 * (sxx + syy - ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
	res = (s1 ^ 2 + s2 ^ 2 - 2 * 0.3 * s1 * s2) ^ 0.5		# elastic strain energy

	S[y, x] = res
		
end # for	
	
	return S	
end # function	
	
end

# ╔═╡ 87be1f09-c729-4b1a-b05c-48c79039390d
begin

function FSDTOPO(niter)	
		
		
th = OffsetArray( zeros(Float64,1:nely+2,1:nelx+2), 0:nely+1,0:nelx+1) # Initialize thickness canvas with ghost cells as padding
th[1:nely,1:nelx] .= thick_ini	# Initialize thickness distribution in domain		

t = view(th, 1:nely,1:nelx) # take a view of the canvas representing the thickness domain			

t_res = []					
		
for iter in 1:niter
			
	t .*= INTERNAL_LOADS(t)	 / sigma_all # Obtain new thickness by FSD algorithm
	
	t = [min(nt, max_all_t) for nt in t] # Limit thickness to maximum

			
			
	penalty = min(1 + iter / full_penalty_iter, max_penalty) # Calculate penalty at this iteration
			
	# Filter loop					

"""			
t = [sum(th[i.+CartesianIndices((-1:1, -1:1))]
				.*( [1 2 1 ;
				   2 4 2 ;
				   1 2 1] ./16)
		) for i in CartesianIndices(t)]						
"""		

if penalty < max_penalty * 1			

for gauss in 1:scale  # apply spatial filter as many times as scale in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)	
				
	for j = 1:nely, i in 1:nelx  # *** CHECK WHETHER INDICES ARE SWAPPED IN ALL CODE, EXPLAINING WHY DoFs 1 AND 2 HAD TO BE SWAPPED WHEN BUILDING K FROM Ke

	(NN_t, NN_w) = (j > 1) ? (t[j-1, i], 2) : (0,0)
	(SS_t, SS_w) = (j < nely) ? (t[j+1, i], 2) : (0,0)							

	(WW_t, WW_w) = i > 1 ? (t[j, i-1], 2) : (0,0)
	(EE_t, EE_w) = i < nelx ? (t[j, i+1], 2) : (0,0)					

	(NW_t, NW_w) = ((j > 1) && (i > 1)) ? (t[j-1, i-1], 1) : (0,0)
	(NE_t, NE_w) = ((j > 1) && (i < nelx)) ? (t[j-1, i+1], 1) : (0,0)				

	(SW_t, SW_w) = ((j < nely) && (i > 1)) ? (t[j+1, i-1], 1) : (0,0)				
	(SE_t, SE_w) = ((j < nely) && (i < nelx)) ? (t[j+1, i+1], 1) : (0,0)				

	t[j,i] = (t[j,i]*4 + NN_t * NN_w + SS_t * SS_w + EE_t * EE_w + WW_t * WW_w + NE_t* NE_w + SE_t * SE_w + NW_t * NW_w + SW_t * SW_w)/(4 + NN_w+ SS_w+ EE_w+ WW_w+ NE_w+ SE_w+ NW_w+ SW_w)			

	end # for j, i			
					
end # for gauss					
					
					
end # if		


		
			
tq = [max((max_all_t*(min(nt,max_all_t)/max_all_t)^penalty), min_thick) for nt in t]


t = copy(tq)  # ??? WHY IS THIS NECESSARY? OTHERWISE HEATMAP DISPLAYS A THICKNESS MAP WITH A MAXIMUM THICKNESS LARGER THAN THE SPECIFIED BOUND
			
push!(t_res, tq)			
			
end	# for	
		
return t_res # returns an array of the views of the canvas containing only the thickness domain for each iteration
		
end # end function
	
end


# ╔═╡ d007f530-9255-11eb-2329-9502dc270b0d
 newt = FSDTOPO(Niter);

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
heatmap(reverse(newt[N], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
begin
	
	@gif for i in 1:Niter	
		heatmap([ reverse(newt[i], dims=(1,2)) reverse(newt[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))
	end
	
end;

# ╔═╡ c58a7360-920c-11eb-2a15-bda7ed075812
#heatmap(reverse(SU_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ c72f9b42-94c7-4377-85cd-5afebbe1d271
md"""
### NOTEBOOK SETUP
"""

# ╔═╡ 13b32a20-9206-11eb-3af7-0feea278594c
TableOfContents(aside=true)

# ╔═╡ Cell order:
# ╟─454494b5-aca5-43d9-8f48-d5ce14fbd5a9
# ╟─58e8e7c4-7122-47e2-8529-1b761ebb8177
# ╟─bb0c8c15-efcc-44da-ad4a-4ca1c7ef9583
# ╠═c08a6bf4-1b23-4fa6-b013-a8f8400b9cae
# ╠═6cf8315f-8fd5-4277-854e-d5aa1e9adbfb
# ╠═b07a27ea-b155-42b3-a1e6-0a31ef2d9028
# ╠═61f3e46a-2f2f-4028-b59a-4fd939b13eea
# ╠═5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
# ╠═30d5a924-7bcd-4eee-91fe-7b10004a4139
# ╠═e95fe0fb-c52c-4636-ad2c-fa6d3bbef926
# ╠═40b439ff-0780-410f-b4f5-b59864f4f7f8
# ╠═0b4e83a6-d8e2-48ea-a9bd-1f15c7b3580e
# ╠═1c090881-4de5-4a5b-b4fc-4f3abf69e1d6
# ╠═c0fd4cf5-99dd-4cb6-87e6-62fc5e3a553e
# ╠═b7c8d956-f723-4a8d-9195-88ffb67f5774
# ╠═d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─965946ba-8217-4202-8870-73d89c0c7340
# ╠═6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╠═52b66b2c-c68e-41eb-aff6-46234e23debf
# ╠═5124345d-2b5f-41fd-b2b3-a2c7cbb832bb
# ╟─2bfb23d9-b434-4f8e-ab3a-b598701aa0e6
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╠═7f47d8ef-98be-416d-852f-97fbaa287eec
# ╟─6bd11d90-93c1-11eb-1368-c9484c1302ee
# ╠═a8c96d92-aee1-4a91-baf0-2a585c2fa51f
# ╠═2c768930-9210-11eb-26f8-0dc24f22afaf
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╠═c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╟─c1711000-920b-11eb-14ba-eb5ce08f3941
# ╠═c58a7360-920c-11eb-2a15-bda7ed075812
# ╠═c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╠═fc7e00a0-9205-11eb-039c-23469b96de19
# ╠═13b32a20-9206-11eb-3af7-0feea278594c
