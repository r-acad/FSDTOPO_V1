### A Pluto.jl notebook ###
# v0.14.3

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

# ╔═╡ 6104ccf7-dfce-4b0b-a869-aa2b71deccde
md"""
$$\mathbf{x}_{k+1} = \mathbf{x}_k + h \, \mathbf{f}(\mathbf{x}_k),$$
"""

# ╔═╡ c08a6bf4-1b23-4fa6-b013-a8f8400b9cae
begin
natoms_c = 40
natoms_r = 10

ndims = 2  # Number of dimensions of the lattice
	
Δa = 1 #  interatomic distance on same axis
Δt = .001
		
Default_Atom_Intensity = 400.0
		
Niter_euler = 4000
		
m = 10
mu = .11
natoms = natoms_c * natoms_r
	
a_p = OffsetArray(zeros(ndims,   natoms_r+2,   natoms_c+2,   Niter_euler+1),
			 1:ndims, 0:natoms_r+1, 0:natoms_c+1, 0:Niter_euler) # Array of positions
	
a_v = similar(a_p)  # Array of velocities

a_F = similar(a_p) # Array of forces	
	
a_I = similar(a_p[1,:,:,:]) # Array of atom "intensities" (makes Klink as product of intensities divided by rest-length)
	
a_E = similar(a_p[1,:,:,:]) # Array of atom "energy level" (sum abs(forces))
			
end;

# ╔═╡ a755dbab-6ac9-4a9e-a397-c47efce4d2f7
begin
function draw_scatter()	
	
plot(a_p[1, 1:natoms_r, 1:natoms_c, end-1][:], 
	 a_p[2, 1:natoms_r, 1:natoms_c, end-1][:], 
	 color = [:black :orange], line = (1), 
	 marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, 
	zcolor = a_E[1:natoms_r, 1:natoms_c, end-1][:]  )		
		
end
end	

# ╔═╡ cea5e286-4bc1-457f-b300-fdff62047cc4
function creategrid()

for t = 0:1  # Initialize matrices at time 0 and 1 to set boundary conditions
for dim = 1:ndims  #Sweep through dimensions		
for i = 0:natoms_r+1, j = 0:natoms_c+1  # create grid
	
	a_p[dim, i,j, t] = dim == 1 ? j * Δa : i * Δa
	a_v[dim,i,j, t] = 0.0
	a_F[dim,i,j, t] = 0.0
	a_E[i,j, t] = 0.0  # Reset atom energy level	
					
end #for i,j
end # for dim			
end # next time		

a_I[1:natoms_r, 1:natoms_c, 0:Niter_euler] .= Default_Atom_Intensity			
	
draw_scatter()		
	
end

# ╔═╡ 5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
begin

creategrid()
	
for t in 1:Niter_euler-1  # Time step	
						
# Compute Forces at time t by solving elastic and intertial equations based on the state at time t
for i = 1:natoms_r, j = 1:natoms_c # Sweep through lattice

# Gravity loads			
a_F[2,i,j, t] = -m * 9.8  # Note weights are constant now!!!
		
indices_hv =   [(-1,0)  (0,-1) (0,1)  (1,0)] # Indices of adjacent atoms in same axes
indices_diag = [(-1,-1) (-1,1) (1,-1) (1,1)] # Indices of adjacent atoms in diags
			
indices = [indices_hv indices_diag]
			
for ind in indices # For the current atom, get the elastic forces coming from neighbours
				
# calculate local stiffness: Rest length of link between atoms (it depends on whether the link is in the same axis or in a diagonal		
rest_length = ind[1] * ind[2] == 0 ? Δa : Δa * √2
								
# Stiffness of the link: product of atom intensities normalized by rest length
Klink = a_I[i,j, t] * a_I[i+ind[1],j+ind[2], t] / rest_length

# Relative position vector of adjacent atom at ind wrt current [i,j]				
rel_pos_vec = [(a_p[dim, i,j, t] - a_p[dim, i+ind[1],j+ind[2], t]) 
				for dim in 1:ndims ]
				
distance = norm(rel_pos_vec)  # Scalar distance with neighbouring atom at ind
extension = distance - rest_length
				
force = extension * Klink

for dim = 1:ndims
 a_F[dim, i,j, t] += -1* force * rel_pos_vec[dim]/distance		
end # next dim					
				
a_E[i,j, t] += sum([ a_F[dim, i,j, t]^2 for dim in 1:ndims ]    )
				
end # for indices
end # for i, j
		
	
# F = m * Δv / Δt   ->   Δv = F/m * Δt
# v = Δx / Δt    ->   Δx = v * Δt  ->    Δx = F/m * Δt ^2
# Strönberg
		
for i = 1:natoms_r, j = 1:natoms_c  					
			
for dim = 1:ndims	
 a_p[dim, i,j, t+1] = 2*a_p[dim, i,j, t] - a_p[dim, i,j, t-1] + a_F[dim, i,j, t] * Δt^2 /m 			
end # next dim				

end		

""
				
		
# Boundary conditions		
a_p[1,1,1, t+1] = 1
a_p[2,1,1, t+1] = 1
		
#a_p[2,1,natoms_c, t+1] = 1		
		
		
end	# next time    -- function
	
draw_scatter()	
end

# ╔═╡ 30d5a924-7bcd-4eee-91fe-7b10004a4139
begin
	
	@gif for t in 1:(Int64(floor(Niter_euler/100))):Niter_euler-1

		"""
		ys = apy[1:natoms_r, 1:natoms_c, t][:]
		xs = apx[1:natoms_r, 1:natoms_c, t][:]
		Es = aE[1:natoms_r, 1:natoms_c, t][:]
		
	
plot(xs, ys, color = [:black :orange], line = (1), marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, zcolor = Es   )		

		"""
plot(a_p[1, 1:natoms_r, 1:natoms_c, t][:], 
	 a_p[2, 1:natoms_r, 1:natoms_c, t][:], 
	 color = [:black :orange], line = (1), 
	 marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, 
	zcolor = a_E[1:natoms_r, 1:natoms_c, t][:]  )			
		
		
		
	end
	
end

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
# ╟─6104ccf7-dfce-4b0b-a869-aa2b71deccde
# ╠═c08a6bf4-1b23-4fa6-b013-a8f8400b9cae
# ╠═cea5e286-4bc1-457f-b300-fdff62047cc4
# ╠═a755dbab-6ac9-4a9e-a397-c47efce4d2f7
# ╠═5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
# ╠═30d5a924-7bcd-4eee-91fe-7b10004a4139
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
