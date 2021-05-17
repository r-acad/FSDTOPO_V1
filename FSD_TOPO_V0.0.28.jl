### A Pluto.jl notebook ###
# v0.14.5

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
	
end

# ╔═╡ 66ba9dc4-1d50-410a-acdd-850c8f27fd3d
md"""
### LARGE DISPLACEMENTS VERLET SOLVER
Version 0.0.14 It works with Störmer solution
Stable version 2021 04 26
Next versions will try to implement RK5

Version 0.0.15. Implementation of MBB problem. Not yet converged, best integration method is Verlet-Störmer

Version 0.0.16. Verlet with velocity and simplification of matrices to remove time dependency of mass, stiffness etc (these will be updated in place when the topo algo is implemented)

Version 0.0.17. Change to Mass as the source of elastic force, remove broadcasting and go back to nested loops

v0.0.21 Optimization of for loops and static array of offsets. Implementation of Threaded loops

V0.0.22 to 0.0.23 failed tests

v0.0.24 WORKS FULLY. BASELINE

v0.0.25 Start of the implementation of nonlinear FSDTOPO

v0.0.26 was a failed attempt to use staticarrays but the size of the target matrices was too large


"""

# ╔═╡ 4da7c9e7-6997-49a1-92bc-462d247f4e12
Threads.nthreads()

# ╔═╡ 454494b5-aca5-43d9-8f48-d5ce14fbd5a9
md"## Non-linear implementation"

# ╔═╡ 10ececaa-5ac8-4870-bcbb-210ffec09515
begin # SET CONSTANTS
		
	const explicit_scale = 2 # Scale multiplier on basic lattice dimensions
	const Δt = .01   # Time step for integration
		
	const Δa = 1.0    #  interatomic distance on same axis
	
	const Niter_ODE = 20000 # Number of iterations in solver
		
	const initial_mass = 3. # Initial atom mass
	
	#const mu =  .7 # Initial atom viscous damping coefficient	
	
	const num_frict = 0.0025 # Friction coefficient in Verlet with friction
	
	const G = 9.81 # Acceleration of gravity
	
	const sigma_all_nolin = 0.3
	
end;

# ╔═╡ 0d026d8d-75a4-4fd5-a95e-b815963c468e
md"""### Allocate Arrays in Nolin"""

# ╔═╡ 402abadb-d500-4801-8005-11d036f8f351
begin # ALLOCATE ARRAYS

const ndims = 2 # Number of dimensions of the lattice

const natoms_c = 6 * explicit_scale # Number of columns of atoms in lattice
const natoms_r = 2 * explicit_scale # Number of rows of atoms in lattice	
	
# Array of atom positions at all times, initialized to 0 and used as template for other matrices. The offset array is to provide a "frame" of one element on each margin in order to facilitate the convolution operations
a_x = OffsetArray(zeros(Float64,ndims,natoms_r+2,natoms_c+2,Niter_ODE+1),
	  1:ndims, 0:natoms_r+1, 0:natoms_c+1, 0:Niter_ODE) 
	
#a_v = copy(a_x)  # Array of velocities of all atoms for all times
a_F = copy(a_x) # Array of sum of forces acting on each atom for all times

# Array of atom "masses" (stiffness of the link between atoms calculated as if masses were springs in series, and then divided by rest-length) at current topo iteration	
a_M = OffsetArray(zeros(Float64,natoms_r+2,natoms_c+2), 0:natoms_r+1, 0:natoms_c+1)  
	
# Array of atom "energy level" (sum abs(forces)) at current topo iteration
a_E = ones(Float64,natoms_r,natoms_c, Niter_ODE+1) 
	
# Array with indice offsets of neighbors, first 4 are same-axis, next 4 are diagonals on the same plane. First two elements are index offset and third is link rest length. The 4th element is the weight of the contribution of the offset atom to the regularization filter
neighbors =  @SVector [
		(-1,  0, Δa, 2.0), (0, -1, Δa, 2.0) , (0, 1, Δa, 2.0),  (1, 0, Δa, 2.0) , 
        (-1,-1,Δa*√2,1.0), (-1,1,Δa*√2,1.0), (1,-1,Δa*√2,1.0), (1,1,Δa*√2, 1.0)]

# Array with nominal link stiffness between each node and its neighbors for a given state of the mass matrix (at each topo iteration)	
Klink = zeros(Float64,natoms_r,natoms_c, length(neighbors))	
		
# Array holding the number of active elastic connections of an atom with neighbors (frame elements are considered non-active). Used for normalization of "energy levels"
n_links = zeros(Int64,natoms_r,natoms_c) 

end;

# ╔═╡ 4db99657-4f4c-47ed-9639-b521d03e45c9
md"""### Initialize Grid and State Matrices"""

# ╔═╡ 8a5761b3-9554-4e48-bb4c-60393baadb3a
function initialize_grid() # INITIALIZE ARRAYS FOR A FRESH NON-LINEAR SOLUTION

# create grid in i,j and sweep through dimensions	
@inbounds  for j = 0:natoms_c+1, i = 0:natoms_r+1 , dim = 1:ndims
	a_x[dim, i,j, :] .= dim == 1 ? Float64(j * Δa) : Float64(i * Δa)
end #for i,j

# set non-zero masses only in the grid, let intensities of canvas frame = 0	
a_M[1:natoms_r, 1:natoms_c] .= initial_mass	

a_E .= 0.0   # Reset initial atom elastic energy
#a_v .= 0.0  # Reset initial atom velocities
a_F .= 0.0  # Reset initial atom forces

# Calculate number of active links on i,j node in order to normalize energy	
@inbounds Threads.@threads for j = 1:natoms_c
	@inbounds Threads.@threads for i = 1:natoms_r
		@inbounds for neigh in 1:length(neighbors) # Check neighbors
		# Calculate number of active links on current atom
		n_links[i,j] += Int(a_M[i+neighbors[neigh][1],j+neighbors[neigh][2]] > 0)
				
		Klink[i,j, neigh] = 30.0 / (1/a_M[i,j] + 1/a_M[i+neighbors[neigh][1], j+neighbors[neigh][2]] ) / neighbors[neigh][3] # link stiffness calculated as springs in series with individual stiffness corresponding to the mass of the i,j node and its corresponding neighbor, corredted by its rest length (= neigh[3])
				
		end	# next neighbour			
	end	# next i	
end # next j
	
end;

# ╔═╡ d37fd3f6-49cb-4738-9536-21ac6212c749
@inline function modulate(n, fulliter)

min(n/fulliter, 1)
		
end

# ╔═╡ 01bfb4bc-d498-4dd4-b2a8-f6a5e59f8ae4
@inline function apply_boundary_conditions(t)
# Boundary conditions		
	
""" #Cantilever beam	
a_x[1:ndims,1,1, t] .= 1.0
a_x[1:ndims,1,natoms_c, t] .= (Float64(natoms_c) , 1.0)
"""

a_x[1,:,1, t] .= 1.0 # Simmetry condition
	
a_x[2,1,natoms_c, t] = 1.0	# simple support in y on the right lower corner
	
end

# ╔═╡ 7aefbb01-1e15-4aa1-9a49-b182e6723764
function solve_nolin()

a_x[:,:,:,:] .= a_x[:,:,:,Niter_ODE-2] 	# Reset initial atom positions for next topo iteration to the converged value of the previous sizing iteration (or zero if this is the first sizing iteration)						

#a_M[1:natoms_r, 1:natoms_c] .= initial_mass		
	
@inbounds for n in 1:Niter_ODE-1  # Time step	

apply_boundary_conditions(n) # at each iteration
	
# Apply external forces on lattice
a_F[2,natoms_r,1,n] += - 1. * modulate(n, Niter_ODE*.2) 	
		
@inbounds Threads.@threads for j = 1:natoms_c # Traverse colums after rows
	      Threads.@threads for i = 1:natoms_r # Traverse rows first
				
# Apply gravitational forces ("external", body force)
#a_F[2,:,:,t] += - a_m[2,:,:, t] * G * amplitude # Use atom mass at time t								
# Compute elastic forces at node i,j		
@inbounds for neigh in 1:length(neighbors) # Traverse neighbors to get their elastic actions
			
# Stiffness of the link modulated at each integration iteration
klink = modulate(n, Niter_ODE*.1) * Klink[i,j, neigh]	
				
# Relative position vector of adjacent atom at offset wrt current [i,j]				
rel_pos_vec = (a_x[:,i+neighbors[neigh][1],j+neighbors[neigh][2],n] - a_x[:, i,j, n])		
rel_pos_direction = normalize(rel_pos_vec) # Unit vector from Atom to neighbor
				
# Elastic force (scalar) between i,j atom and atom at offset
force = (norm(rel_pos_vec) - neighbors[neigh][3]) * klink # Force = Extension * stiffness
			
# Build elastic force vector acting on atom i,j from atom at i,j+offset at time t
a_F[:, i,j, n] .+= force * rel_pos_direction[:]	# Elastic force vector

# Accumulate "elastic energy" status of atom i,j at this iteration adding forces from  neighbors 
a_E[i,j, n] += abs(force) / n_links[i,j] #norm(a_F[:, i,j, n])			

			
# Internal damping force			
#scalar_relative_axial_velocity = (a_v[:, i+offset[1],j+offset[2], n] - a_v[:, i,j, n]) * rel_pos_direction' # scalar product				
	
#a_F[:, i,j, n] += mu * scalar_relative_axial_velocity * rel_pos_direction				
end # for offset			
			

# Drag force at i,j 
#a_F[1:ndims,:,:,n] .+= -mu*(norm(a_v[1:ndims,:,:,n]) .* a_v[1:ndims,:,:,n])			
			
# Verlet integration with artificial friction damping			
fr = num_frict #* modulate(n, Niter_ODE*.25) # Calculate friction coeff. at this iter_n
				
a_x[:, i,j, n+1] = (2-fr) * a_x[:, i,j, n] - (1-fr)* a_x[:, i,j, n-1] + a_F[:, i,j, n] * Δt^2 / a_M[i,j] 		
				
#a_v[dim, i,j, n+1] = (a_x[dim, i,j, n+1] - a_x[dim, i,j, n]) / (Δt)
		
end # next j
end # next i
				
end	# next step	

a_E[:,:, end-2]	# return the array of atom energies
	
end	

# ╔═╡ 39dafb9a-5f99-40eb-b98f-6d0071b20827
a_M

# ╔═╡ 8914feb6-bf7c-46a7-9d6c-7c54538802f5
md"""
N2 = $(@bind N2 Slider(1:Niter_ODE-2, show_value=true, default=1))
"""

# ╔═╡ fd6b2897-ff79-4ea3-b662-3eca9b8755d1
heatmap(a_E[:,:, N2], aspect_ratio = 1)

# ╔═╡ 8ba10b72-027d-4266-a002-b1b6bbe0c8d5
heatmap(a_M[0:natoms_r+1, 0:natoms_c+1], aspect_ratio = 1	)

# ╔═╡ d7469640-9b09-4262-b738-29810bd19305
plot([ a_x[2, 1,5, 1:end]     ])

# ╔═╡ a755dbab-6ac9-4a9e-a397-c47efce4d2f7
begin
function draw_scatter()	
	
plot(a_x[1,1:natoms_r, 1:natoms_c, end-1][:], 
	 a_x[2, 1:natoms_r, 1:natoms_c, end-1][:], 
	 color = [:black :orange], line = (1), 
	 marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, 
			zcolor = log.(a_E[1:natoms_r, 1:natoms_c, end-1][:])  )		
		
end
end	

# ╔═╡ 6960420d-bc50-4be3-9a26-2f43f14b903d
function draw_animated_heatmap(fn)  # fn= identity for no transformation, log for logarthmit transformation of values
	
	@gif for Nit in 1:(Int64(floor(Niter_ODE/500))):Niter_ODE-1
		
	heatmap( fn.(a_E[1:natoms_r, 1:natoms_c, Nit])
			, aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), clims=(0.0, 0.3))
	
	end
	
end

# ╔═╡ 30d5a924-7bcd-4eee-91fe-7b10004a4139
function draw_animation()
	
@gif for t in 1:(Int64(floor(Niter_ODE/100))):Niter_ODE-1
		
#@gif for t in 1:Niter_ODE-1		

plot(a_x[1,1:natoms_r, 1:natoms_c, t][:], 
	 a_x[2, 1:natoms_r, 1:natoms_c, t][:], 
	 color = [:black :orange], line = (1), 
	 marker = ([:hex :d], 6, 0.5, Plots.stroke(.5, :green)), leg = false, aspect_ratio = 1, 
	zcolor = a_E[1:natoms_r, 1:natoms_c, t][:]  )					
	end # for time step
	
end

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
		
scale = 20
nelx = 6*scale ; nely = 2*scale  #mesh size

Niter = 25

full_penalty_iter = Niter*.1
end;

# ╔═╡ 8940ead8-cf2a-440e-ab7b-cc1919ae996d
begin

function FSDTOPO_Nolin(niter_TOPO)	
		
t = view(a_M, 1:natoms_r, 1:natoms_c) # Lattice thicknesses at this topo iteration 
		
t_res = []	# Initialize array of intermediate thicknesses of each iteration
		
for iter in 1:niter_TOPO  # Iterate with TOPO process

t .*= solve_nolin()	 / sigma_all_nolin # Obtain new thickness by FSD algorithm	
t = [min(nt, initial_mass) for nt in t] # Limit thickness to a maximum equal to the initial "thickness" (atom mass)
			
penalty = min(1 + iter / full_penalty_iter, max_penalty) # Calculate penalty at this iteration
			
if penalty < 0 #max_penalty * 1			
# Filter loop, mesh regularization
for gauss in 1:1 #scale  # apply spatial filter as many times as scale in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)	
@inbounds Threads.@threads for j = 1:natoms_r
@inbounds Threads.@threads for i in 1:natoms_c  

kernel_denominator = 4 # Initialize kernel denominator with central weight of mask
weighted_thickness = 4 * t[j,i] # same as above for weighted thickness
							
@inbounds for neigh in 1:length(neighbors) # Traverse neighbors 
								
weighted_thickness += t[i+neighbors[neigh][1],j+neighbors[neigh][2]] * neighbors[neigh][4]
								
kernel_denominator += weighted_thickness > 0 ? neighbors[neigh][4] : 0
							
end # neighbors								
							
t[j,i] = weighted_thickness / kernel_denominator
							
end # for j
end # for i								
end # for gauss					
					
					
end # if penalty

			
tq = [max((initial_mass*(min(nt,initial_mass)/initial_mass)^penalty), min_thick) for nt in t]


t = copy(tq)  # ??? WHY IS THIS NECESSARY? OTHERWISE HEATMAP DISPLAYS A THICKNESS MAP WITH A MAXIMUM THICKNESS LARGER THAN THE SPECIFIED BOUND
			
push!(t_res, t)			
			
end	# for iter

#draw_animation()		
		
return t_res # returns an array of the views of the canvas containing only the thickness domain for each iteration
		
end # end function

	
	
	
end


# ╔═╡ 5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
begin
# MAIN control flow of no-lin program
	
initialize_grid() # Reset all matrices
	
#solve_nolin() # Solve one nonlinear problem

	
Niter_FSD_Nolin = 4	
	
tres_nolin = FSDTOPO_Nolin(Niter_FSD_Nolin)	# Solve topology optimization problem
	
draw_animation()
#draw_animated_heatmap()
	
end

# ╔═╡ 83badc22-dc5b-4b69-9f6b-c7d4825178c2
tres_nolin[3]

# ╔═╡ a1fc0684-5379-43d0-9dbd-b2efd1963e0f
md"""
Nit = $(@bind Nit Slider(1:Niter_FSD_Nolin, show_value=true, default=1))
"""

# ╔═╡ 55af748b-456b-465d-aeea-9859c36279f5
heatmap(reverse(tres_nolin[Nit], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

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


	
end;

# ╔═╡ 7ae886d4-990a-4b14-89d5-5708f805ef93
md"""
#### Call FSDTOPO with Niter
"""

# ╔═╡ b9ec0cbf-f9a2-4980-b7cd-1ecda0566631
@inline function convolution3x3(matr, kern)

# matr is convolved with kern. The key assumption is that the padding elements are 0 and no element in the "interior" of matr are = 0 (THIS IS A STRONG ASSUMPTION IN THE GENERAL CASE BUT VALID IN FSD-TOPO AS THERE IS A MINIMUM ELEMENT THICKNESS <0)	
	
canvas = OffsetArray(zeros(Float64, 1:size(matr,1)+2, 1:size(matr,2)+2), 0:size(matr,1)+1,0:size(matr,2)+1)
	
canvas[1:size(matr,1), 1:size(matr,2)] = matr

# Return the sum the product of a subarray centered in the cartesian indices corresponding to i, of the interior matrix, and the kernel elements, centered in CartInd i. Then .divide by the sum of the weights multiplied by a 1 or a 0 depending on whether the base element is >0 or not. Note: the lines below are a single expression

	"""	
[sum(canvas[i.+CartesianIndices((-1:1, -1:1))].*kern) for i in CartesianIndices(matr)]./ [sum((canvas[i.+CartesianIndices((-1:1,-1:1))] .> 0.0).*kern) for i in CartesianIndices(matr)]	
	"""

[sum( canvas[i.+CartesianIndices((-1:1, -1:1))] .* kern)	/ 
 sum((canvas[i.+CartesianIndices((-1:1, -1:1))] .> 0.0) .* kern) 
		for i in CartesianIndices(matr)]
	

end

# ╔═╡ 2bfb23d9-b434-4f8e-ab3a-b598701aa0e6
md"""
N = $(@bind N Slider(1:Niter, show_value=true, default=1))
"""

# ╔═╡ 6bd11d90-93c1-11eb-1368-c9484c1302ee
md""" ### FE SOLVER FUNCTIONS  """

# ╔═╡ d108d820-920d-11eb-2eee-bb6470fb4a56
md"""
### AUXILIARY FUNCTIONS and MATRICES
"""

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
			 			GK4  AK4  DK4  BK4  -GK4  CK4 -DK4  1]	
	
# Matrix relating cartesian stress components (sxx, syy, sxy) with nodal displacements in CQUAD4 element, reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
const AS4 = -1.209677E+00; const BS4 = -3.629032E-01; const CS4 = -4.233871E-01  	
SU_CQUAD4 = @SMatrix [ 	 AS4  BS4  -AS4 BS4 -AS4 -BS4  AS4 -BS4;
						 BS4  AS4  -BS4 AS4 -BS4 -AS4  BS4 -AS4;
						 CS4  CS4  CS4 -CS4 -CS4 -CS4 -CS4  CS4]

Gauss_3x3_kernel = @SMatrix [1.0 2.0 1.0 ;
				    		2.0 4.0 2.0 ;
				    		1.0 2.0 1.0] 	
	
end	;	

# ╔═╡ 2c768930-9210-11eb-26f8-0dc24f22afaf
function SOLVE_INTERNAL_LOADS(thick)
		
# First solve for global displacements
sK = reshape(KE_CQUAD4[:]*thick[:]', 64*nelx*nely)		
K = Symmetric(sparse(iK,jK,sK));
U[freedofs] = K[freedofs,freedofs]\F[freedofs]		
	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)
				
@inbounds Threads.@threads 	for y = 1:nely
@inbounds Threads.@threads  for x = 1:nelx # Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
			
n2 = (nely+1)* x +y	; 	n1 = n2	- (nely+1)
			
Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
		
Te = (SU_CQUAD4 * Ue) .* nelx  # Element stress vector in x, y coordinates. Scaled by mesh size
sxx = Te[1] ; syy = Te[2] ; sxy = Te[3]
	
# Principal stresses
s1 = 0.5 * (sxx + syy + ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
s2 = 0.5 * (sxx + syy - ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
S[y, x] = (s1 ^ 2 + s2 ^ 2 - 2 * 0.3 * s1 * s2) ^ 0.5 # elastic strain energy??

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
 t_iter = [min(nt, max_all_t) for nt in t_iter] 
		
# Calculate penalty at this iteration			
penalty = min(1 + iter / full_penalty_iter, max_penalty) 
			
# apply spatial filter a decreasing number of times function of the iteration number, but proportional to the scale, in order to remove mesh size dependency of solution (effectively increasing the variance of the Gauss kernel)	
		
for gauss in 1:max(0,(ceil(scale*(iter-.5*Niter)/(2*-.5*Niter)))) 
	t_iter .= convolution3x3(t_iter, Gauss_3x3_kernel)					
end # for gauss								

tq = [max((max_all_t*(min(nt,max_all_t)/max_all_t)^penalty), min_thick) for nt in t_iter]

t_iter = copy(tq)  # ??? WHY IS THIS NECESSARY? OTHERWISE HEATMAP DISPLAYS A THICKNESS MAP WITH A MAXIMUM THICKNESS LARGER THAN THE SPECIFIED BOUND
			
push!(t_res, t_iter)			
	
end	# for topo iter
		
return t_res # returns an array of the views of the canvas containing only the thickness domain for each iteration
		
end # end function

# ╔═╡ d007f530-9255-11eb-2329-9502dc270b0d
 newt = FSDTOPO(Niter);

# ╔═╡ 16547ee0-3f3b-4324-b761-14a5d51ccf24
newt

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
heatmap(reverse(newt[N], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ 7f47d8ef-98be-416d-852f-97fbaa287eec
begin
	
	anim_evolution = @animate for i in 1:Niter	
		heatmap([ reverse(newt[i], dims=(1,2)) reverse(newt[i], dims=1)], aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true), fps=3)
	end
	
end

# ╔═╡ 200a3956-d229-434b-bf25-105e34acb35b
gif(anim_evolution, "anim_fps6.gif", fps = 6)

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

# ╔═╡ Cell order:
# ╟─66ba9dc4-1d50-410a-acdd-850c8f27fd3d
# ╠═4da7c9e7-6997-49a1-92bc-462d247f4e12
# ╟─454494b5-aca5-43d9-8f48-d5ce14fbd5a9
# ╠═10ececaa-5ac8-4870-bcbb-210ffec09515
# ╟─0d026d8d-75a4-4fd5-a95e-b815963c468e
# ╠═402abadb-d500-4801-8005-11d036f8f351
# ╟─4db99657-4f4c-47ed-9639-b521d03e45c9
# ╟─8a5761b3-9554-4e48-bb4c-60393baadb3a
# ╠═d37fd3f6-49cb-4738-9536-21ac6212c749
# ╟─01bfb4bc-d498-4dd4-b2a8-f6a5e59f8ae4
# ╠═7aefbb01-1e15-4aa1-9a49-b182e6723764
# ╠═5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
# ╠═83badc22-dc5b-4b69-9f6b-c7d4825178c2
# ╠═39dafb9a-5f99-40eb-b98f-6d0071b20827
# ╟─a1fc0684-5379-43d0-9dbd-b2efd1963e0f
# ╠═55af748b-456b-465d-aeea-9859c36279f5
# ╠═8940ead8-cf2a-440e-ab7b-cc1919ae996d
# ╟─8914feb6-bf7c-46a7-9d6c-7c54538802f5
# ╠═fd6b2897-ff79-4ea3-b662-3eca9b8755d1
# ╠═8ba10b72-027d-4266-a002-b1b6bbe0c8d5
# ╠═d7469640-9b09-4262-b738-29810bd19305
# ╟─a755dbab-6ac9-4a9e-a397-c47efce4d2f7
# ╟─6960420d-bc50-4be3-9a26-2f43f14b903d
# ╟─30d5a924-7bcd-4eee-91fe-7b10004a4139
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─965946ba-8217-4202-8870-73d89c0c7340
# ╠═6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═16547ee0-3f3b-4324-b761-14a5d51ccf24
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
# ╟─b9ec0cbf-f9a2-4980-b7cd-1ecda0566631
# ╟─2bfb23d9-b434-4f8e-ab3a-b598701aa0e6
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╠═7f47d8ef-98be-416d-852f-97fbaa287eec
# ╠═200a3956-d229-434b-bf25-105e34acb35b
# ╟─6bd11d90-93c1-11eb-1368-c9484c1302ee
# ╠═2c768930-9210-11eb-26f8-0dc24f22afaf
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╠═0342ef5f-484d-4ed7-8260-b2b1330fead0
# ╠═c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╠═c58a7360-920c-11eb-2a15-bda7ed075812
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╟─fc7e00a0-9205-11eb-039c-23469b96de19
