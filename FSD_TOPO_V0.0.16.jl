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
			Pkg.PackageSpec(name="Plots", version="1.10"), 
			Pkg.PackageSpec(name="OffsetArrays"),
			Pkg.PackageSpec(name="SparseArrays"),
			Pkg.PackageSpec(name="StaticArrays")
			])

	using PlutoUI
	using Plots, OffsetArrays, SparseArrays
	using StaticArrays
	using Statistics, LinearAlgebra  # standard libraries
end

# ╔═╡ e3a19f76-87b8-4bb8-95fc-750a48321cfb
Fv(v) =   0 #-3 *v

# ╔═╡ 9b279318-f7d7-4ac2-b6ba-9edb83c1555f
Fe(x) = -400 * x

# ╔═╡ 1c4f4c21-8e67-4df7-bec1-1a08c9f67464
mymass = 20

# ╔═╡ 813a6e2b-4bd0-4b7a-9545-107638e8f26e
Fg(t) = mymass * 9.8

# ╔═╡ 5eb95c3c-488a-4d2f-a4df-1cd22bc189ef
f(x, u, t) = (1/mymass) * (Fv(u) + Fe(x) + Fg(t) )

# ╔═╡ d3578df6-5592-4806-9114-91c0e0d1bf35
g(x, u, t) = u

# ╔═╡ 66ba9dc4-1d50-410a-acdd-850c8f27fd3d
md"""
### LARGE DISPLACEMENTS EXPLICIT SOLVER
Version 0.0.14 It works with Stronberg solution
Stable version 2021 04 26
Next versions will try to implement RK5

Version 0.0.15. Implementation of MBB problem. Not yet converged, best integration method is Verlet-Strongman

Version 0.0.16. Verlet with velocity and simplification of matrices to remove time dependency of mass, stiffness etc (these will be updated in place when the topo algo is implemented)


"""

# ╔═╡ 454494b5-aca5-43d9-8f48-d5ce14fbd5a9
md"### Soft body section"

# ╔═╡ 6104ccf7-dfce-4b0b-a869-aa2b71deccde
md"""
$$\mathbf{x}_{k+1} = \mathbf{x}_k + h \, \mathbf{f}(\mathbf{x}_k),$$

 F = m * Δv / Δt   ->   Δv = F/m * Δt
 v = Δx / Δt    ->   Δx = v * Δt  ->    Δx = F/m * Δt ^2
"""


# ╔═╡ 10ececaa-5ac8-4870-bcbb-210ffec09515
begin
		
	explicit_scale = 2
		
	natoms_c = 6 * explicit_scale # Number of columns of atoms in lattice
	natoms_r = 2 * explicit_scale # Number of rows of atoms in lattice
	
	const Δa = 1.0    #  interatomic distance on same axis
	const Δt = .01  # Time step
						
	const Niter_ODE = 1400 # Number of iterations in solver
	
	
	
	Default_Atom_Intensity = 50.0 * explicit_scale^.5 # This will build the stiffness
	initial_mass =   10. *  explicit_scale  # Initial atom mass
	const mu = 20 * explicit_scale # Initial atom damping coefficient
	
	const G = 9.81 * 1.0
end;

# ╔═╡ 3c33576c-0320-4a1a-b375-7ccbcb177ce1
begin 
nsteps = 10_000

x = zeros(nsteps)
u = zeros(nsteps)
t = zeros(nsteps)	

t[1] = Δt
	
	
for i = 2:nsteps-2
		
t[i] = t[i-1] + Δt
		
"""
	k1 = f(x[i], u[i] , t[1])
	k2 = f(x[i], u[i] + .5 * k1 * Δt , t[i] + Δt/2)
	k3 = f(x[i], u[i] + .5 * k2 * Δt , t[i] + Δt/2)
	k4 = f(x[i], u[i] +      k3 * Δt , t[i] + Δt)
	
u[i+1] = u[i] + 1/6 * Δt *(k1 + 2*k2 + 2*k3 + k4)

		
	k1 = g(x[i], u[i] , t[1])
	k2 = g(x[i], u[i] + .5 * k1 * Δt , t[i] + Δt/2)
	k3 = g(x[i], u[i] + .5 * k2 * Δt , t[i] + Δt/2)
	k4 = g(x[i], u[i] +      k3 * Δt , t[i] + Δt)
	
x[i+1] = x[i] + 1/6 * Δt *(k1 + 2*k2 + 2*k3 + k4)
"""


"""		
u[i+1] = u[i] + Δt * f(x[i], u[i], t[i])
		
x[i+1] = x[i] + Δt * u[i]		
"""	

		
"""
x[i+1] = 2*x[i] - x[i-1] + f(x[i], u[i], t[i]) * Δt^2 		
		
u[i+1] = (x[i+1] - x[i-1]) / 2Δt	
"""
		
x[i+1] = x[i] + u[i]* Δt + f(x[i], u[i], t[i]) * Δt^2 		
		
u[i+1] = (x[i+1] - x[i]) / Δt	
		
		
	
	
end

end

# ╔═╡ 46662437-8f37-4226-b371-96a4938c44b8
plot(u)

# ╔═╡ 402abadb-d500-4801-8005-11d036f8f351
begin

ndims = 2 # Number of dimensions of the lattice
	
# Array of atom positions at all times, initialized to 0 and used as template for other matrices
a_x = OffsetArray(zeros(Float64,ndims,natoms_r+2,natoms_c+2,Niter_ODE+1),
	  1:ndims, 0:natoms_r+1, 0:natoms_c+1, 0:Niter_ODE) 
	
a_v = copy(a_x)  # Array of velocities of all atoms at all times
a_F = copy(a_x) # Array of sum of forces acting on each atom at all times

a_I = OffsetArray(ones(Float64,natoms_r+2,natoms_c+2),
	              0:natoms_r+1, 0:natoms_c+1)  # Array of atom "intensities" (makes Klink as product of intensities divided by rest-length) at current topo iteration	
	
	
a_E = ones(Float64,natoms_r,natoms_c, Niter_ODE+1) # Array of atom "energy level" (sum abs(forces)) at current topo iteration
	
a_m = copy(a_x)  # Array of atom masses at current topo iteration (mass is the same in all dimensions and all times, done this way to facilitate broadcast and remove for loops)
	
# Array with all indice offsets of neighbors, first 4 are same-axis, next 4 are diagonals on the same plane. First two elements are index offset and third is link rest length
offsets =  @SVector [
		(-1,  0, Δa), (0, -1, Δa) , (0, 1, Δa),  (1, 0, Δa) , 
		(-1, -1, Δa * √2), (-1, 1, Δa * √2) , (1, -1, Δa * √2), (1, 1, Δa * √2)]	

	
	
buffer_matrix = copy(a_x) # Array to store temporarily internal states for debuging	
	
end;

# ╔═╡ d7469640-9b09-4262-b738-29810bd19305
plot([ a_x[2, 2,5, 1:end]     ])

# ╔═╡ 01bfb4bc-d498-4dd4-b2a8-f6a5e59f8ae4
function apply_boundary_conditions(t)
# Boundary conditions		
	
""" #Cantilever beam	
a_x[1:ndims,1,1, t] .= 1.0
a_x[1:ndims,1,natoms_c, t] .= (Float64(natoms_c) , 1.0)
"""

a_x[1,:,1, t] .= 1.0	
	
a_x[2,1,natoms_c, t] = 1.0	# simple support on the right
	
	
end

# ╔═╡ e084941c-447a-41bd-bf06-59dea45af028
"""
Compute Forces acting on all atoms of the lattice at time t by solving elastic and inertial equations based on the position of the atoms and the external forces at time t
"""
function elastic_forces(t)
		
for i = 1:natoms_r, j = 1:natoms_c # Traverse complete lattice
for offset in offsets # Traverse neighbors to get their elastic actions
			
# Link stiffness: product of atom intensities normalized by rest length (= offset[3])
@inbounds Klink = a_I[i,j] * a_I[i+offset[1],j+offset[2]] / offset[3]

# Relative position vector of adjacent atom at ind wrt current [i,j]				
@inbounds rel_pos_vec = (a_x[:, i+offset[1],j+offset[2], t] - a_x[:, i,j, t])		
			
# Elastic force (scalar) between i,j atom and atom at ind
force = (norm(rel_pos_vec) - offset[3]) * Klink # Extension x stiffness
			
# Build elastic force vector acting on atom i,j from atom at i,j+offset at time t
a_F[:, i,j, t] += force * normalize(rel_pos_vec)[:]	# Elastic force	

# Update "elastic energy" status at atom i,j at time t
a_E[i,j, t] += norm(a_F[:, i,j, t])				

end # for offset (elastic forces created by neighbours)
end # for i, j	
	
end

# ╔═╡ 96740b63-f5d6-4721-8722-37baae48f47b
function damping_forces(t)
"""
for i = 1:natoms_r, j = 1:natoms_c # Traverse complete lattice
for offset in offsets # Traverse neighbors to get their elastic actions	

# Internal damping force	

rel_pos_vec = (a_x[:, i+offset[1],j+offset[2], t] - a_x[:, i,j, t])			
			
scalar_relative_axial_velocity = (a_v[:, i+offset[1],j+offset[2], t-1] - a_v[:, i,j, t-1]) * normalize(rel_pos_vec)' # scalar product				
	
a_F[:, i,j, t] += 1. * mu * scalar_relative_axial_velocity * normalize(rel_pos_vec)	

end # for offset (elastic forces created by neighbours)
end # for i, j				
"""			
			
			
# Drag force at i,j 
a_F[1:ndims,:,:,t] .+= -10.0*mu*(norm(a_v[1:ndims,:,:,t]) .* a_v[1:ndims,:,:,t])
	
end

# ╔═╡ 0b3749f4-846f-4c66-858b-c560e0e27bca
function external_forces(t)

# Apply gravitational forces ("external", body force)
#a_F[2,:,:,t] += - a_m[2,:,:, t] * G  # Use atom mass at time t		

	
a_F[2,natoms_r,1,t] += - 4000.  # Use atom mass at time t		
	
	
end	

# ╔═╡ 30d5a924-7bcd-4eee-91fe-7b10004a4139
function draw_animation()
	
	@gif for t in 1:(Int64(floor(Niter_ODE/100))):Niter_ODE-1

plot(a_x[1,1:natoms_r, 1:natoms_c, t][:], 
	 a_x[2, 1:natoms_r, 1:natoms_c, t][:], 
	 color = [:black :orange], line = (1), 
	 marker = ([:hex :d], 6, 0.5, Plots.stroke(3, :green)), leg = false, aspect_ratio = 1, 
	zcolor = a_E[1:natoms_r, 1:natoms_c, t][:]  )					
	end # for time step
	
end

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
function draw_animated_heatmap()
	
	@gif for t in 1:(Int64(floor(Niter_ODE/100))):Niter_ODE-1
		
	heatmap( log.(a_E[1:natoms_r, 1:natoms_c, t])
			, aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))
	
	end
	
end

# ╔═╡ a8011889-d844-4c98-bd3f-014e7eb58254
draw_animated_heatmap()

# ╔═╡ cea5e286-4bc1-457f-b300-fdff62047cc4
function initialize_grid()

# create grid in i,j and sweep through dimensions	
for dim = 1:ndims , i = 0:natoms_r+1, j = 0:natoms_c+1  
	a_x[dim, i,j, :] .= dim == 1 ? Float64(j * Δa) : Float64(i * Δa)
end #for i,j

# set non-zero intensities only in the grid, let intensities of canvas margins = 0	
a_I[1:natoms_r, 1:natoms_c] .= Default_Atom_Intensity	
	
a_m .= initial_mass   # Reset initial atom masses
a_E .= 0.0   # Reset initial atom elastic energy
a_v .= 0.0  # Reset initial atom velocities
a_F .= 0.0  # Reset initial atom forces

	
end

# ╔═╡ 5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
begin

# INTEGRATE EQUATIONS OF MOTION AND SET BOUNDARY CONDITIONS	
	
initialize_grid() # Reset all matrices
	
for n in 1:Niter_ODE-1  # Time step	

apply_boundary_conditions(n)		
		
elastic_forces(n)	
damping_forces(n)	
external_forces(n)			

"""
# Verlet-Strönberg			
@. a_x[:, :,:, n+1] = 2*a_x[:, :,:, n] - a_x[:, :,:, n-1] + a_F[:, :,:, n] * Δt^2 / a_m[:,:,:, n] 				
		
@. a_v[:,:,:, n+1] = (a_x[:, :,:, n+1] - a_x[:, :,:, n-1]) / (2*Δt)
"""
		

# Verlet with velocity. This formulation adds a bit of numerical damping

@. a_x[:, :,:, n+1] = a_x[:, :,:, n] + a_v[:, :,:, n]* Δt + a_F[:, :,:, n] * Δt^2 / a_m[:,:, :, n] 				
		
@. a_v[:,:,:, n+1] = (a_x[:, :,:, n+1] - a_x[:, :,:, n-1]) / (2*Δt)		
	
	
end	# next step
	
draw_animation()
#draw_animated_heatmap()
	
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
#TableOfContents(aside=true)

# ╔═╡ Cell order:
# ╠═e3a19f76-87b8-4bb8-95fc-750a48321cfb
# ╠═9b279318-f7d7-4ac2-b6ba-9edb83c1555f
# ╠═813a6e2b-4bd0-4b7a-9545-107638e8f26e
# ╠═1c4f4c21-8e67-4df7-bec1-1a08c9f67464
# ╠═5eb95c3c-488a-4d2f-a4df-1cd22bc189ef
# ╠═d3578df6-5592-4806-9114-91c0e0d1bf35
# ╠═3c33576c-0320-4a1a-b375-7ccbcb177ce1
# ╠═46662437-8f37-4226-b371-96a4938c44b8
# ╟─66ba9dc4-1d50-410a-acdd-850c8f27fd3d
# ╟─454494b5-aca5-43d9-8f48-d5ce14fbd5a9
# ╟─6104ccf7-dfce-4b0b-a869-aa2b71deccde
# ╠═10ececaa-5ac8-4870-bcbb-210ffec09515
# ╟─402abadb-d500-4801-8005-11d036f8f351
# ╠═5c5e95fb-4ee2-4f37-9aaf-9ceaa05def57
# ╠═d7469640-9b09-4262-b738-29810bd19305
# ╠═a8011889-d844-4c98-bd3f-014e7eb58254
# ╟─01bfb4bc-d498-4dd4-b2a8-f6a5e59f8ae4
# ╠═e084941c-447a-41bd-bf06-59dea45af028
# ╟─96740b63-f5d6-4721-8722-37baae48f47b
# ╟─0b3749f4-846f-4c66-858b-c560e0e27bca
# ╠═30d5a924-7bcd-4eee-91fe-7b10004a4139
# ╠═a755dbab-6ac9-4a9e-a397-c47efce4d2f7
# ╠═6960420d-bc50-4be3-9a26-2f43f14b903d
# ╠═cea5e286-4bc1-457f-b300-fdff62047cc4
# ╟─bef1cd36-be8d-4f36-b5b9-e4bc034f0ac1
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─965946ba-8217-4202-8870-73d89c0c7340
# ╠═6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
# ╟─b23125f6-7118-4ce9-a10f-9c3d3061f8ce
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╟─7ae886d4-990a-4b14-89d5-5708f805ef93
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═87be1f09-c729-4b1a-b05c-48c79039390d
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
# ╟─c72f9b42-94c7-4377-85cd-5afebbe1d271
# ╟─fc7e00a0-9205-11eb-039c-23469b96de19
# ╟─13b32a20-9206-11eb-3af7-0feea278594c
