### A Pluto.jl notebook ###
# v0.14.0

using Markdown
using InteractiveUtils

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
			])

	using PlutoUI
	using Colors, ColorSchemes, Images
	using Plots, OffsetArrays, SparseArrays
	using LaTeXStrings
	
	using Statistics, LinearAlgebra  # standard libraries
end

# ╔═╡ 13b32a20-9206-11eb-3af7-0feea278594c
TableOfContents(aside=true)

# ╔═╡ d3a4076c-04a1-4f2c-8c0c-d2a0a86c51a4
md""" ### VERSION STATUS

- It runs but it seems numbering convention is wrong as results don't make sense, strain field is non physical (4/4/21)  v0.0.6
- v0.07 Attempt to renumber the matrices according to Nastran ordering with: 

    3        4

    1        2

"""

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### INITIALIZE MODEL
"""

# ╔═╡ 6b8a46b1-50b2-4103-831f-f002afb65b9c
begin
sigma_all	= 6
max_all_t = 5
full_penalty_iter = 15
max_penalty = 5
thick_ini = 1.0		
min_thick = 0.00001
	
end

# ╔═╡ 9228af13-9913-410a-abb3-2a57d9656616


# ╔═╡ d007f530-9255-11eb-2329-9502dc270b0d
 #newt = FSDTOPO(2);

# ╔═╡ 87da1a10-3010-498d-8568-76cba4be38e5
#heatmap(reverse([Matrix(K) zeros(8,1) KE zeros(8,1) 2. * KE_CQUAD4_88()], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
heatmap(reverse(newt, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

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

KE = [ 	[ 1  D  A -G    C  G   B -D ];
		[ D  1  G  C   -G  A  -D  B ];
		[ A  G  1 -D    B  D   C -G ];
		[-G  C -D  1    D  B   G  A ];
		[ C -G  B  D    1 -D   A  G ];
		[ G  A  D  B   -D  1  -G  C ];		
		[ B -D  C  G    A -G   1  D ];
		[-D  B -G  A    G  C   D  1 ]
		]'	
end	
	

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin

scale = 1
nelx = 60*scale ; nely = 20*scale  #mesh size

nDoF = 	2*(nely+1)*(nelx+1)  # Total number of degrees of freedom
	
F = zeros(Float64, nDoF)	# Initialize external forces vector
#F[2*nely*(nelx+1)+2] = -1.0	   # Applied external force
	
F[2] = -1.0	   # Applied external force	
		
U = zeros(nDoF)	# Initialize global displacements
	
th = OffsetArray( zeros(Float64,1:nely+2,1:nelx+2), 0:nely+1,0:nelx+1) # Initialize thickness canvas with ghost cells as padding

th[1:nely,1:nelx] .= thick_ini	# Initialize thickness distribution in domain		
t = view(th, 1:nely,1:nelx) # take a view of the canvas representing the thickness 	
	
#fixeddofs = [Vector(1:2:2*(nely+1)) ; [nDoF] ]  # 88 lines
	
fixeddofs = [[ 1 + (y-1)*2*(nelx+1) for y in 1:(nely+1)] ; (nelx+1)*2] 	

	
alldofs   = Vector(1:nDoF)
freedofs  = setdiff(alldofs,fixeddofs)	

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx)

edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:]  # 88 lines
edofMat = repeat(edofVec,1,8) + repeat([0 1 2*nely.+[2 3 0 1] -2 -1],nelx*nely) 
	

	

n1arr = [ (x + (nelx+1)*(y-1)) for x in 1:nelx, y in 1:nely][:] # array with ids of first nodes of quads

darr = hcat([[2*n1-1, 2*n1, 2*n1+1, 2*n1+2, 
		(2*n1+2 + 2*(nelx+1) -3), (2*n1+2 + 2*(nelx+1) -2), 
		(2*n1+2 + 2*(nelx+1) -1), (2*n1+2 + 2*(nelx+1) -0)]  for n1 in n1arr ]...)[:]
	
	
iK = repeat(darr,8)
jK = kron(darr, ones(Int64, 8))

	
iK = convert(Array{Int64}, kron(darr,ones(8,1))'[:])
jK = convert(Array{Int64}, kron(darr,ones(1,8))'[:])		
	
KE = KE_CQUAD4()

end;

# ╔═╡ aae944e1-06b6-44f0-aaaf-a176149cf6ac
fixeddofs

# ╔═╡ bf741625-e24b-4a5c-ad9b-ce95c0f83e93
freedofs

# ╔═╡ 4520f754-3e71-4762-832e-4112e5c36d6f
n1arr

# ╔═╡ fb70c9ed-e163-4e7a-bdb8-38b03a69e75b
darr

# ╔═╡ 59bbe39e-43d0-477d-94ab-54b34558b903
iK, jK

# ╔═╡ c269f0fb-4949-4a5e-8f67-62ccdbf035fd
sK = hcat([KE[:].*t[:][l]  for l in 1:nelx*nely  ]...)[:] #  ',64*nelx*nely

# ╔═╡ 8342ee61-62af-4fb7-ab91-013b5a51fefd
K = Symmetric(sparse(iK,jK,sK))

# ╔═╡ 819e3038-a2c8-425a-b5fb-fd896967979c
heatmap(reverse(Matrix(K), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ 86b31c22-2ce8-4c37-80a1-df40f287cdbd
U[freedofs] = K[freedofs,freedofs]\F[freedofs]

# ╔═╡ ded9ded3-d1fb-42be-97f5-a3b0049f871d
det(K[freedofs,freedofs])

# ╔═╡ a8c96d92-aee1-4a91-baf0-2a585c2fa51f
begin

function NODAL_DISPLACEMENTS()
KE = KE_CQUAD4()

sK = hcat([KE[:].*t[:][l]  for l in 1:nelx*nely  ]...)[:] #  ',64*nelx*nely		
	
K = Symmetric(sparse(iK,jK,sK))
			
		
U[freedofs] = K[freedofs,freedofs]\F[freedofs]	

return K		
		
end # function
	
end

# ╔═╡ f37be92f-03ef-49a5-a968-c760fb7ae657
NODAL_DISPLACEMENTS();

# ╔═╡ 944f5b10-9236-11eb-05c2-45824bc3b532
begin

function NODAL_DISPLACEMENTS_99(th)
K = zeros(Float64,nDoF, nDoF)	# Initialize global stiffness matrix
KE = KE_CQUAD4()
	
for y = 1:nely, x = 1:nelx			
# Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code	
	n1 = (nely+1)*(x-1)+y ;	n2 = (nely+1)* x +y		
	edof = [2*n1-1; 2*n1; 2*n2-1; 2*n2; 2*n2+1;2*n2+2;2*n1+1; 2*n1+2]				
	K[edof, edof] += th[y,x] * KE					
end # for

	# Solve linear system and get global displacements vector U
	U[freedofs] = K[freedofs,freedofs] \ F[freedofs]

end # function
	
end

# ╔═╡ bbdf32f0-933b-41c6-9e08-f6762bc440ed
function KE_CQUAD4_orig()
# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03

# Node orientation;
#  3 4
#  1 2
	
	
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

# ╔═╡ 3ef71d2c-4511-4d2c-8527-b7c07529c715
begin
	
function KE_CQUAD4_88()   # from 88 lines
	nu =0.3
		
	A11 = [12  3 -6 -3;  3 12  3  0; -6  3 12 -3; -3  0 -3 12];
	A12 = [-6 -3  0  3; -3 -6 -3 -6;  0 -3 -6  3;  3 -6  3 -6];
	B11 = [-4  3 -2  9;  3 -4 -9  4; -2 -9 -4 -3;  9  4 -3 -4];
	B12 = [ 2 -3  4 -9; -3  2  9 -2;  4  9  2  3; -9 -2  3  2];
	KE = 1/(1-nu^2)/24*([A11 A12;A12' A11]+nu*[B11 B12;B12' B11]);
		
end		
	
end

# ╔═╡ c652e5c0-9207-11eb-3310-ddef16cdb1ac
heatmap(reverse(KE_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

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

function INTERNAL_LOADS()
#######################	
	
S = zeros(Float64,1:nely,1:nelx)  # Initialize matrix containing field results (typically a stress component or function)
SUe = SU_CQUAD4() # Matrix that relates element stresses to nodal displacements
		
for y = 1:nely, x = 1:nelx		
	# Node numbers, starting at top left corner and growing in columns going down as per in 99 lines of code		
	n1 = (nely+1)*(x-1)+y;	n2 = (nely+1)* x +y	
	Ue = U[[2*n1-1;2*n1; 2*n2-1;2*n2; 2*n2+1;2*n2+2; 2*n1+1;2*n1+2],1]
		
	Te = (SUe * Ue) .* nelx  # Element stress vector in x, y coordinates. Scaled by mesh size
	sxx = Te[1] ; syy = Te[2] ; sxy = Te[3]
	
	# Principal stresses
	s1 = 0.5 * (sxx + syy + ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
	s2 = 0.5 * (sxx + syy - ((sxx - syy) ^ 2 + 4 * sxy ^ 2) ^ 0.5)
	ese = (s1 ^ 2 + s2 ^ 2 + 2 * 0.3 * s1 * s2) ^ 0.5		# elastic strain energy
		
	#S[y, x] = Ue'*KE*Ue    # Compliance from 99 lines	
	S[y, x] = ese
		
end # for	
	
	return S	
end # function	
	
end;

# ╔═╡ e00e6e1b-f775-4435-be78-25038f7e4fe6
ESE = INTERNAL_LOADS()		;	

# ╔═╡ ff86dec7-f6a3-416b-ad6e-8affc3800bd6
heatmap(reverse(ESE, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ c4c9ace0-9237-11eb-1f26-334caba1248d
begin

function FSDTOPO( niter)	
		
for iter in 1:niter

NODAL_DISPLACEMENTS(t)

	ESE = INTERNAL_LOADS()		
				
	t .*= ESE / sigma_all # Obtain new thickness by FSD algorithm
	t = [min(nt, max_all_t) for nt in t]

			
	# Filter loop					
"""
t = [sum(th[i.+CartesianIndices((-1:1, -1:1))]
				.*( [1 2 1 ;
				   2 4 2 ;
				   1 2 1] ./16)
		) for i in CartesianIndices(t)]						
"""
			
penalty = min(1 + iter / full_penalty_iter, max_penalty)		
			
t = [max(max_all_t*(nt^penalty), min_thick) for nt in t]				
		
			
#t = [max((max_all_t*(min(nt,max_all_t)/max_all_t)^penalty), min_thick) for nt in t]
		
end		
		
return t # retuns a view of the canvas containing only the thickness domain
end # end function
	
end

# ╔═╡ 77182619-207e-43af-bb53-60cbc9ec605f
begin  # element DoFs matrix from 99 lines

n1(x,y) = (nely+1)*(x-1)+y
n2(x,y) = (nely+1)* x +y	
	
edof(x,y) = [2*n1(x,y)-1, 2*n1(x,y), 2*n2(x,y)-1, 2*n2(x,y), 2*n2(x,y)+1, 2*n2(x,y)+2, 2*n1(x,y)+1, 2*n1(x,y)+2]
	
n1s = [ edof(x,y) for y in 1:nely, x in 1:nelx ]

	
end	;

# ╔═╡ c58a7360-920c-11eb-2a15-bda7ed075812
#heatmap(reverse(SU_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ Cell order:
# ╟─13b32a20-9206-11eb-3af7-0feea278594c
# ╟─fc7e00a0-9205-11eb-039c-23469b96de19
# ╟─d3a4076c-04a1-4f2c-8c0c-d2a0a86c51a4
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─6b8a46b1-50b2-4103-831f-f002afb65b9c
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╠═aae944e1-06b6-44f0-aaaf-a176149cf6ac
# ╠═bf741625-e24b-4a5c-ad9b-ce95c0f83e93
# ╠═4520f754-3e71-4762-832e-4112e5c36d6f
# ╠═fb70c9ed-e163-4e7a-bdb8-38b03a69e75b
# ╠═59bbe39e-43d0-477d-94ab-54b34558b903
# ╠═c269f0fb-4949-4a5e-8f67-62ccdbf035fd
# ╠═8342ee61-62af-4fb7-ab91-013b5a51fefd
# ╠═9228af13-9913-410a-abb3-2a57d9656616
# ╠═86b31c22-2ce8-4c37-80a1-df40f287cdbd
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═f37be92f-03ef-49a5-a968-c760fb7ae657
# ╠═819e3038-a2c8-425a-b5fb-fd896967979c
# ╠═ded9ded3-d1fb-42be-97f5-a3b0049f871d
# ╠═e00e6e1b-f775-4435-be78-25038f7e4fe6
# ╠═ff86dec7-f6a3-416b-ad6e-8affc3800bd6
# ╠═87da1a10-3010-498d-8568-76cba4be38e5
# ╟─c4c9ace0-9237-11eb-1f26-334caba1248d
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╟─6bd11d90-93c1-11eb-1368-c9484c1302ee
# ╠═a8c96d92-aee1-4a91-baf0-2a585c2fa51f
# ╟─944f5b10-9236-11eb-05c2-45824bc3b532
# ╟─2c768930-9210-11eb-26f8-0dc24f22afaf
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╠═cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─bbdf32f0-933b-41c6-9e08-f6762bc440ed
# ╟─3ef71d2c-4511-4d2c-8527-b7c07529c715
# ╠═c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╟─c1711000-920b-11eb-14ba-eb5ce08f3941
# ╟─77182619-207e-43af-bb53-60cbc9ec605f
# ╟─c58a7360-920c-11eb-2a15-bda7ed075812
