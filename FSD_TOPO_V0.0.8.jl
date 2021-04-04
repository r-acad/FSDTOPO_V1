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

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### INITIALIZE MODEL  v 0 0 8
- Back to original formulation in 88 lines after attemt to reorder elements in v 0 0 6

"""

# ╔═╡ d007f530-9255-11eb-2329-9502dc270b0d
 #newt = FSDTOPO(2);

# ╔═╡ 0316391f-3ba8-46da-9c5f-d11e44aba9dc
#sK = vcat([KE[:].*(t[:][l])  for l in 1:nelx*nely]...)    #  ',64*nelx*nely

# ╔═╡ ff86dec7-f6a3-416b-ad6e-8affc3800bd6
#heatmap(reverse([reshape(sK, 8,8) zeros(8,1) KE], dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ c4c9ace0-9237-11eb-1f26-334caba1248d

begin

function FSDTOPO( niter)	
	
sigma_all	= 6
max_all_t = 5
full_penalty_iter = 15
max_penalty = 5
thick_ini = 1.0		
min_thick = 0.00001
		

th[1:nely,1:nelx] .= thick_ini	# Initialize thickness distribution in domain
		
t = view(th, 1:nely,1:nelx) # take a view of the canvas representing the thickness domain			
	

			
sK = vcat([KE[:].*(t[:][l])  for l in 1:nelx*nely]...)    #  ',64*nelx*nely
		
#sK = ones(Int64,64)
	
K = sparse(iK,jK,sK)
			
#K = (K+K')./2
		
U(freedofs) = K(freedofs,freedofs)\F(freedofs)			
			
			
	ESE = INTERNAL_LOADS()		
				
	t .*= ESE / sigma_all # Obtain new thickness by FSD algorithm
	t = [min(nt, max_all_t) for nt in t]

			
	# Filter loop					

#t = [sum(th[i.+CartesianIndices((-1:1, -1:1))].*( [1 2 1 ; 2 4 2 ;  1 2 1] ./16)) for i in CartesianIndices(t)]						
			
penalty = min(1 + iter / full_penalty_iter, max_penalty)		
			
t = [max(max_all_t*(nt^penalty), min_thick) for nt in t]				
		
			
#t = [max((max_all_t*(min(nt,max_all_t)/max_all_t)^penalty), min_thick) for nt in t]
		
end		
		
return t # retuns a view of the canvas containing only the thickness domain
end # end function

end


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
	

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin

scale = 1
nelx = 60*scale ; nely = 20*scale  #mesh size

nDoF = 	2*(nely+1)*(nelx+1)  # Total number of degrees of freedom
	
F = zeros(Float64, nDoF)	# Initialize external forces vector
F[2] = -1.0	   # Applied external force
		
U = zeros(Float64, nDoF)	# Initialize global displacements
	
fixeddofs = [(1:2:2*(nely+1))..., nDoF ]
	
alldofs   = Vector(1:nDoF)
freedofs  = setdiff(alldofs,fixeddofs)	
	
"""

edofVec = 2*nodenrs[1:end-1,1:end-1].+1
	
edofMat = repeat(edofVec,1,8) + repeat([0 1 2*nely.+[2 3 0 1] -2 -1],nelx*nely,1);

iK = convert(Array{Int64}, kron(edofMat,ones(8,1))'[:])
jK = convert(Array{Int64}, kron(edofMat,ones(1,8))'[:])	
	
nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx)	
"""	

nodenrs = reshape(1:(1+nelx)*(1+nely),1+nely,1+nelx)
	
edofVec = ((nodenrs[1:end-1,1:end-1].*2).+1)[:]	

edofMat = repeat(edofVec,1,8) + repeat([0 1 2*nely.+[2 3 0 1] -2 -1],nelx*nely)

iK = reshape(kron(edofMat,ones(Int64,8,1))',64*nelx*nely)
jK = reshape(kron(edofMat,ones(Int64,1,8))',64*nelx*nely)
	
iK = kron(edofMat,ones(Int64,8,1))'[:]
jK = kron(edofMat,ones(Int64,1,8))'[:]
	
#iK = convert(Array{Int64}, kron(edofMat,ones(8,1))'[:])
#jK = convert(Array{Int64}, kron(edofMat,ones(1,8))'[:])	
	
KE = KE_CQUAD4()

end;

# ╔═╡ 6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
begin
	sigma_all	= 6
	max_all_t = 5
	full_penalty_iter = 15
	max_penalty = 5
	thick_ini = 1.0		
	min_thick = 0.00001
			
	th = OffsetArray( zeros(Float64,1:nely+2,1:nelx+2), 0:nely+1,0:nelx+1) # Initialize thickness canvas with ghost cells as padding
	th[1:nely,1:nelx] .= thick_ini	# Initialize thickness distribution in domain
			
	t = view(th, 1:nely,1:nelx) # take a view of the canvas representing the thickness domain	

end;

# ╔═╡ d1f6b4c5-85fa-466d-913c-534d03dd504e
nodenrs;

# ╔═╡ b5d2f972-1e8b-496c-ad3e-a9f3a0b8a6af
edofVec;

# ╔═╡ 5a570368-95a9-4427-b378-7e59f02ae20a
edofMat;

# ╔═╡ c407d096-aa8c-496d-9444-202b21d20a01
iK

# ╔═╡ 20e299bc-9f3a-4f2d-a4ab-9355425afb12
jK

# ╔═╡ 64e213d1-672e-4bc0-9fd6-3b1f933566a6
sK = reshape(KE[:]*t[:]', 64*nelx*nely)

# ╔═╡ 11d356ed-ecd8-4521-a961-70c4768d64c9
KE

# ╔═╡ f2b44b9a-596a-4025-9029-1306ec3d4f1e
#sK = ones(Int64,64)
	
K = Symmetric(sparse(iK,jK,sK))

# ╔═╡ eb797772-c2fe-45e5-a4e5-eeb6c478c15b
K

# ╔═╡ 87da1a10-3010-498d-8568-76cba4be38e5
heatmap(reverse(Matrix(K) , dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true))

# ╔═╡ 12cc1b97-e1eb-4a81-b287-2136aa155a34
#K = (K+K')./2
		
U[freedofs] = K[freedofs,freedofs]\F[freedofs]

# ╔═╡ a8c96d92-aee1-4a91-baf0-2a585c2fa51f
begin

function NODAL_DISPLACEMENTS(th)
KE = KE_CQUAD4()
		
sK = reshape(KE[:],64*nelx*nely,1)
		
K = sparse(iK,jK,sK); K = (K+K')/2
		
U(freedofs) = K(freedofs,freedofs)\F(freedofs)
		
end # function
	
end

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
	
end

# ╔═╡ d01c2419-3eb1-410f-b9cf-809146707e47
SE1 = INTERNAL_LOADS()

# ╔═╡ 9d8f3527-3bad-4cdd-abd5-674f3d7a08ec
heatmap(reverse(SE1, dims=1), aspect_ratio = 1, c=cgrad(:jet1, 10, categorical = true)) 

# ╔═╡ c58a7360-920c-11eb-2a15-bda7ed075812
#heatmap(reverse(SU_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ Cell order:
# ╠═13b32a20-9206-11eb-3af7-0feea278594c
# ╟─fc7e00a0-9205-11eb-039c-23469b96de19
# ╠═d88f8062-920f-11eb-3f57-63a28f681c3a
# ╟─6ec04b8d-e5d9-4f62-b5c5-349a5f71e3e4
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╠═d1f6b4c5-85fa-466d-913c-534d03dd504e
# ╠═b5d2f972-1e8b-496c-ad3e-a9f3a0b8a6af
# ╠═5a570368-95a9-4427-b378-7e59f02ae20a
# ╠═c407d096-aa8c-496d-9444-202b21d20a01
# ╠═20e299bc-9f3a-4f2d-a4ab-9355425afb12
# ╠═d007f530-9255-11eb-2329-9502dc270b0d
# ╠═0316391f-3ba8-46da-9c5f-d11e44aba9dc
# ╠═64e213d1-672e-4bc0-9fd6-3b1f933566a6
# ╠═11d356ed-ecd8-4521-a961-70c4768d64c9
# ╠═f2b44b9a-596a-4025-9029-1306ec3d4f1e
# ╠═12cc1b97-e1eb-4a81-b287-2136aa155a34
# ╠═ff86dec7-f6a3-416b-ad6e-8affc3800bd6
# ╠═eb797772-c2fe-45e5-a4e5-eeb6c478c15b
# ╠═87da1a10-3010-498d-8568-76cba4be38e5
# ╠═d01c2419-3eb1-410f-b9cf-809146707e47
# ╠═9d8f3527-3bad-4cdd-abd5-674f3d7a08ec
# ╠═c4c9ace0-9237-11eb-1f26-334caba1248d
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╠═6bd11d90-93c1-11eb-1368-c9484c1302ee
# ╠═a8c96d92-aee1-4a91-baf0-2a585c2fa51f
# ╠═944f5b10-9236-11eb-05c2-45824bc3b532
# ╟─2c768930-9210-11eb-26f8-0dc24f22afaf
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╟─c1711000-920b-11eb-14ba-eb5ce08f3941
# ╟─c58a7360-920c-11eb-2a15-bda7ed075812
