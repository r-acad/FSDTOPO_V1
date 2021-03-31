### A Pluto.jl notebook ###
# v0.12.21

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

			])

	using PlutoUI
	using Colors, ColorSchemes, Images
	using Plots
	using LaTeXStrings
	
	using Statistics, LinearAlgebra  # standard libraries
end

# ╔═╡ 13b32a20-9206-11eb-3af7-0feea278594c
TableOfContents(aside=true)

# ╔═╡ d88f8062-920f-11eb-3f57-63a28f681c3a
md"""
### INITIALIZE MODEL
"""

# ╔═╡ f60365a0-920d-11eb-336a-bf5953215934
begin

nelx = 9 ; nely = 3  #mesh size

F = zeros(2*(nelx+1)*(nely+1))	# Initialize external forces vector
F[2] = -1.0	   # Applied external force
		
U = zeros(2*(nelx+1)*(nely+1))	# Initialize global displacements

t = 1.0 .* ones(1:nelx,1:nely) # Initialize thickness distribution
			
fixeddofs = [Vector(1:2:2*(nely+1)) ; [2*(nelx+1)*(nely+1)] ]
alldofs = Vector(1:2*(nelx+1)*(nely+1))
freedofs = setdiff(alldofs,fixeddofs)			
end;

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
	

# ╔═╡ 2c768930-9210-11eb-26f8-0dc24f22afaf
begin

K = zeros(2*(nely+1)*(nelx+1), 2*(nely+1)*(nelx+1))	# Initialize global stiffness matrix

KE = KE_CQUAD4()
	
for x = 1:nelx, y = 1:nely	
		
	# Node numbers, starting at top left corner and growing in columns going down as per 99 lines of code
	n1 = (nely+1)*(x-1) + y ; n2 = n1 + 1
	n3 = (nely+1)*x + y     ; n4 = n3 + 1
		
	# Numbering of node DoFs in global stiffness matrix; m_nnode_dof
	m12 = 2*n1   ; m11 = m12 -1
	m22 = 2*n2   ; m21 = m22 -1
	m32 = 2*n3   ; m31 = m32 -1	
	m42 = 2*n4   ; m41 = m42 -1

	edof = [m11, m12, m21, m22, m31, m32, m41, m42 ]
			
	K[edof, edof] += t[x,y] * KE			

		
end # for

	U[freedofs] = K[freedofs,freedofs] \ F[freedofs]
	
	
	
end

# ╔═╡ 4aba92de-9212-11eb-2089-073a71342bb0
heatmap(reverse(K, dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ c652e5c0-9207-11eb-3310-ddef16cdb1ac
heatmap(reverse(KE_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ c1711000-920b-11eb-14ba-eb5ce08f3941
function SU_CQUAD4()
# Matrix relating cartesian stress components (sxx, syy, sxy) with nodal displacement in CQUAD4 element, reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
		
A = -1.209677E+00  
B = -3.629032E-01 
C = -4.233871E-01   	

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

# ╔═╡ d36a8b50-920c-11eb-2667-1735bcf374de
 SU_CQUAD4()

# ╔═╡ c58a7360-920c-11eb-2a15-bda7ed075812
heatmap(reverse(SU_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ Cell order:
# ╟─13b32a20-9206-11eb-3af7-0feea278594c
# ╟─fc7e00a0-9205-11eb-039c-23469b96de19
# ╟─d88f8062-920f-11eb-3f57-63a28f681c3a
# ╠═f60365a0-920d-11eb-336a-bf5953215934
# ╠═2c768930-9210-11eb-26f8-0dc24f22afaf
# ╠═4aba92de-9212-11eb-2089-073a71342bb0
# ╟─d108d820-920d-11eb-2eee-bb6470fb4a56
# ╟─cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╟─c652e5c0-9207-11eb-3310-ddef16cdb1ac
# ╟─c1711000-920b-11eb-14ba-eb5ce08f3941
# ╟─d36a8b50-920c-11eb-2667-1735bcf374de
# ╟─c58a7360-920c-11eb-2a15-bda7ed075812
