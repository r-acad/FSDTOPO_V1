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
PlutoUI.TableOfContents(aside=true)

# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
function KE_CQUAD4()
# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
	
	
A = -5.766129E-01   
B = -6.330645E-01  
C =  2.096774E-01   	
D = 3.931452E-01	
G = 3.024194E-02	


KE = [ 	[1 D A -G     B -D C G];
		[D 1 G C   -D B -G A];
		[A G 1 -D   C -G B D];
		[-G C -D 1   G A D B];
		[B -D C G   1 D A -G];
		[-D B -G A   D 1 G C];
		[C -G B D   A G 1 -D];
		[G A D B   -G C -D 1]		
		]'	


end	
	

# ╔═╡ f2946650-91fc-11eb-3f39-65d37b9dfe24
KE_CQUAD4()

# ╔═╡ c652e5c0-9207-11eb-3310-ddef16cdb1ac
heatmap(reverse(KE_CQUAD4(), dims=1), aspect_ratio = 1, c=cgrad(:roma, 10, categorical = true))

# ╔═╡ Cell order:
# ╠═13b32a20-9206-11eb-3af7-0feea278594c
# ╠═fc7e00a0-9205-11eb-039c-23469b96de19
# ╠═cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╠═f2946650-91fc-11eb-3f39-65d37b9dfe24
# ╠═c652e5c0-9207-11eb-3310-ddef16cdb1ac
