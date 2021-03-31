### A Pluto.jl notebook ###
# v0.12.21

using Markdown
using InteractiveUtils

# ╔═╡ cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
function KE_CQUAD4()
# Element stiffness matrix reverse-engineered from NASTRAN with E = 1, t = 1, nu=.03
	
	
A = -5.766129E-01   
B = -6.330645E-01  
C =  2.096774E-01   	
D = 3.931452E-01	
G = -3.024194E-02	


KE = [ 	[1 D A G     B -D C G];
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

# ╔═╡ Cell order:
# ╠═cd707ee0-91fc-11eb-134c-2fdd7aa2a50c
# ╠═f2946650-91fc-11eb-3f39-65d37b9dfe24
