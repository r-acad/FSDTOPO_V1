for i in 1:20


println("Iter " * string(i))

A = rand(1000, 1000)
b = rand(1000)

A\b

end