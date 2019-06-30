import  torch  as t 

a = t.arange(0, 4).view(4, 1)

print(a.squeeze().size())
