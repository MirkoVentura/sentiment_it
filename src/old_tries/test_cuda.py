import torch
dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
t1 = torch.randn(1,2)
t2 = torch.randn(1,2).to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]])
print(t2)  # tensor([[ 0.5117, -3.6247]], device='cuda:0')
t1.to(dev) 
print(t1)  # tensor([[-0.2678,  1.9252]]) 
print(t1.is_cuda) # False
t1=t1.to(dev)
print(t1)  # tensor([[-0.2678,  1.9252]], device='cuda:0') 
print(t1.is_cuda) # True