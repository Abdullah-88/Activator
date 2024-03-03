import torch
from torch import nn





   



class ActivatorGatingUnit(nn.Module):
    def __init__(self,dim, hidden_dim):
        super().__init__()
        self.proj_1 =  nn.Linear(dim, hidden_dim)
        self.proj_2 =  nn.Linear(dim, hidden_dim)
        self.proj_3 = nn.Linear(hidden_dim , dim)     
        self.gelu = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
             	   
    def forward(self, x):
        u, v = x, x 
        u = self.proj_1(u)
        u = self.gelu(u)
        u = self.norm(u)
        
        v = self.proj_2(v)
        v = self.norm(v)
       
        g = u * v
        
        out = self.proj_3(g)
        return out



class ActivatorBlock(nn.Module):
    def __init__(self, d_model, d_ffn,dropout):
        super().__init__()
       
        self.norm = nn.LayerNorm(d_model)       
        self.actgu = ActivatorGatingUnit(d_model, d_ffn)
        #self.ffn = FeedForward(d_model,d_ffn,dropout)
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.actgu(x)           
        x = x + residual      
        
        out = x
        return out



class ACTIVATOR(nn.Module):
    def __init__(self, d_model, d_ffn, num_layers,dropout):
        super().__init__()
        
        self.model = nn.Sequential(
            *[ActivatorBlock(d_model,d_ffn,dropout) for _ in range(num_layers)]
        )

    def forward(self, x):
       
        return self.model(x)








