import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PhyCell_Cell(nn.Module):
    def __init__(self, input_dim, F_hidden_dim, kernel_size, bias=1):
        super(PhyCell_Cell, self).__init__()
        self.input_dim  = input_dim
        self.F_hidden_dim = F_hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        
        self.F = nn.Sequential()
        self.F.add_module('bn1',nn.GroupNorm( 4 ,input_dim))          
        self.F.add_module('conv1', nn.Conv2d(in_channels=input_dim, out_channels=F_hidden_dim, kernel_size=self.kernel_size, stride=(1,1), padding=self.padding))  
        #self.F.add_module('f_act1', nn.LeakyReLU(negative_slope=0.1))        
        self.F.add_module('conv2', nn.Conv2d(in_channels=F_hidden_dim, out_channels=input_dim, kernel_size=(1,1), stride=(1,1), padding=(0,0)))

        self.convgate = nn.Conv2d(in_channels=self.input_dim + self.input_dim,
                              out_channels= self.input_dim,
                              kernel_size=(3,3),
                              padding=(1,1), bias=self.bias)

    def forward(self, x, hidden): # x [batch_size, hidden_dim, height, width]      
        hidden_tilde = hidden + self.F(hidden)        # prediction
        
        combined = torch.cat([x, hidden_tilde], dim=1)  # concatenate along channel axis
        combined_conv = self.convgate(combined)
        K = torch.sigmoid(combined_conv)
        
        next_hidden = hidden_tilde + K * (x-hidden_tilde)   # correction , Haddamard product     
        return next_hidden

   
class PhyCell(nn.Module):
    def __init__(self, input_shape, input_dim, F_hidden_dims, n_layers, kernel_size, device):
        super(PhyCell, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.F_hidden_dims = F_hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H = []  
        self.device = device
             
        cell_list = []
        for i in range(0, self.n_layers):
        #    cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]

            cell_list.append(PhyCell_Cell(input_dim=input_dim,
                                          F_hidden_dim=self.F_hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j] = cell(input_, self.H[j])
            else:
                self.H[j] = cell(self.H[j-1],self.H[j])
        
        return self.H , self.H 
    
    def initHidden(self,batch_size):
        self.H = [] 
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size, self.input_dim, self.input_shape[0], self.input_shape[1]).to(self.device) )

    def setHidden(self, H):
        self.H = H
  
   
class ConvLSTM_Cell(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dim, kernel_size, bias=1):              
        """
        input_shape: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """
        super(ConvLSTM_Cell, self).__init__()
        
        self.height, self.width = input_shape
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding     = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias        = bias
        
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding, bias=self.bias)
                 
    # we implement LSTM that process only one timestep 
    def forward(self,x, hidden): # x [batch, hidden_dim, width, height]          
        h_cur, c_cur = hidden
        
        combined = torch.cat([x, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1) 
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


    
    
class ConvLSTM(nn.Module):
    def __init__(self, input_shape, input_dim, hidden_dims, n_layers, kernel_size,device):
        super(ConvLSTM, self).__init__()
        self.input_shape = input_shape
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.H, self.C = [],[]   
        self.device = device
        
        cell_list = []
        for i in range(0, self.n_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dims[i-1]
            print('layer ',i,'input dim ', cur_input_dim, ' hidden dim ', self.hidden_dims[i])
            cell_list.append(ConvLSTM_Cell(input_shape=self.input_shape,
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dims[i],
                                          kernel_size=self.kernel_size))                                     
        self.cell_list = nn.ModuleList(cell_list)
        
       
    def forward(self, input_, first_timestep=False): # input_ [batch_size, 1, channels, width, height]    
        batch_size = input_.data.size()[0]
        if (first_timestep):   
            self.initHidden(batch_size) # init Hidden at each forward start
              
        for j,cell in enumerate(self.cell_list):
            if j==0: # bottom layer
                self.H[j], self.C[j] = cell(input_, (self.H[j],self.C[j]))
            else:
                self.H[j], self.C[j] = cell(self.H[j-1],(self.H[j],self.C[j]))
        
        return (self.H,self.C) , self.H   # (hidden, output)
    
    def initHidden(self,batch_size):
        self.H, self.C = [],[]  
        for i in range(self.n_layers):
            self.H.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
            self.C.append( torch.zeros(batch_size,self.hidden_dims[i], self.input_shape[0], self.input_shape[1]).to(self.device) )
    
    def setHidden(self, hidden):
        H,C = hidden
        self.H, self.C = H,C
 

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(in_channels=nin, out_channels=nout, kernel_size=(3,3), stride=stride, padding=1),
                nn.GroupNorm(4,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

        
class dcgan_upconv(nn.Module):
    def __init__(self, nin, nout, stride):
        super(dcgan_upconv, self).__init__()
        if (stride ==2):
            output_padding = 1
        else:
            output_padding = 0
        self.main = nn.Sequential(
                nn.ConvTranspose2d(in_channels=nin,out_channels=nout,kernel_size=(3,3), stride=stride,padding=1,output_padding=output_padding),
                nn.GroupNorm(4,nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
     

class image_encoder(nn.Module):
    def __init__(self, nc=1):
        super(image_encoder, self).__init__()
        nf = 16
        # input is (nc) x 64 x 64
        self.c1 = dcgan_conv(nc, int(nf/2), stride=1) # (nf) x 64 x 64
        self.c2 = dcgan_conv(int(nf/2), nf, stride=1) # (nf) x 64 x 64
        self.c3 = dcgan_conv(nf, nf*2, stride=2) # (2*nf) x 32 x 32
        self.c4 = dcgan_conv(nf*2, nf*2, stride=1) # (2*nf) x 32 x 32
        self.c5 = dcgan_conv(nf*2, nf*4, stride=2) # (4*nf) x 16 x 16
        self.c6 = dcgan_conv(nf*4, nf*4, stride=1) # (4*nf) x 16 x 16          

    def forward(self, input):
        h1 = self.c1(input)  # (nf/2) x 64 x 64
        h2 = self.c2(h1)     # (nf) x 64 x 64
        h3 = self.c3(h2)     # (2*nf) x 32 x 32
        h4 = self.c4(h3)     # (2*nf) x 32 x 32
        h5 = self.c5(h4)     # (4*nf) x 16 x 16
        h6 = self.c6(h5)     # (4*nf) x 16 x 16          
        return h6, [h1, h2, h3, h4, h5, h6]


class image_decoder(nn.Module):
    def __init__(self, nc=1):
        super(image_decoder, self).__init__()
        nf = 16
        self.upc1 = dcgan_upconv(nf*4*2, nf*4, stride=1) #(nf*4) x 16 x 16
        self.upc2 = dcgan_upconv(nf*4*2, nf*2, stride=2) #(nf*2) x 32 x 32
        self.upc3 = dcgan_upconv(nf*2*2, nf*2, stride=1) #(nf*2) x 32 x 32
        self.upc4 = dcgan_upconv(nf*2*2, nf, stride=2)   #(nf) x 64 x 64
        self.upc5 = dcgan_upconv(nf*2, int(nf/2), stride=1)   #(nf/2) x 64 x 64
        self.upc6 = nn.ConvTranspose2d(in_channels=nf,out_channels=nc,kernel_size=(3,3),stride=1,padding=1)  #(nc) x 64 x 64

    def forward(self, input):
        vec, skip = input    # vec: (4*nf) x 16 x 16          
        [h1, h2, h3, h4, h5, h6] = skip
        d1 = self.upc1(torch.cat([vec, h6], dim=1))  #(nf*4) x 16 x 16
        d2 = self.upc2(torch.cat([d1, h5], dim=1))   #(nf*2) x 32 x 32
        d3 = self.upc3(torch.cat([d2, h4], dim=1))   #(nf*2) x 32 x 32
        d4 = self.upc4(torch.cat([d3, h3], dim=1))   #(nf) x 64 x 64
        d5 = self.upc5(torch.cat([d4, h2], dim=1))   #(nf/2) x 64 x 64
        d6 = self.upc6(torch.cat([d5, h1], dim=1))   #(nc) x 64 x 64
        return d6
        

class EncoderRNN(torch.nn.Module):
    def __init__(self,phycell,convlstm, device):
        super(EncoderRNN, self).__init__()
        self.image_cnn_enc = image_encoder().to(device) # image encoder 64x64x1 -> 16x16x64
        self.image_cnn_dec = image_decoder().to(device) # image decoder 16x16x64 -> 64x64x1 
        
        self.phycell = phycell.to(device)
        self.convlstm = convlstm.to(device)

        
    def forward(self, input, first_timestep=False, decoding=False):
        if decoding:  # input=None in decoding phase
            output_phys = None
        else:
            output_phys,skip = self.image_cnn_enc(input)
        output_conv,skip = self.image_cnn_enc(input)     

        hidden1, output1 = self.phycell(output_phys, first_timestep)
        hidden2, output2 = self.convlstm(output_conv, first_timestep)

        out_phys = torch.sigmoid(self.image_cnn_dec([output1[-1],skip])) # partial reconstructions for vizualization
        out_conv = torch.sigmoid(self.image_cnn_dec([output2[-1],skip]))

        concat = output1[-1]+output2[-1]
        output_image = torch.sigmoid( self.image_cnn_dec([concat,skip]) )
        return out_phys, hidden1, output_image, out_phys, out_conv

