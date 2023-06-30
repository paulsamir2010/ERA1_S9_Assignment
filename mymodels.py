import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#
# Assignment S9 with Dilated Convolution, use of stride as 2 and Albumentations for Augmentation (with CIFAR10 Dataset - Image size 32 x 32)
class Model_AdvConv_Albumentation(nn.Module):
    def __init__(self):
        super(Model_AdvConv_Albumentation, self).__init__()
        dout = 0.01
        # Input Block with convolution C11
        self.convblock_C11 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dout)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1 C12
        self.convblock_C12 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(3, 3), padding=1, bias=False,dilation =2), # DILATED CONVOLUTION
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dout)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1 C13  -- use of stride 2 instead of mxpool
        self.convblock_C13 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout(dout)
        ) # output_size = 16

        # TRANSITION BLOCK 1    -- 1 X 1 
        # self.convblock_T1 = nn.Sequential(
        # nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
       # ) # output_size = 16


        # CONVOLUTION BLOCK 2 C21
        self.convblock_C21 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dout)
        ) # output_size = 16 

        self.convblock_C22 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dout)
        ) # output_size = 16

        self.convblock_C23 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False, stride = 2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            #nn.Dropout(dout)
        )  # output_size = 8

        # TRANSITION BLOCK 2 -- 1X 1
        #self.convblock_T2 = nn.Sequential(
            #nn.Conv2d(in_channels=128, out_channels=32, kernel_size=(1, 1), padding=0, bias=False),
        #) # output_size = 8
        # P2


        # CONVOLUTION BLOCK 3 C31
        self.convblock_C31 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dout)
        ) # output_size = 8

        self.convblock_C32 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(dout)
        ) # output_size = 8 

        self.convblock_C33 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=2, bias=False, stride = 2),
            #nn.ReLU()
        )
            # output_size = 5 

        #  GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=5)
        ) 

        #CONVOLUTION BLOCK 1x 1 after gap
        self.convblock_T3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 5

        self.dropout = nn.Dropout(dout)

    def forward(self, x):
        x = self.convblock_C11(x) 
        x = self.convblock_C12(x) 
        x = self.convblock_C13(x)  # sTRIDE OF 2
        #x = self.convblock_T1(x)  # 1 X 1
        x = self.convblock_C21(x)
        x = self.convblock_C22(x)
        x = self.convblock_C23(x) # STRIDE OF 2
        #x = self.convblock_T2(x) # 1 x 1
        x = self.convblock_C31(x)
        x = self.convblock_C32(x) 
        x = self.convblock_C33(x) # STRIDE OF 2
        x = self.gap(x)
        x = self.convblock_T3(x) # 1 X 1
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Assignment S8 with Batch Normalization (with CIFAR10 Dataset - Image size 32 x 32)
class Model_Batch_Normalization(nn.Module):
    def __init__(self):
        super(Model_Batch_Normalization, self).__init__()
        dout = 0.1
        # Input Block with convolution C1
        self.convblock_C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dout)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1 C2
        self.convblock_C2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(8),
            nn.Dropout(dout)
        ) # output_size = 32

        # TRANSITION BLOCK 1 c3    
        self.convblock_c3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=7, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32
        # P1
        self.pool_P1 = nn.MaxPool2d(2, 2) 
        # output_size = 16

        # CONVOLUTION BLOCK 2 C3
        self.convblock_C3 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dout)
        ) # output_size = 16 

        self.convblock_C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dout)
        ) # output_size = 16

        self.convblock_C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(dout)
        )
            # output_size = 16

        # TRANSITION BLOCK 2 c6
        self.convblock_c6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16
        # P2
        self.pool_P2 = nn.MaxPool2d(2, 2) 
        # output_size = 8

        # CONVOLUTION BLOCK 3 C7
        self.convblock_C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dout)
        ) # output_size = 8

        self.convblock_C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Dropout(dout)
        ) # output_size = 8 

        self.convblock_C9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU())
            # output_size = 8 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        #CONVOLUTION BLOCK C10
        self.convblock_C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(dout)

    def forward(self, x):
        x = self.convblock_C1(x) 
        x = x + self.convblock_C2(x) 
        x = self.convblock_c3(x) # 1 x 1
        x = self.pool_P1(x)
        x = self.convblock_C3(x)
        x = x + self.convblock_C4(x)
        x = x + self.convblock_C5(x)
        x = self.convblock_c6(x) # 1 x 1
        x = self.pool_P2(x) 
        x = self.convblock_C7(x)
        x = x + self.convblock_C8(x)
        x = x + self.convblock_C9(x)
        x = self.gap(x)
        x = self.convblock_C10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Assignment S8 with Layer Normalization (with CIFAR10 Dataset - Image size 32 x 32)

class Model_Layer_Normalization(nn.Module):
    def __init__(self):
        super(Model_Layer_Normalization, self).__init__()
        dout = 0.1
        # Input Block with convolution C1
        self.convblock_C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,8), 
            nn.Dropout(dout)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1 C2
        self.convblock_C2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,8),
            nn.Dropout(dout)
        ) # output_size = 32

        # TRANSITION BLOCK 1 c3    
        self.convblock_c3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=7, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32
        # P1
        self.pool_P1 = nn.MaxPool2d(2, 2) 
        # output_size = 16

        # CONVOLUTION BLOCK 2 C3
        self.convblock_C3 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,16),
            nn.Dropout(dout)
        ) # output_size = 16 

        self.convblock_C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,16),
            nn.Dropout(dout)
        ) # output_size = 16

        self.convblock_C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,16),
            nn.Dropout(dout)
        )
            # output_size = 16

        # TRANSITION BLOCK 2 c6
        self.convblock_c6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16
        # P2
        self.pool_P2 = nn.MaxPool2d(2, 2) 
        # output_size = 8

        # CONVOLUTION BLOCK 3 C7
        self.convblock_C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,32),
            nn.Dropout(dout)
        ) # output_size = 8

        self.convblock_C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(1,32),
            nn.Dropout(dout)
        ) # output_size = 8 

        self.convblock_C9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU())
            # output_size = 8 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        #CONVOLUTION BLOCK C10
        self.convblock_C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(dout)

    def forward(self, x):
        x = self.convblock_C1(x) 
        x = x + self.convblock_C2(x) 
        x = self.convblock_c3(x) # 1 x 1
        x = self.pool_P1(x)
        x = self.convblock_C3(x)
        x = x + self.convblock_C4(x)
        x = x + self.convblock_C5(x)
        x = self.convblock_c6(x) # 1 x 1
        x = self.pool_P2(x) 
        x = self.convblock_C7(x)
        x = x + self.convblock_C8(x)
        x = x + self.convblock_C9(x)
        x = self.gap(x)
        x = self.convblock_C10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Assignment S8 with Group Normalization (with CIFAR10 Dataset - Image size 32 x 32)

class Model_Group_Normalization(nn.Module):
    def __init__(self):
        super(Model_Group_Normalization, self).__init__()
        dout = 0.1
        # Input Block with convolution C1
        self.convblock_C1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,8), 
            nn.Dropout(dout)
        ) # output_size = 32

        # CONVOLUTION BLOCK 1 C2
        self.convblock_C2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,8),
            nn.Dropout(dout)
        ) # output_size = 32

        # TRANSITION BLOCK 1 c3    
        self.convblock_c3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=7, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 32
        # P1
        self.pool_P1 = nn.MaxPool2d(2, 2) 
        # output_size = 16

        # CONVOLUTION BLOCK 2 C3
        self.convblock_C3 = nn.Sequential(
            nn.Conv2d(in_channels=7, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout(dout)
        ) # output_size = 16 

        self.convblock_C4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout(dout)
        ) # output_size = 16

        self.convblock_C5 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,16),
            nn.Dropout(dout)
        )
            # output_size = 16

        # TRANSITION BLOCK 2 c6
        self.convblock_c6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 16
        # P2
        self.pool_P2 = nn.MaxPool2d(2, 2) 
        # output_size = 8

        # CONVOLUTION BLOCK 3 C7
        self.convblock_C7 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,32),
            nn.Dropout(dout)
        ) # output_size = 8

        self.convblock_C8 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
            nn.GroupNorm(4,32),
            nn.Dropout(dout)
        ) # output_size = 8 

        self.convblock_C9 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU())
            # output_size = 8 

        # OUTPUT BLOCK GAP
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=8)
        ) 

        #CONVOLUTION BLOCK C10
        self.convblock_C10 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )

        self.dropout = nn.Dropout(dout)

    def forward(self, x):
        x = self.convblock_C1(x) 
        x = x + self.convblock_C2(x) 
        x = self.convblock_c3(x) # 1 x 1
        x = self.pool_P1(x)
        x = self.convblock_C3(x)
        x = x + self.convblock_C4(x)
        x = x + self.convblock_C5(x)
        x = self.convblock_c6(x) # 1 x 1
        x = self.pool_P2(x) 
        x = self.convblock_C7(x)
        x = x + self.convblock_C8(x)
        x = x + self.convblock_C9(x)
        x = self.gap(x)
        x = self.convblock_C10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

# Assignment S8 Wrapper Function for invoking Model

def S8_model(norm_type):
    if norm_type == 'BN':
        model = Model_Batch_Normalization()
    elif norm_type== 'GN':
        model = Model_Group_Normalization()
    elif norm_type == 'LN':
        model = Model_Layer_Normalization()
    else:
        raise Exception("Normalization type should be 'BN' or 'GN' or 'LN'")
    
    return model


# Previous Assignment (prior to S8) - Lighter Model with Batch Normalization

class Net1(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
         ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
        ) # output_size = 6


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


        self.dropout = nn.Dropout(dropout_value)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        #x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

#Add Dropout Regularization on Net1 Model (Lighter Model with Batch Normalization) - Used in Assignment Prior to S8

class Net2(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=12, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(12),
            nn.Dropout(0.1)
        ) # output_size = 26

        # CONVOLUTION BLOCK 1
        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.1)
        ) # output_size = 24

        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        ) # output_size = 24
        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 12

        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=14, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(14),
            nn.Dropout(0.1)
        ) # output_size = 10
        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=14, out_channels=16, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Dropout(0.1)
        ) # output_size = 8
        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(18),
            nn.Dropout(0.1)
        ) # output_size = 6


        # OUTPUT BLOCK
        self.gap = nn.Sequential(
            nn.AvgPool2d(kernel_size=6)
        ) # output_size = 1

        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
        )


        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        #x = self.convblock7(x)
        x = self.gap(x)
        x = self.convblock8(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)


