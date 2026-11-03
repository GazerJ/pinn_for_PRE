import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os

# 设置随机种子
torch.manual_seed(43)

alpha0=1e3
gamma0=100
k=4
#Titheta=np.pi/2+0.5
os.system("cp model/B.m model/B_back.m ")

def gogogo():

    # ======================================
    # 1. 定义网络结构（强制边界约束）
    # ======================================

    #继承nn.Module
    class HardConstraintPINN(nn.Module):
        def __init__(self):
            #调用父级的初始化方法
            super(HardConstraintPINN, self).__init__()
            self.net = nn.Sequential(
                nn.Linear(5,800),
                nn.Tanh(),
                nn.Linear(800, 800),
                nn.Tanh(),
                nn.Linear(800, 800),
                nn.Tanh(),
                nn.Linear(800, 800),
                nn.Tanh(),
                nn.Linear(800, 8)
            )

            # 定义可学习的方差参数（log_var 保证数值稳定性）
            self.sigma1 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma2 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma3 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma4 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma5 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma6 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma7 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma8 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma9 = nn.Parameter(torch.tensor(1.0)) 
            self.sigma10 = nn.Parameter(torch.tensor(1.0)) 
        # 前向传播的意思就是 输入x 计算对应的u ，虽然self.net可以直接计算，但是这里可以通过构造一些距离函数强化边界
        def forward(self, x):
            y = self.net(x)
            #print(y.shape)
            theta,phi,Ttheta,Tphi,lx,ly,lz,l1= y[:, 0:1], y[:, 1:2],y[:, 2:3], y[:, 3:4], y[:, 4:5],y[:, 5:6], y[:, 6:7], y[:, 7:8]
            # 输入 x 的 shape: (N, 2), 包含坐标 (x, y)
            x_coord = x[:,0]  # x 分量
            Titheta=x[:,-1]# Titheta 
            
            
            # 计算距离函数 D(x,y) = x*(1-x)*y*(1-y),边界处距离函数为0，把网络的预测值消掉，进而叠加自己的边界值
            D0 = x_coord 
            D1= (1 - x_coord)
            # 网络预测残差项
            itheta=np.pi/2
            iphi=np.pi

            iTtheta=Titheta#+np.pi/2
            iTphi=np.pi/2
            
            ilx=0
            ily=0
            ilz=0
            il1=0
            # 合成最终解：u = initial + D * residual
            #print(theta.shape)
            y=[itheta+D0*theta[:,0],iphi+D0*phi[:,0],iTtheta+D0*Ttheta[:,0],iTphi+D0*Tphi[:,0],D1*lx[:,0],D1*ly[:,0],D1*lz[:,0],D1*l1[:,0]]
            
            return th.stack(y).T

    # ======================================
    # 2. 定义损失函数（仅 PDE 残差）
    # ======================================
    def compute_loss(model, points):

       
        X = points.clone().requires_grad_(True)
        #print(X.shape)
        x = X[:, 0]  # Already requires grad from X
        a=X[:,1]
        v=X[:,2]
        k=X[:,3]
        #y = model(X)#[:,0]  # Assuming model output is 3D [batch, vars, 1]
        y = model(X)#[:,0]
        alpha0=10**(a/2)
        beta0 =alpha0/2/(1+v)
        L=1


        alpha=alpha0*th.exp(k*L*x)
        beta=beta0*th.exp(k*L*x)
        gamma=gamma0*th.exp(-k*L*x)
       # print(gamma.shape)
        l2=x-1
        #print(y.shape)
        theta,phi,Ttheta,Tphi,lx,ly,lz,l1= y[:,0], y[:,1],y[:,2], y[:,3], y[:,4],y[:,5], y[:,6], y[:,7]

        theta=theta.squeeze()
        phi=phi.squeeze()
        Ttheta=Ttheta.squeeze()
        Tphi=Tphi.squeeze()
        lx=lx.squeeze()
        ly=ly.squeeze()
        lz=lz.squeeze()
        l1=l1.squeeze()

        #print(l1.shape)
        tx=th.sin(Ttheta)*th.cos(Tphi)
        ty=th.sin(Ttheta)*th.sin(Tphi)
        tz=th.cos(Ttheta)
        tzmean=th.mean(tz-0.000001)
       
        
        dtheta = torch.autograd.grad(theta, X, grad_outputs=torch.ones_like(theta) ,create_graph=True,retain_graph=True)[0][:,0]
        dphi = torch.autograd.grad(phi, X, grad_outputs=torch.ones_like(phi),create_graph=True,retain_graph=True)[0][:,0]
        dTtheta = torch.autograd.grad(Ttheta, X, grad_outputs=torch.ones_like(Ttheta),create_graph=True,retain_graph=True)[0][:,0]
        dTphi = torch.autograd.grad(Tphi, X, grad_outputs=torch.ones_like(Tphi),create_graph=True,retain_graph=True)[0][:,0]
        dlx = torch.autograd.grad(lx, X, grad_outputs=torch.ones_like(lx),create_graph=True,retain_graph=True)[0][:,0]
        dly = torch.autograd.grad(ly, X, grad_outputs=torch.ones_like(ly),create_graph=True,retain_graph=True)[0][:,0]
        dlz = torch.autograd.grad(lz, X, grad_outputs=torch.ones_like(lz),create_graph=True,retain_graph=True)[0][:,0]
        dl1 = torch.autograd.grad(l1, X, grad_outputs=torch.ones_like(l1),create_graph=True,retain_graph=True)[0][:,0]
    
        #print(dl1[0].shape)

 

        dtx=th.cos(Ttheta)*th.cos(Tphi)*dTtheta-th.sin(Ttheta)*th.sin(Tphi)*dTphi
        dty=th.cos(Ttheta)*th.sin(Tphi)*dTtheta+th.sin(Ttheta)*th.cos(Tphi)*dTphi
        dtz=-th.sin(Ttheta)*dTtheta
        
        kappa=(lx*th.sin(theta)*th.cos(phi)+ly*th.sin(theta)*th.sin(phi)+lz*th.cos(theta))/2/alpha+gamma
        tau=-l1/2/beta
        
        kt=[th.cos(theta)*th.cos(phi),
            th.cos(theta)*th.sin(phi),
            -th.sin(theta)]

        kf=[-th.sin(theta)*th.sin(phi),
            th.sin(theta)*th.cos(phi),
            0]



        ppx=(th.cos(phi-2*theta)-th.cos(phi+2*theta))/4
        ppy=(th.sin(phi-2*theta)-th.sin(phi+2*theta))/4
        PX=(ppy*dphi-th.sin(phi)*dtheta)
        PY=(-ppx*dphi+th.cos(phi)*dtheta)
        PZ=((th.sin(theta)**2)*dphi)
        
        Tdt=-tx*th.sin(phi)+ty*th.cos(phi)
        Tdf= tz

        eq1=  (-lx*kappa*kt[0]-ly*kappa*kt[1]-lz*kappa*kt[2]+dl1*Tdt)/kappa
        eq2=  (-lx*kappa*kf[0]-ly*kappa*kf[1]-lz*kappa*kf[2]+dl1*Tdf)/kappa

        #Mt=1#*th.sqrt(tx**2+ty**2+tz**2-1)
        eq3= dtx-kappa*th.sin(theta)*th.cos(phi)
        eq4= dty-kappa*th.sin(theta)*th.sin(phi)
        eq5= dtz-kappa*th.cos(theta)
        
        eq6= (dlx+l1*PX)#-Mt*tx
        eq7= (dly+l1*PY)#-Mt*ty
        eq8= (dlz+l1*PZ+l2)#-Mt*tz

        eq9= tau-tx*PX-ty*PY-tz*PZ

       

        eeq1=(eq1)**2 #**(1/4)#+sigma1#+th.log(sigma1)
        eeq2=(eq2)**2 #**(1/4)#+sigma2#+th.log(sigma2)
    
        eeq3=(eq3)**2 #**(1/4)#+sigma3#+th.log(sigma3)
        eeq4=(eq4)**2 ##**(1/4)#+sigma4#+th.log(sigma4)
        eeq5=(eq5)**2 #**(1/4)#+sigma5#+th.log(sigma5)        
        eeq6=(eq6)**2 #**(1/4)#+sigma6#+th.log(sigma6)
        eeq7=(eq7)**2 #**(1/4)#+sigma7#+th.log(sigma7)
        eeq8=(eq8)**2 #**(1/4)#+sigma8#+th.log(sigma8)
        eeq9=(eq9)**2 

                
 
        

        meeq1=th.mean(eeq1)
        meeq2=th.mean(eeq2)
        meeq3=th.mean(eeq3)
        meeq4=th.mean(eeq4)
        meeq5=th.mean(eeq5)
        meeq6=th.mean(eeq6)
        meeq7=th.mean(eeq7)
        meeq8=th.mean(eeq8)
        meeq9=th.mean(eeq9)
        



        eqs=th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9])



        Ssigma1=1#(th.std(th.tensor([meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9])))
        Ssigma2=1#(th.std(th.tensor([meeq1,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9])))
        Ssigma3=1#(th.std(th.tensor([meeq1,meeq2,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9])))
        Ssigma4=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq5,meeq6,meeq7,meeq8,meeq9])))
        Ssigma5=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq6,meeq7,meeq8,meeq9])))
        Ssigma6=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq7,meeq8,meeq9])))
        Ssigma7=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq8,meeq9])))
        Ssigma8=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq9])))
        Ssigma9=1#(th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8])))

        #LL=th.max(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9]))+th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9]))#+th.std(th.tensor([Ssigma1,Ssigma2,Ssigma3,Ssigma4,Ssigma5,Ssigma6,Ssigma7,Ssigma8,Ssigma9]))
        LL=th.mean(eqs)+th.std(eqs)+th.max(eqs)
        Asigma1=model.sigma1**2+1e-9
        Asigma2=model.sigma2**2+1e-9
        Asigma3=model.sigma3**2+1e-9
        Asigma4=model.sigma4**2+1e-9
        Asigma5=model.sigma5**2+1e-9
        Asigma6=model.sigma6**2+1e-9
        Asigma7=model.sigma7**2+1e-9
        Asigma8=model.sigma8**2+1e-9
        Asigma9=model.sigma9**2+1e-9
        

        loss1=(meeq1*Ssigma1)/Asigma1+th.log(Asigma1)
        loss2=(meeq2*Ssigma2)/Asigma2+th.log(Asigma2)    
        loss3=(meeq3*Ssigma3)/Asigma3+th.log(Asigma3)  
        loss4=(meeq4*Ssigma4)/Asigma4+th.log(Asigma4)  
        loss5=(meeq5*Ssigma5)/Asigma5+th.log(Asigma5)  
        loss6=(meeq6*Ssigma6)/Asigma6+th.log(Asigma6)  
        loss7=(meeq7*Ssigma7)/Asigma7+th.log(Asigma7)
        loss8=(meeq8*Ssigma8)/Asigma8+th.log(Asigma8)  
        loss9=(meeq9*Ssigma9)/Asigma9+th.log(Asigma9)
        loss=((loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9))

   


        return torch.mean(loss),LL,tzmean
        #return torch.mean(eeq1+eeq2+eeq3+eeq4+eeq5+eeq6+eeq7+eeq8+eeq9),torch.mean(eeq1+eeq2+eeq3+eeq4+eeq5+eeq6+eeq7+eeq8+eeq9)
    
    # ======================================
    # 3. 训练过程
    # ======================================
    model = HardConstraintPINN()
    file_path ="model/B-test.m"
    fig_path ="fig/B.pdf"
    
    if os.path.isfile(file_path):
        #print("load ..."+file_path)
        model.load_state_dict(th.load(file_path))

    npoint=100
    lll=4
   

    min_loss=np.inf
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-10*np.sqrt(npoint),amsgrad=True,weight_decay=0) # optimizer 可以直接修改model中的parameters
    epochs = 600000
    log_interval = 1000
    NMAX=400
    

    
    best_params = model.state_dict()
    for epoch in range(epochs):
      
        points = torch.rand(npoint, 5)  # 在 [0,1]内采样，生成1000个采样点，维度为2 矩阵 为1000行，2列 
        
        
        points[:,0]=th.linspace(0,1,npoint)#points[:,1]*10-5
        #points[:,1]=th.linspace(0,1,npoint)#points[:,1]*10-5
        #points[:,2]=th.linspace(0,1,npoint)#points[:,1]*10-5
        #points[:,3]=th.linspace(0,1,npoint)#points[:,1]*10-5
        #points[:,4]=th.linspace(0,1,npoint)#points[:,1]*10-5
        #points[:,0]=th.linspace(0,1,npoint)#points[:,1]*10-5
        
        points[:,1]=points[:,1]*(-8)
        points[:,2]=points[:,2]*1
        points[:,3]=points[:,3]*4
        points[:,4]=points[:,4]*np.pi/8+np.pi/2

        loss,Tloss,tzmean = compute_loss(model, points) #前向传播计算损失                
        if Tloss < min_loss:
            min_loss = Tloss
            epoch_i=epoch
            best_params = model.state_dict()
            th.save(model.state_dict(),file_path)
            if min_loss<float("5e-5"):
                model.load_state_dict(best_params)
                break
        #if epoch-epoch_i>10000:top
        #    model.load_state_dict(best_params)
        #    break
        if (epoch+1) % log_interval == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4e},    Minloss: {Tloss.item():.4e}")
        

        optimizer.zero_grad()# #optimizer 初始化-梯度清零 
        loss.backward() #计算过程中的 每个参数的梯度
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) #梯度剪裁 当范数超过1的时候，设为1
        optimizer.step() #optimizer 更新参数 
    #model.load_state_dict(best_params)
    model.load_state_dict(best_params)
    th.save(model.state_dict(),file_path)
    

gogogo()