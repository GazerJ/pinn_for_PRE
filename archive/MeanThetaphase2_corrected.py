
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch as th
import os
import scienceplots
import numpy as np
plt.style.use(['science'])
# NOTE: paper figures use LaTeX-rendered Computer Modern (text.usetex=True, serif,
# mathtext cm) = scienceplots 'science' base style. 'nature'/'no-latex' would give
# DejaVu Sans and break font consistency with the paper's existing figures.
# 设置随机种子
torch.manual_seed(42)
labelsize=24
tickssize=24

labelsize2=32
tickssize2=28

alpha0=1e3
gamma0=100
from scipy import interpolate, signal
Titheta=np.pi/2
def find_inflection_points(x, y, smoothing_factor=0.5, min_distance=5):
    """
    寻找曲线的拐点（二阶导数为零的点）
    
    参数:
    x: x坐标数据
    y: y坐标数据
    smoothing_factor: 平滑因子，越大平滑程度越高
    min_distance: 拐点之间的最小距离（避免检测到相邻的假拐点）
    
    返回:
    inflection_points: 拐点坐标列表 [(x1, y1), (x2, y2), ...]
    """
    
    # 确保输入是numpy数组
    x = np.array(x)
    y = np.array(y)
    
    # 1. 平滑处理 - 使用Savitzky-Golay滤波器减少小波动
    window_length = min(11, len(y) // 10 * 2 + 1)  # 自适应窗口大小
    if window_length > 3:
        y_smooth = signal.savgol_filter(y, window_length=window_length, polyorder=3)
    else:
        y_smooth = y
    
    # 2. 使用样条插值获得平滑的函数表示
    tck = interpolate.splrep(x, y_smooth, s=len(x)*smoothing_factor)
    
    # 3. 计算一阶和二阶导数
    x_dense = np.linspace(x.min(), x.max(), len(x)*10)  # 更密集的点用于精确计算
    y_spline = interpolate.splev(x_dense, tck, der=0)
    dy_dx = interpolate.splev(x_dense, tck, der=1)      # 一阶导数
    d2y_dx2 = interpolate.splev(x_dense, tck, der=2)    # 二阶导数
    
    # 4. 寻找二阶导数为零的点（符号变化的点）
    zero_crossings = np.where(np.diff(np.sign(d2y_dx2)))[0]
    
    # 5. 精确定位拐点位置
    inflection_points = []
    for idx in zero_crossings:
        if idx > 0 and idx < len(x_dense) - 1:
            # 使用线性插值更精确地找到零点位置
            x1, x2 = x_dense[idx], x_dense[idx+1]
            y1, y2 = d2y_dx2[idx], d2y_dx2[idx+1]
            
            if y1 * y2 < 0:  # 确认确实有符号变化
                # 线性插值找到精确的零点
                t = -y1 / (y2 - y1) if (y2 - y1) != 0 else 0.5
                x_inflect = x1 + t * (x2 - x1)
                y_inflect = interpolate.splev(x_inflect, tck, der=0)
                inflection_points.append((x_inflect, y_inflect))
    
    # 6. 去除过于接近的拐点
    if len(inflection_points) > 1 and min_distance > 0:
        filtered_points = [inflection_points[0]]
        for point in inflection_points[1:]:
            if abs(point[0] - filtered_points[-1][0]) >= min_distance * (x.max() - x.min()) / len(x):
                filtered_points.append(point)
        inflection_points = filtered_points
    
    return inflection_points, x_dense, y_spline, dy_dx, d2y_dx2


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
        Titheta=x[:,-1]
        
        # 计算距离函数 D(x,y) = x*(1-x)*y*(1-y),边界处距离函数为0，把网络的预测值消掉，进而叠加自己的边界值
        D0 = x_coord 
        D1= (1 - x_coord)
        # 网络预测残差项
        itheta=np.pi/2
        iphi=np.pi

        iTtheta=Titheta
        iTphi=np.pi/2
        
        ilx=0
        ily=0
        ilz=0
        il1=0
        # 合成最终解：u = initial + D * residual
        #print(theta.shape)
        y=[itheta+D0*theta[:,0],iphi+D0*phi[:,0],iTtheta+D0*Ttheta[:,0],iTphi+D0*Tphi[:,0],D1*lx[:,0],D1*ly[:,0],D1*lz[:,0],D1*l1[:,0]]
        
        return th.stack(y).T




file_pathA ="model/B.m"
file_pathB ="model/B.m"

modelA = HardConstraintPINN()
modelA.load_state_dict(th.load(file_pathA))

modelB = HardConstraintPINN()
modelB.load_state_dict(th.load(file_pathB))

#if os.path.isfile(file_path):
#    print("load ..."+file_path)
#    model.load_state_dict(th.load(file_path))
def compute_loss(model, points):

    
    X = points.clone().requires_grad_(True)
    x = X[:, 0]  # Already requires grad from X
    a=X[:,1]
    v=X[:,2]
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

    #print(l1.shape)
    tx=th.sin(Ttheta)*th.cos(Tphi)
    ty=th.sin(Ttheta)*th.sin(Tphi)
    tz=th.cos(Ttheta)
    tzmean=th.mean(tz)
    
    
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

    eq1=  (-lx*kappa*kt[0]-ly*kappa*kt[1]-lz*kappa*kt[2]+dl1*Tdt)
    eq2=  (-lx*kappa*kf[0]-ly*kappa*kf[1]-lz*kappa*kf[2]+dl1*Tdf)

    #Mt=1#*th.sqrt(tx**2+ty**2+tz**2-1)
    eq3= dtx-kappa*th.sin(theta)*th.cos(phi)
    eq4= dty-kappa*th.sin(theta)*th.sin(phi)
    eq5= dtz-kappa*th.cos(theta)
    
    eq6= dlx+l1*PX#-Mt*tx
    eq7= dly+l1*PY#-Mt*ty
    eq8= dlz+l1*PZ+l2#-Mt*tz

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

            
    Ssigma1=th.log(th.std(eeq1)+1)
    Ssigma2=th.log(th.std(eeq2)+1)
    Ssigma3=th.log(th.std(eeq3)+1)
    Ssigma4=th.log(th.std(eeq4)+1)
    Ssigma5=th.log(th.std(eeq5)+1)
    Ssigma6=th.log(th.std(eeq6)+1)
    Ssigma7=th.log(th.std(eeq7)+1)
    Ssigma8=th.log(th.std(eeq8)+1)
    Ssigma9=th.log(th.std(eeq9)+1)
    

    meeq1=th.mean(eeq1)
    meeq2=th.mean(eeq2)
    meeq3=th.mean(eeq3)
    meeq4=th.mean(eeq4)
    meeq5=th.mean(eeq5)
    meeq6=th.mean(eeq6)
    meeq7=th.mean(eeq7)
    meeq8=th.mean(eeq8)
    meeq9=th.mean(eeq9)
    #print([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9])
    LL=th.std(th.tensor([meeq1,meeq2,meeq3,meeq4,meeq5,meeq6,meeq7,meeq8,meeq9]))

    Asigma1=model.sigma1**2+1e-9
    Asigma2=model.sigma2**2+1e-9
    Asigma3=model.sigma3**2+1e-9
    Asigma4=model.sigma4**2+1e-9
    Asigma5=model.sigma5**2+1e-9
    Asigma6=model.sigma6**2+1e-9
    Asigma7=model.sigma7**2+1e-9
    Asigma8=model.sigma8**2+1e-9
    Asigma9=model.sigma9**2+1e-9
    

    loss1=(meeq1+Ssigma1)/Asigma1+th.log(Asigma1)  
    loss2=(meeq2+Ssigma2)/Asigma2+th.log(Asigma2)    
    loss3=(meeq3+Ssigma3)/Asigma3+th.log(Asigma3)  
    loss4=(meeq4+Ssigma4)/Asigma4+th.log(Asigma4)  
    loss5=(meeq5+Ssigma5)/Asigma5+th.log(Asigma5)  
    loss6=(meeq6+Ssigma6)/Asigma6+th.log(Asigma6)  
    loss7=(meeq7+Ssigma7)/Asigma7+th.log(Asigma7)
    loss8=(meeq8+Ssigma8)/Asigma8+th.log(Asigma8)  
    loss9=(meeq9+Ssigma9)/Asigma9+th.log(Asigma9)
    loss=((loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8+loss9))




    return torch.mean(loss),LL+1e1*th.sign(tzmean)+1e1,tzmean
        #return torch.mean(eeq1+eeq2+eeq3+eeq4+eeq5+eeq6+eeq7+eeq8+eeq9),torch.mean(eeq1+eeq2+eeq3+eeq4+eeq5+eeq6+eeq7+eeq8+eeq9)
    
def draw(a,v,K):
    def compute_number_of_turns(x, y, z):
        # 将输入转换为numpy数组
        x = np.asarray(x)
        y = np.asarray(y)
        
        # 计算角度theta（弧度）
        theta = np.arctan2(y, x)
        
        # 展开角度以消除2π的跳变
        theta_unwrapped = np.unwrap(theta)
        
        # 计算总的角度变化量
        delta_theta = theta_unwrapped[-1] - theta_unwrapped[0]
        
        # 计算圈数：总角度变化除以2π
        number_of_turns = abs(delta_theta) / (2 * np.pi)
        
        return number_of_turns
    fig_path ="fig/a_%.2frate%.2ftheta%.2f.pdf"%(a,v,Titheta)
    fig_path2 ="fig/xya_%.2frate%.2ftheta%.2f.pdf"%(a,v,Titheta)
    if a<=0:
        #print("load A")
        model = modelA
       # model.load_state_dict(th.load(file_pathA))
    if a>0:
        #print("load B")
        model = modelB
       
        
    ds=0.001
    s=torch.arange(0,1,ds)
    Aa=torch.ones(len(s))*a
    Av=torch.ones(len(s))*v
    k=torch.ones(len(s))*K
    titheta=torch.ones(len(s))*Titheta
    x=th.stack((s,Aa,Av,k,titheta),dim=0).T
    #print(x[:,1])
    y = model(x).detach().numpy()#[:,:,0]
    #loss,Tloss,tzmean = compute_loss(model, x) #前向传播计算损失
    #print(tzmean)  
    theta,phi,Ttheta,Tphi,lx,ly,lz,l1= y[:,0], y[:,1],y[:,2], y[:,3], y[:,4],y[:,5], y[:,6], y[:,7]
    
    tx=np.sin(Ttheta)*np.cos(Tphi)
    ty=np.sin(Ttheta)*np.sin(Tphi)
    tz=np.cos(Ttheta)
    #print(np.mean(tz))
    x=np.array([np.sum(tx[:i])*ds for i in range(len(tx))])
    y=np.array([np.sum(ty[:i])*ds for i in range(len(tx))])
    z=np.array([np.sum(tz[:i])*ds for i in range(len(tx))])

    
    """
    fig= plt.figure(figsize=(8,6))
    ax=fig.add_subplot(1,1,1)
    
    plt.plot(x,z,lw=3,color="blue")
    plt.xlim(-0.5,0.5)
    plt.ylim(-1,0)
    plt.xlabel("$\\rm x$",fontsize=tickssize2)
    plt.ylabel("$\\rm z$",fontsize=tickssize2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.xticks(fontsize=labelsize2)
    plt.yticks(fontsize=labelsize2)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close() 
    fig= plt.figure(figsize=(8,6))
    plt.plot(x,y,lw=3,color="blue")
    plt.xlim(-0.2,0.2)
    plt.ylim(-0.2,0.2)
    plt.xlabel("$\\rm x$",fontsize=tickssize2)
    plt.ylabel("$\\rm z$",fontsize=tickssize2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)
    plt.xticks(fontsize=labelsize2)
    plt.yticks(fontsize=labelsize2)
    plt.tight_layout()
    plt.savefig(fig_path2)
    plt.close() 

    """
    n=compute_number_of_turns(x,y,z)
    return -z.mean(),n
    #return (np.sqrt(tz[1:]**2)).mean()#/np.sqrt((tx[1:]**2+ty[1:]**2))
    #return (np.sqrt(tz[1:]**2)/np.sqrt((tx[1:]**2+ty[1:]**2))).mean(),n

oo=int(os.environ.get('OO','1'))
# ================= outer shell (corrected I=pi a^4/4, J=pi a^4/2) =================
# The trained model embeds beta0=alpha0/2/(1+v), i.e. the OLD assumption I=J=pi a^4/2
# (then GJ/EI = G/E = 1/(2(1+nu))). The paper text is updated to I=pi a^4/4, J=pi a^4/2,
# so the correct ratio is GJ/EI = 2G/E = 1/(1+nu). To reproduce the correct physics with
# the SAME trained network, feed an effective v_eff=(nu-1)/2 so that
# 1/(2(1+v_eff)) = 1/(1+nu). The y-axis below still shows the TRUE nu in [-1, 0.5].
def nu_to_vfeed(nu):
    return (nu - 1.0) / 2.0

if oo==1:
    Z=np.zeros((64*2+1,30*2+1))   
    N=np.zeros((64*2+1,30*2+1)) 
    for i in range(0,65+64):
        print(i)
        for j in range(61):
            nu_true = 0.025*j-1                 # true Poisson ratio, -1 .. 0.5
            v_feed  = nu_to_vfeed(nu_true)      # shell mapping into trained model input
            Z[i,j],N[i,j]=(draw((i+(-16)*7)*0.125,v_feed,4))
    Z=-Z
    np.savetxt("corrected/zm.csv",Z)
    np.savetxt("corrected/nm.csv",N)
else:
    Z=np.loadtxt("corrected/zm.csv")[:,:]
    N=np.loadtxt("corrected/nm.csv")[:,:]



def remove_outlier_replace_with_mean(matrix):
    """
    输入：二维矩阵（列表的列表或numpy数组）
    输出：异常值替换为周围均值的矩阵（numpy数组）
    """
    arr = np.array(matrix)
    rows, cols = arr.shape
    
    # 计算IQR范围
    flat_data = arr.flatten()
    Q1 = np.percentile(flat_data, 25)
    Q3 = np.percentile(flat_data, 75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR

    # 查找所有异常点坐标
    outlier_coords = np.argwhere(arr > upper_bound)

    # 遍历所有异常点进行替换
    for r, c in outlier_coords:
        neighbors = []
        # 获取上下左右四个有效邻域值
        if r > 0: neighbors.append(arr[r-1, c])    # 上
        if r < rows-1: neighbors.append(arr[r+1, c]) # 下
        if c > 0: neighbors.append(arr[r, c-1])     # 左
        if c < cols-1: neighbors.append(arr[r, c+1]) # 右
        
        # 如果有邻居则替换为均值，否则保留原值
        if len(neighbors) > 0:
            arr[r, c] = np.mean(neighbors)

    return arr
#Z=remove_outlier_replace_with_mean(Z)

import matplotlib.colors as mcolors
colors = ['#9C8147', '#47629C'] 
colors2 = ['#9C8147',"#57479C", '#47629C'] 
colors3 = ['#9C8147',"#57479C", '#47629C',"#9C4762"] 

colors4 = ['#9C8147',"#9C4762"] 


custom_cmap = mcolors.LinearSegmentedColormap.from_list('my_custom_gradient', colors, N=256)
custom_cmap2 = mcolors.LinearSegmentedColormap.from_list('my_custom_gradient', colors3, N=256)
custom_cmap3 = mcolors.LinearSegmentedColormap.from_list('my_custom_gradient', colors4, N=256)

plt.figure(figsize=(8,6))
fig, ax = plt.subplots()
# 调整子图参数，设置左边距为0.2
fig.subplots_adjust(left=0.2)
plt.imshow(Z.T,interpolation='antialiased',origin='lower',cmap=custom_cmap,aspect='auto')
#plt.xticks(range(0,23,5),"%.2f")
plt.yticks(range(0,61,20),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
plt.xticks(range(0,129,32),["$10^{%.d}$"%((-8+i/2)) for i in np.arange(0,17,4)])
plt.xlabel("${\\rm \\bf Bending~stiffness/electric~field~ratio}~\\alpha_0$",fontsize=labelsize)
plt.ylabel("${\\rm  \\bf Poisson's~ratio}~\\nu$",fontsize=labelsize)
cb=plt.colorbar(aspect =15,fraction=0.04)
cb.set_label("${\\rm \\bf Mean~depth}~<z>$", fontsize=labelsize)
cb.ax.tick_params(labelsize=tickssize)
plt.tight_layout()

plt.xticks(fontsize=tickssize)
plt.yticks(fontsize=tickssize)
plt.savefig("corrected/phasez.pdf")

from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.close()
fig, ax = plt.subplots(figsize=(8, 6))
im =ax.imshow(N.T[:,:-128//8],interpolation='antialiased',origin='lower',cmap=custom_cmap,aspect='auto')
plt.yticks(range(0,61,20),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-7+i/2)) for i in np.arange(0,13,4)])
plt.xlabel("${\\rm \\bf Bending~stiffness/electric~field~ratio}~\\alpha_0$",fontsize=labelsize)
plt.ylabel("${\\rm \\bf Poisson's~ratio}~\\nu$",fontsize=labelsize)
# cb=plt.colorbar(aspect =15,fraction=0.04)
# cb.set_label("${\\rm \\bf Spiral~number}~n$", fontsize=labelsize)
# cb.ax.tick_params(labelsize=tickssize)

plt.xticks(fontsize=tickssize)
plt.yticks(fontsize=tickssize)
divider = make_axes_locatable(ax)
# 在右侧添加一个 axes，宽度为 "5%"，间距为 0.05
cax = divider.append_axes("right", size="5%", pad=0.15)

# 3. 将 colorbar 绘制在这个新的 cax 上
cb=plt.colorbar(im, cax=cax,aspect =15,fraction=0.04, )

#cb=plt.colorbar(aspect =15,fraction=0.04, shrink=1.5)
cb.set_label("${\\rm \\bf Spiral~number}~n$", fontsize=labelsize)
cb.ax.tick_params(labelsize=tickssize)

plt.tight_layout()

plt.savefig("corrected/phasen.pdf")
plt.close()


#t=np.where(N<2.5)
#Z[t]=np.nan


from scipy import ndimage
plt.grid(False)
o=4
kernel = np.ones((o,o)) / o/o  # 5x5 均值滤波器
fig, ax = plt.subplots(figsize=(8, 6))
blurred = ndimage.convolve(Z, kernel)
im = ax.imshow(blurred.T[:,:-128//8],interpolation='antialiased',origin='lower',cmap=custom_cmap2,aspect='auto')
#plt.xticks(range(0,23,5),"%.2f")
#plt.yticks(range(0,151,50),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
#plt.xticks(range(0,161,40),["$10^{%.d}$"%((-8+i)//2) for i in np.arange(0,17,4)])
#plt.yticks(range(0,61,20),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
#plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-8+i/2)) for i in np.arange(0,13,4)])
plt.yticks(range(0,61,20),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-7+i/2)) for i in np.arange(0,13,4)])

plt.xlabel("${\\rm \\bf Bending~stiffness/electric~field~ratio}~\\alpha_0$",fontsize=labelsize)
plt.ylabel("${\\rm \\bf Poisson's~ratio}~\\nu$",fontsize=labelsize)
#from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.xticks(fontsize=tickssize)
plt.yticks(fontsize=tickssize)

divider = make_axes_locatable(ax)
# 在右侧添加一个 axes，宽度为 "5%"，间距为 0.05
cax = divider.append_axes("right", size="5%", pad=0.15)

# 3. 将 colorbar 绘制在这个新的 cax 上
cb=plt.colorbar(im, cax=cax,aspect =15,fraction=0.04, )

#cb=plt.colorbar(aspect =15,fraction=0.04, shrink=1.5)
cb.set_label("${\\rm \\bf Mean~depth}~<z>$", fontsize=labelsize)
cb.ax.tick_params(labelsize=tickssize)
plt.tight_layout()
plt.savefig("corrected/phasez.pdf")
plt.close()


#x=[10**((-10+i*1)/2) for i in np.arange(0,201)]
#plt.plot(blurred[::,0]/np.min(blurred),label="$\\nu=-1.0$")
#plt.plot(blurred[::,5]/np.min(blurred),label="$\\nu=-0.5$")
#plt.plot(blurred[::,10]/np.min(blurred),label="$\\nu=0$")
#plt.plot(blurred[::,15]/np.min(blurred),label="$\\nu=0.5$")


#tickssize=
#labelsize=
plt.figure(figsize=(8,6))
plt.plot(blurred[:-128//8,0],label="$\\nu=-1.0$",lw=3,c="#47629C")
plt.plot(blurred[:-128//8,len(Z[0,])//3],label="$\\nu=-0.5$",lw=3,c="#629C47")
plt.plot(blurred[:-128//8,-len(Z[0,])//3],label="$\\nu=0$",lw=3,c="#9C8147")
plt.plot(blurred[:-128//8,-1],label="$\\nu=0.5$",lw=3,c="#9C4762")
#plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-8+i/2)) for i in np.arange(0,13,4)])

plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-7+i/2)) for i in np.arange(0,13,4)])
plt.xlabel("${\\rm \\bf Bending~stiffness/electric~field~ratio}~\\alpha_0$",fontsize=labelsize2)
plt.ylabel("${\\rm \\bf Mean~depth}~<z>$",fontsize=labelsize2)

plt.xticks(fontsize=tickssize)
plt.yticks(fontsize=tickssize)
#plt.yticks(range(0,16,5),["$%.1f$"%i for i in np.arange(-1,0.6,0.5)])
#plt.xticks(range(0,201,40),["$10^{%.d}$"%((-10+i)//2) for i in np.arange(0,21,4)])
#plt.yscale("log")
#plt.xscale("log")
plt.legend(fontsize=tickssize)
plt.tight_layout()
plt.savefig("corrected/Z.pdf")
plt.close()




dz1=np.diff(blurred[:-128//8,0],n=1)
dz2=np.diff(blurred[:-128//8,len(Z[0,])//3],n=1)
dz3=np.diff(blurred[:-128//8,-len(Z[0,])//3],n=1)
dz4=np.diff(blurred[:-128//8,-1],n=1)

#第一个点是 变化率开始的地方
alp1=np.array(np.where(dz1>1.5*(np.min(dz1)))[0])
alp2=np.array(np.where(dz2>1.5*np.min(dz2))[0])
alp3=np.array(np.where(dz3>1.5*np.min(dz3))[0])
alp4=np.array(np.where(dz4>1.5*np.min(dz4))[0])

#po1=alp1[0]

ay1=blurred[:-128//8,0];ax1=range(len(ay1))
ay2=blurred[:-128//8,len(Z[0,])//3];ax2=range(len(ay2))
ay3=blurred[:-128//8,-len(Z[0,])//3];ax3=range(len(ay3))
ay4=blurred[:-128//8,-1];ax4=range(len(ay4))

aay1=np.diff(blurred[:-128//8,0]);aax1=range(len(aay1))
aay2=np.diff(blurred[:-128//8,len(Z[0,])//3]);aax2=range(len(aay2))
aay3=np.diff(blurred[:-128//8,-len(Z[0,])//3]);aax3=range(len(aay3))
aay4=np.diff(blurred[:-128//8,-1]);aax4=range(len(aay4))



plt.figure(figsize=(8,6))
# plt.plot((blurred[:-128//8,0]),label="$\\nu=-1.0$",lw=3,zorder=1)
# plt.plot((blurred[:-128//8,len(Z[0,])//3]),label="$\\nu=-0.5$",lw=3,zorder=1)
# plt.plot((blurred[:-128//8,-len(Z[0,])//3]),label="$\\nu=0$",lw=3,zorder=1)
# plt.plot((blurred[:-128//8,-1]),label="$\\nu=0.5$",lw=3,zorder=1)

plt.plot(blurred[:-128//8,0],label="$\\nu=-0.99$",lw=3,c="#47629C")
plt.plot(blurred[:-128//8,len(Z[0,])//3],label="$\\nu=-0.5$",lw=3,c="#629C47")
plt.plot(blurred[:-128//8,-len(Z[0,])//3],label="$\\nu=0$",lw=3,c="#9C8147")
plt.plot(blurred[:-128//8,-1],label="$\\nu=0.5$",lw=3,c="#9C4762")
#second hline

ay1=blurred[:-128//8,0];ax1=range(len(ay1))

aax1=int(47-5)
aax2=int(47-5)
aax3=int(47-5)
aax4=int(47-5)

aaz1=blurred[aax1,0]
aaz2=blurred[aax2,len(Z[0,])//3]
aaz3=blurred[aax3,-len(Z[0,])//3]
aaz4=blurred[aax4,-1]


plt.scatter(aax1,aaz1,s=50,color="#47629C",zorder=2)
plt.scatter(aax2,aaz2,s=50,color="#629C47",zorder=2)
plt.scatter(aax3,aaz3,s=50,color="#9C8147",zorder=2)
plt.scatter(aax4,aaz4,s=50,color="#9C4762",zorder=2)

alp1=np.array(np.where(dz1>5*np.abs(np.min(dz1)))[0])
alp2=np.array(np.where(dz2>5*np.abs(np.min(dz2)))[0])
alp3=np.array(np.where(dz3>5*np.abs(np.min(dz3)))[0])
alp4=np.array(np.where(dz4>5*np.abs(np.min(dz4)))[0])


print(alp1)

aax1=alp1[-1]#int(axy1[0][0]//1)
print(aax1)
aax2=alp2[-1]#int(axy2[0][0]//1)
print(aax2)
aax3=alp3[-1]#int(axy3[0][0]//1)
print(aax3)
aax4=alp4[-1]#int(axy4[0][0]//1)
print(aax4)

aaz1=blurred[aax1,0]
aaz2=blurred[aax2,len(Z[0,])//3]
aaz3=blurred[aax3,-len(Z[0,])//3]
aaz4=blurred[aax4,-1]


#the curve line
plt.scatter(aax1,aaz1,s=50,color="#47629C",zorder=2)
plt.scatter(aax2,aaz2,s=50,color="#629C47",zorder=2)
plt.scatter(aax3,aaz3,s=50,color="#9C8147",zorder=2)
plt.scatter(aax4,aaz4,s=50,color="#9C4762",zorder=2)
print("ax:")
print(aax1*8/128-7)
print(aax4*8/128-7)



aax1=int((-5.75+7)*128//8)
aaz1=blurred[aax1,0]
aaz2=blurred[aax1,len(Z[0,])//3]
aaz3=blurred[aax1,-len(Z[0,])//3]
aaz4=blurred[aax1,-1]
plt.scatter(aax1,aaz1,s=50,color="#47629C",zorder=2)
plt.scatter(aax1,aaz2,s=50,color="#629C47",zorder=2)
plt.scatter(aax1,aaz3,s=50,color="#9C8147",zorder=2)
plt.scatter(aax1,aaz4,s=50,color="#9C4762",zorder=2)

plt.xticks(range(0,128*3//4+1,32),["$10^{%.d}$"%((-7+i/2)) for i in np.arange(0,13,4)])
plt.xlabel("${\\rm \\bf Bending~stiffness/electric~field~ratio}~\\alpha_0$",fontsize=labelsize)
plt.ylabel("${\\rm \\bf Mean~depth}~<z>$",fontsize=labelsize)
plt.xticks(fontsize=tickssize)
plt.yticks(fontsize=tickssize)

plt.legend(fontsize=tickssize)
plt.tight_layout()
plt.savefig("corrected/dZ.pdf")
plt.close()