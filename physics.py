import torch
import torch as th

gamma0 = 100

def compute_loss(model, points):
    X = points.clone().requires_grad_(True)
    x = X[:, 0]
    a = X[:, 1]
    v = X[:, 2]
    k = X[:, 3]
    y = model(X)
    alpha0 = 10 ** (a / 2)
    beta0 = alpha0 / 2 / (1 + v)
    L = 1
    alpha = alpha0 * th.exp(k * L * x)
    beta = beta0 * th.exp(k * L * x)
    gamma = gamma0 * th.exp(-k * L * x)
    l2 = x - 1
    (theta, phi, Ttheta, Tphi, lx, ly, lz, l1) = (y[:, 0], y[:, 1], y[:, 2], y[:, 3], y[:, 4], y[:, 5], y[:, 6], y[:, 7])
    theta = theta.squeeze()
    phi = phi.squeeze()
    Ttheta = Ttheta.squeeze()
    Tphi = Tphi.squeeze()
    lx = lx.squeeze()
    ly = ly.squeeze()
    lz = lz.squeeze()
    l1 = l1.squeeze()
    tx = th.sin(Ttheta) * th.cos(Tphi)
    ty = th.sin(Ttheta) * th.sin(Tphi)
    tz = th.cos(Ttheta)
    tzmean = th.mean(tz - 1e-06)
    dtheta = torch.autograd.grad(theta, X, grad_outputs=torch.ones_like(theta), create_graph=True, retain_graph=True)[0][:, 0]
    dphi = torch.autograd.grad(phi, X, grad_outputs=torch.ones_like(phi), create_graph=True, retain_graph=True)[0][:, 0]
    dTtheta = torch.autograd.grad(Ttheta, X, grad_outputs=torch.ones_like(Ttheta), create_graph=True, retain_graph=True)[0][:, 0]
    dTphi = torch.autograd.grad(Tphi, X, grad_outputs=torch.ones_like(Tphi), create_graph=True, retain_graph=True)[0][:, 0]
    dlx = torch.autograd.grad(lx, X, grad_outputs=torch.ones_like(lx), create_graph=True, retain_graph=True)[0][:, 0]
    dly = torch.autograd.grad(ly, X, grad_outputs=torch.ones_like(ly), create_graph=True, retain_graph=True)[0][:, 0]
    dlz = torch.autograd.grad(lz, X, grad_outputs=torch.ones_like(lz), create_graph=True, retain_graph=True)[0][:, 0]
    dl1 = torch.autograd.grad(l1, X, grad_outputs=torch.ones_like(l1), create_graph=True, retain_graph=True)[0][:, 0]
    dtx = th.cos(Ttheta) * th.cos(Tphi) * dTtheta - th.sin(Ttheta) * th.sin(Tphi) * dTphi
    dty = th.cos(Ttheta) * th.sin(Tphi) * dTtheta + th.sin(Ttheta) * th.cos(Tphi) * dTphi
    dtz = -th.sin(Ttheta) * dTtheta
    kappa = (lx * th.sin(theta) * th.cos(phi) + ly * th.sin(theta) * th.sin(phi) + lz * th.cos(theta)) / 2 / alpha + gamma
    tau = -l1 / 2 / beta
    kt = [th.cos(theta) * th.cos(phi), th.cos(theta) * th.sin(phi), -th.sin(theta)]
    kf = [-th.sin(theta) * th.sin(phi), th.sin(theta) * th.cos(phi), 0]
    ppx = (th.cos(phi - 2 * theta) - th.cos(phi + 2 * theta)) / 4
    ppy = (th.sin(phi - 2 * theta) - th.sin(phi + 2 * theta)) / 4
    PX = ppy * dphi - th.sin(phi) * dtheta
    PY = -ppx * dphi + th.cos(phi) * dtheta
    PZ = th.sin(theta) ** 2 * dphi
    Tdt = -tx * th.sin(phi) + ty * th.cos(phi)
    Tdf = tz
    eq1 = (-lx * kappa * kt[0] - ly * kappa * kt[1] - lz * kappa * kt[2] + dl1 * Tdt) / gamma
    eq2 = (-lx * kappa * kf[0] - ly * kappa * kf[1] - lz * kappa * kf[2] + dl1 * Tdf) / gamma
    eq3 = (dtx - kappa * th.sin(theta) * th.cos(phi)) / gamma
    eq4 = (dty - kappa * th.sin(theta) * th.sin(phi)) / gamma
    eq5 = (dtz - kappa * th.cos(theta)) / gamma
    eq6 = (dlx + l1 * PX) / gamma
    eq7 = (dly + l1 * PY) / gamma
    eq8 = (dlz + l1 * PZ + l2) / gamma
    eq9 = (tau - tx * PX - ty * PY - tz * PZ) / gamma
    eeq1 = eq1 ** 2
    eeq2 = eq2 ** 2
    eeq3 = eq3 ** 2
    eeq4 = eq4 ** 2
    eeq5 = eq5 ** 2
    eeq6 = eq6 ** 2
    eeq7 = eq7 ** 2
    eeq8 = eq8 ** 2
    eeq9 = eq9 ** 2
    meeq1 = th.mean(eeq1)
    meeq2 = th.mean(eeq2)
    meeq3 = th.mean(eeq3)
    meeq4 = th.mean(eeq4)
    meeq5 = th.mean(eeq5)
    meeq6 = th.mean(eeq6)
    meeq7 = th.mean(eeq7)
    meeq8 = th.mean(eeq8)
    meeq9 = th.mean(eeq9)
    eqs = th.tensor([meeq1, meeq2, meeq3, meeq4, meeq5, meeq6, meeq7, meeq8, meeq9])
    Ssigma1 = 1
    Ssigma2 = 1
    Ssigma3 = 1
    Ssigma4 = 1
    Ssigma5 = 1
    Ssigma6 = 1
    Ssigma7 = 1
    Ssigma8 = 1
    Ssigma9 = 1
    LL = th.mean(eqs) + th.std(eqs) + th.max(eqs)
    Asigma1 = model.sigma1 ** 2 + 1e-09
    Asigma2 = model.sigma2 ** 2 + 1e-09
    Asigma3 = model.sigma3 ** 2 + 1e-09
    Asigma4 = model.sigma4 ** 2 + 1e-09
    Asigma5 = model.sigma5 ** 2 + 1e-09
    Asigma6 = model.sigma6 ** 2 + 1e-09
    Asigma7 = model.sigma7 ** 2 + 1e-09
    Asigma8 = model.sigma8 ** 2 + 1e-09
    Asigma9 = model.sigma9 ** 2 + 1e-09
    loss1 = meeq1 * Ssigma1 / Asigma1 + th.log(Asigma1)
    loss2 = meeq2 * Ssigma2 / Asigma2 + th.log(Asigma2)
    loss3 = meeq3 * Ssigma3 / Asigma3 + th.log(Asigma3)
    loss4 = meeq4 * Ssigma4 / Asigma4 + th.log(Asigma4)
    loss5 = meeq5 * Ssigma5 / Asigma5 + th.log(Asigma5)
    loss6 = meeq6 * Ssigma6 / Asigma6 + th.log(Asigma6)
    loss7 = meeq7 * Ssigma7 / Asigma7 + th.log(Asigma7)
    loss8 = meeq8 * Ssigma8 / Asigma8 + th.log(Asigma8)
    loss9 = meeq9 * Ssigma9 / Asigma9 + th.log(Asigma9)
    loss = loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8 + loss9
    return (torch.mean(loss), LL, tzmean)
