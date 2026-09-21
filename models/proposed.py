import torch
import torch.nn as nn
import torch.nn.functional as F

class OT_Evidential_Fusion(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512, hidden_dim=64):
        super().__init__()
        self.num_classes, self.epsilon, self.sinkhorn_iter = num_classes, 0.1, 50
        
        self.encoder_endo = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        self.encoder_histo = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU())
        
        self.evidential_head_endo = nn.Linear(hidden_dim, num_classes)
        self.evidential_head_histo = nn.Linear(hidden_dim, num_classes)
        
        self.reliability_endo = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        self.reliability_histo = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def sinkhorn_ot(self, cost_matrix):
        B, N, M = cost_matrix.shape
        mu, nu = torch.ones(B, N, device=cost_matrix.device)/N, torch.ones(B, M, device=cost_matrix.device)/M
        K = torch.exp(-cost_matrix / self.epsilon)
        u, v = torch.ones(B, N, device=cost_matrix.device), torch.ones(B, M, device=cost_matrix.device)
        for _ in range(self.sinkhorn_iter):
            u = mu / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10)
            v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10)
        return u.unsqueeze(-1) * K * v.unsqueeze(1)

    def forward(self, feat_endo, feat_histo):
        z_endo, z_histo = self.encoder_endo(feat_endo), self.encoder_histo(feat_histo)
        
        cost = torch.cdist(z_endo.unsqueeze(-1), z_histo.unsqueeze(-1), p=2) ** 2  
        T = self.sinkhorn_ot(cost)  
        
        r_endo, r_histo = self.reliability_endo(z_endo), self.reliability_histo(z_histo)   
        w_endo, w_histo = r_endo / (r_endo + r_histo + 1e-10), r_histo / (r_endo + r_histo + 1e-10)

        alpha_endo = F.softplus(self.evidential_head_endo(z_endo)) + 1.0
        alpha_histo = F.softplus(self.evidential_head_histo(z_histo)) + 1.0
        
        alpha_endo_weighted = 1 + (alpha_endo - 1) * w_endo
        alpha_histo_weighted = 1 + (alpha_histo - 1) * w_histo

        def to_bba(alpha):
            S = alpha.sum(dim=-1, keepdim=True)
            return (alpha - 1) / S, self.num_classes / S

        b_e, u_e = to_bba(alpha_endo_weighted)
        b_h, u_h = to_bba(alpha_histo_weighted)

        match_belief = (b_e * b_h).sum(dim=-1, keepdim=True)
        conflict = ((1 - u_e) * (1 - u_h) - match_belief).clamp(min=1e-10, max=1 - 1e-10)

        b_combined = (b_e * b_h + b_e * u_h + b_h * u_e) / (1 - conflict)
        u_combined = (u_e * u_h) / (1 - conflict)

        return b_combined + u_combined / self.num_classes, T, w_endo, w_histo