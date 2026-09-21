import torch
import torch.nn as nn
import torch.nn.functional as F

class OT_CGSF(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=64):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Conv2d(in_channels * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, 1),
            nn.Sigmoid() 
        )

    def optimal_transport_alignment(self, source, target, epsilon=0.1, n_iter=50, ot_size=32):
        B, C, H, W = source.shape
        src_small = F.interpolate(source, size=(ot_size, ot_size), mode='area')
        tgt_small = F.interpolate(target, size=(ot_size, ot_size), mode='area')
        
        src_flat, tgt_flat = src_small.view(B, C, -1), tgt_small.view(B, C, -1)   
        src_mean, tgt_mean = src_flat.mean(dim=1), tgt_flat.mean(dim=1)    
        
        cost = (src_mean.unsqueeze(2) - tgt_mean.unsqueeze(1)) ** 2  
        mu = torch.ones(B, src_mean.size(1), device=source.device) / src_mean.size(1)
        nu = torch.ones(B, tgt_mean.size(1), device=source.device) / tgt_mean.size(1)

        K = torch.exp(-cost / epsilon)
        u, v = torch.ones_like(mu), torch.ones_like(nu)
        for _ in range(n_iter):
            u = mu / (K @ v.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10)
            v = nu / (K.transpose(1, 2) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-10)

        T = u.unsqueeze(-1) * K * v.unsqueeze(1)  
        aligned_src_small = torch.bmm(T, src_flat.permute(0, 2, 1)).permute(0, 2, 1).view(B, C, ot_size, ot_size)
        return F.interpolate(aligned_src_small, size=(H, W), mode='bilinear', align_corners=False)

    def forward(self, modality_a, modality_b):
        aligned_a = self.optimal_transport_alignment(modality_a, modality_b)
        score = self.score_net(torch.cat([aligned_a, modality_b], dim=1)) 
        return score * aligned_a + (1 - score) * modality_b, aligned_a

class OT_CGSF_Classifier(nn.Module):
    def __init__(self, fusion_model):
        super().__init__()
        self.fusion = fusion_model
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(3, 2)
        
    def forward(self, a, b):
        fused_img, aligned_a = self.fusion(a, b)
        logits = self.classifier(self.pool(fused_img).view(a.size(0), -1))
        return logits, aligned_a

class ResilientSinkhornFusion(nn.Module):
    def __init__(self, num_classes=2, epsilon=1e-8, max_iter=50, stop_thresh=1e-9):
        super().__init__()
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.stop_thresh = stop_thresh

    def sinkhorn_knopp_log_domain(self, cost_matrix):
        B, N, M = cost_matrix.shape
        device = cost_matrix.device
        log_mu = torch.log(torch.ones(B, N, device=device) / N)
        log_nu = torch.log(torch.ones(B, M, device=device) / M)
        C = cost_matrix / self.epsilon
        log_u, log_v = torch.zeros(B, N, device=device), torch.zeros(B, M, device=device)

        for _ in range(self.max_iter):
            log_u_prev = log_u.clone()
            log_u = log_mu - torch.logsumexp(-C + log_v.unsqueeze(1), dim=2)
            log_v = log_nu - torch.logsumexp(-C.transpose(1, 2) + log_u.unsqueeze(1), dim=2)
            if torch.max(torch.abs(torch.exp(log_u) - torch.exp(log_u_prev))) < self.stop_thresh:
                break
        return torch.exp(log_u.unsqueeze(-1) - C + log_v.unsqueeze(1))

    def forward(self, logits_a, logits_b):
        p_a, p_b = F.softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1)
        cost = (p_a.unsqueeze(2) - p_b.unsqueeze(1)) ** 2  
        T = self.sinkhorn_knopp_log_domain(cost)  
        aligned_p_a = torch.bmm(T, p_a.unsqueeze(-1)).squeeze(-1)  

        ent_a = -(p_a * torch.log(p_a + 1e-10)).sum(dim=-1, keepdim=True)  
        ent_b = -(p_b * torch.log(p_b + 1e-10)).sum(dim=-1, keepdim=True)  
        w_a = torch.exp(-ent_a) / (torch.exp(-ent_a) + torch.exp(-ent_b))
        return w_a * aligned_p_a + (1 - w_a) * p_b, T

class DDEF(nn.Module):
    def __init__(self, num_classes=2, feature_dim=512):
        super().__init__()
        self.num_classes = num_classes
        self.encoder_a = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.encoder_b = nn.Sequential(nn.Linear(feature_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.evidential_head_a = nn.Linear(128, num_classes)
        self.evidential_head_b = nn.Linear(128, num_classes)
        self.bba_weight = nn.Parameter(torch.tensor(0.5))

    def dirichlet_to_bba(self, alpha):
        S = alpha.sum(dim=-1, keepdim=True)  
        return (alpha - 1) / S, self.num_classes / S

    def dempster_rule(self, b1, u1, b2, u2):
        match_belief = (b1 * b2).sum(dim=-1, keepdim=True)  
        conflict = ((1 - u1) * (1 - u2) - match_belief).clamp(min=1e-10, max=1 - 1e-10)
        b_combined = (b1 * b2 + b1 * u2 + b2 * u1) / (1 - conflict)
        return b_combined, (u1 * u2) / (1 - conflict)

    def forward(self, feat_a, feat_b):
        z_a, z_b = self.encoder_a(feat_a), self.encoder_b(feat_b)
        alpha_a, alpha_b = F.softplus(self.evidential_head_a(z_a)) + 1.0, F.softplus(self.evidential_head_b(z_b)) + 1.0
        
        b_a, u_a = self.dirichlet_to_bba(alpha_a)
        b_b, u_b = self.dirichlet_to_bba(alpha_b)
        
        w = torch.sigmoid(self.bba_weight)
        b_fused_bba = w * b_a + (1 - w) * b_b
        u_fused_bba = w * u_a + (1 - w) * u_b
        
        b_final, u_final = self.dempster_rule(b_a, u_a, b_b, u_b)
        b_out, u_out = 0.5 * b_fused_bba + 0.5 * b_final, 0.5 * u_fused_bba + 0.5 * u_final
        return b_out + u_out / self.num_classes, u_out