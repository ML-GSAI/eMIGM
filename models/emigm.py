from functools import partial

import numpy as np
from tqdm import tqdm
import scipy.stats as stats
import math
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from timm.models.vision_transformer import Block

from models.diffloss import DiffLoss

class eMIGM(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=256, vae_stride=16, patch_size=1,
                 encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
                 decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 vae_embed_dim=16,
                 label_drop_prob=0.1,
                 use_unsup_cfg=False,
                 cfg_t_min=0.0,
                 cfg_t_max=1.0,
                 class_num=1000,
                 attn_dropout=0.1,
                 proj_dropout=0.1,
                 buffer_size=64,
                 diffloss_d=3,
                 diffloss_w=1024,
                 mask_strategy='cosine', # 'cosine' or 'linear' or 'poly-0.5'
                 use_weighted_loss=False,
                 use_mae=True,
                 clamp_t_min=0,
                 num_sampling_steps='100',
                 sampling_algo='ddpm',
                 diffusion_batch_mul=4,
                 grad_checkpointing=False,
                 ):
        super().__init__()

        # --------------------------------------------------------------------------
        # VAE and patchify specifics
        self.vae_embed_dim = vae_embed_dim

        self.img_size = img_size
        self.vae_stride = vae_stride
        self.patch_size = patch_size
        self.seq_h = self.seq_w = img_size // vae_stride // patch_size
        self.seq_len = self.seq_h * self.seq_w
        self.token_embed_dim = vae_embed_dim * patch_size**2
        self.grad_checkpointing = grad_checkpointing

        # --------------------------------------------------------------------------
        # Class Embedding
        self.num_classes = class_num
        self.class_emb = nn.Embedding(class_num, encoder_embed_dim)
        self.label_drop_prob = label_drop_prob
        # Fake class embedding for CFG's unconditional generation
        self.fake_latent = nn.Parameter(torch.zeros(1, encoder_embed_dim))
        self.use_unsup_cfg = use_unsup_cfg

        self.cfg_t_min = cfg_t_min
        self.cfg_t_max = cfg_t_max

        # --------------------------------------------------------------------------
        # eMIGM mask strategy
        self.mask_strategy = mask_strategy
        self.use_weighted_loss = use_weighted_loss
        self.use_mae = use_mae
        self.clamp_t_min = clamp_t_min

        # --------------------------------------------------------------------------
        # eMIGM encoder specifics
        self.z_proj = nn.Linear(self.token_embed_dim, encoder_embed_dim, bias=True)
        self.z_proj_ln = nn.LayerNorm(encoder_embed_dim, eps=1e-6)
        self.buffer_size = buffer_size
        self.encoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, encoder_embed_dim))

        self.encoder_blocks = nn.ModuleList([
            Block(encoder_embed_dim, encoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(encoder_depth)])
        self.encoder_norm = norm_layer(encoder_embed_dim)

        # --------------------------------------------------------------------------
        # eMIGM decoder specifics
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=True)
        if self.use_unsup_cfg:
            assert encoder_embed_dim == decoder_embed_dim
            self.mask_token = self.fake_latent
        else:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len + self.buffer_size, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                  proj_drop=proj_dropout, attn_drop=attn_dropout) for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.diffusion_pos_embed_learned = nn.Parameter(torch.zeros(1, self.seq_len, decoder_embed_dim))

        self.initialize_weights()

        # --------------------------------------------------------------------------
        # Diffusion Loss
        self.diffloss = DiffLoss(
            target_channels=self.token_embed_dim,
            z_channels=decoder_embed_dim,
            width=diffloss_w,
            depth=diffloss_d,
            num_sampling_steps=num_sampling_steps,
            sampling_algo=sampling_algo,
            grad_checkpointing=grad_checkpointing
        )
        self.sampling_algo = sampling_algo
        self.diffusion_batch_mul = diffusion_batch_mul

    def initialize_weights(self):
        # parameters
        torch.nn.init.normal_(self.class_emb.weight, std=.02)
        torch.nn.init.normal_(self.fake_latent, std=.02)
        if not self.use_unsup_cfg:
            torch.nn.init.normal_(self.mask_token, std=.02)
        torch.nn.init.normal_(self.encoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed_learned, std=.02)
        torch.nn.init.normal_(self.diffusion_pos_embed_learned, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def patchify(self, x):
        bsz, c, h, w = x.shape
        p = self.patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x):
        bsz = x.shape[0]
        p = self.patch_size
        c = self.vae_embed_dim
        h_, w_ = self.seq_h, self.seq_w

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]
        
    def random_masking(self, x):
        bsz, seq_len, embed_dim = x.shape
        
        eps = 1e-3
        valid_mask_generated = False  # Flag to indicate whether a valid mask has been generated

        while not valid_mask_generated:
            t = torch.rand((), device=x.device)
            t = torch.clamp(t, min=self.clamp_t_min)
            p_mask = (1 - eps) * t + eps  # Shared p_mask scalar
            p_mask = p_mask.expand(seq_len)  # Shape is (seq_len,)

            # Generate random_vals based on sequence length
            random_vals = torch.rand((seq_len,), device=x.device)  # Shape is (seq_len,)

            # Select masking strategy
            if self.mask_strategy == 'cosine':  # 'cosine' or 'linear' or 'poly-0.5'
                mask_prob = torch.cos(math.pi / 2 * (1 - p_mask))
            elif self.mask_strategy == 'linear': 
                mask_prob = p_mask
            elif self.mask_strategy == 'poly-0.5':
                mask_prob = p_mask ** 0.5
            elif self.mask_strategy == 'exp':
                mask_prob = 1 - torch.exp(-5 * p_mask)
            elif self.mask_strategy == 'log':
                mask_prob = torch.log(1 + (math.exp(5) - 1) * p_mask) / 5

            # Generate mask
            mask = (random_vals < mask_prob).float()  # Shape is (seq_len,)

            # Check if the number of masks meets the requirement
            if mask.sum() >= 2:
                valid_mask_generated = True

        # Expand the mask to match the batch_size
        mask = mask.expand(bsz, -1)  # Shape is (batch_size, seq_len)
        p_mask = p_mask.expand(bsz, -1)  # Shape is (batch_size, seq_len) 

        random_mask = mask.clone()
        # Shuffle the seq_len order of each sample
        for i in range(bsz):
            random_indices = torch.randperm(seq_len)
            random_mask[i] = random_mask[i][random_indices]
        mask = random_mask

        return mask, p_mask, self.mask_strategy

    def random_masking_batch(self, x):
        bsz, seq_len, embed_dim = x.shape
        t = torch.rand((bsz,), device=x.device)
        eps = 1e-3
        p_mask = (1 - eps) * t + eps  # Shape is (batch_size,)
        p_mask = p_mask[:, None].expand(-1, seq_len)  # Shape is (batch_size, seq_len)
        random_vals = torch.rand((bsz, seq_len), device=x.device)
        # Select masking strategy
        if self.mask_strategy == 'cosine':  # 'cosine' or 'linear' or 'poly-0.5'
            mask_prob = torch.cos(math.pi / 2 * (1 - p_mask))
        elif self.mask_strategy == 'linear': 
            mask_prob = p_mask
        elif self.mask_strategy == 'poly-0.5':
            mask_prob = p_mask ** 0.5
        elif self.mask_strategy == 'exp':
            mask_prob = 1 - torch.exp(-5 * p_mask)
        elif self.mask_strategy == 'log':
            mask_prob = torch.log(1 + (math.exp(5) - 1) * p_mask) / 5
        mask = (random_vals < mask_prob).float()
        
        return mask, p_mask, self.mask_strategy

    def forward_mae_encoder(self, x, mask, class_embedding):
        x = self.z_proj(x)
        bsz, seq_len, embed_dim = x.shape

        # concat buffer
        x = torch.cat([torch.zeros(bsz, self.buffer_size, embed_dim, device=x.device), x], dim=1)
        mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

        # random drop class embedding during training
        if self.training:
            drop_latent_mask = torch.rand(bsz) < self.label_drop_prob
            drop_latent_mask = drop_latent_mask.unsqueeze(-1).cuda().to(x.dtype)
            class_embedding = drop_latent_mask * self.fake_latent + (1 - drop_latent_mask) * class_embedding

        x[:, :self.buffer_size] = class_embedding.unsqueeze(1)

        # encoder position embedding
        x = x + self.encoder_pos_embed_learned
        x = self.z_proj_ln(x)

        if self.use_mae:
            # dropping
            x = x[(1-mask_with_buffer).nonzero(as_tuple=True)].reshape(bsz, -1, embed_dim)
        else: 
            # Create mask tokens
            if self.use_unsup_cfg:
                mask_tokens = self.mask_token.unsqueeze(0).repeat(bsz, mask_with_buffer.shape[1], 1).to(x.dtype)
            else:
                mask_tokens = self.mask_token.repeat(bsz, mask_with_buffer.shape[1], 1).to(x.dtype)

            # Apply mask tokens to masked positions
            x_with_mask = mask_tokens.clone()
            x_with_mask[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x[(1 - mask_with_buffer).nonzero(as_tuple=True)]
            x = x_with_mask

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.encoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.encoder_blocks:
                x = block(x)
        x = self.encoder_norm(x)

        return x

    def forward_mae_decoder(self, x, mask):

        x = self.decoder_embed(x)
        if self.use_mae:
            mask_with_buffer = torch.cat([torch.zeros(x.size(0), self.buffer_size, device=x.device), mask], dim=1)

            # pad mask tokens
            if self.use_unsup_cfg:
                mask_tokens = self.mask_token.unsqueeze(0).repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            else:
                mask_tokens = self.mask_token.repeat(mask_with_buffer.shape[0], mask_with_buffer.shape[1], 1).to(x.dtype)
            x_after_pad = mask_tokens.clone()
            x_after_pad[(1 - mask_with_buffer).nonzero(as_tuple=True)] = x.reshape(x.shape[0] * x.shape[1], x.shape[2])
            x = x_after_pad

        # decoder position embedding
        x = x + self.decoder_pos_embed_learned

        # apply Transformer blocks
        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.decoder_blocks:
                x = checkpoint(block, x)
        else:
            for block in self.decoder_blocks:
                x = block(x)
        x = self.decoder_norm(x)

        x = x[:, self.buffer_size:]
        x = x + self.diffusion_pos_embed_learned
        return x

    def forward_loss(self, z, target, mask, p_mask=None, mask_strategy=None):
        bsz, seq_len, _ = target.shape
        target = target.reshape(bsz * seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(bsz*seq_len, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
        if p_mask is not None:
            p_mask = p_mask.reshape(bsz*seq_len).repeat(self.diffusion_batch_mul)
            loss = self.diffloss(z=z, target=target, mask=mask, p_mask=p_mask, mask_strategy=mask_strategy, use_weighted_loss=self.use_weighted_loss)
        else: 
            loss = self.diffloss(z=z, target=target, mask=mask)
        return loss
    
    def forward(self, imgs, labels):

        # class embed
        class_embedding = self.class_emb(labels)

        # patchify and mask (drop) tokens
        x = self.patchify(imgs)
        gt_latents = x.clone().detach()
        if not self.use_mae and self.use_weighted_loss: 
            mask, p_mask, mask_strategy = self.random_masking_batch(x)
        else:
            mask, p_mask, mask_strategy = self.random_masking(x)

        # mae encoder
        x = self.forward_mae_encoder(x, mask, class_embedding)

        # mae decoder
        z = self.forward_mae_decoder(x, mask)

        # diffloss
        loss = self.forward_loss(z=z, target=gt_latents, mask=mask, p_mask=p_mask, mask_strategy=mask_strategy)

        return loss

    def sample_tokens(self, bsz, num_iter=64, cfg=1.0, cfg_schedule="constant", labels=None, temperature=1.0, progress=False, eps=1e-5):

        # init and sample generation orders
        mask = torch.ones(bsz, self.seq_len).cuda().bool()
        tokens = torch.zeros(bsz, self.seq_len, self.token_embed_dim).cuda()

        indices = list(range(num_iter))
        if progress:
            indices = tqdm(indices)
        
        # Generate timesteps
        timesteps = torch.linspace(1.0, eps, num_iter + 1, device='cuda')
        
        # generate latents
        for step in indices:
            t = timesteps[step]
            s = timesteps[step + 1]
            if step < num_iter - 1:
                if self.mask_strategy == 'cosine':
                    p_transfer = 1 - torch.cos(math.pi / 2 * (1 - s)) / torch.cos(math.pi / 2 * (1 - t))
                elif self.mask_strategy == 'linear':
                    p_transfer = 1 - s / t
                elif self.mask_strategy == 'poly-0.5':
                    p_transfer = 1 - s ** 0.5 / t ** 0.5
                elif self.mask_strategy == 'exp':
                    p_transfer = 1 - (1 - torch.exp(-5 * s))/(1 - torch.exp(-5 * t))
                elif self.mask_strategy == 'log':
                    p_transfer = 1 - (torch.log(1 + (math.exp(5) - 1) * s) / torch.log(1 + (math.exp(5) - 1) * t))
                else:
                    raise NotImplementedError
                
            else:
                p_transfer = 1.0

            cur_tokens = tokens.clone()

            # class embedding and CFG
            if labels is not None:
                class_embedding = self.class_emb(labels)
            else:
                class_embedding = self.fake_latent.repeat(bsz, 1)
            if (not cfg == 1.0) and (s <= self.cfg_t_max and s > self.cfg_t_min):
                tokens_cfg = torch.cat([tokens, tokens], dim=0)
                class_embedding_cfg = torch.cat([class_embedding, self.fake_latent.repeat(bsz, 1)], dim=0)
                mask_cfg = torch.cat([mask.float(), mask.float()], dim=0)
            else:
                tokens_cfg = tokens
                class_embedding_cfg = class_embedding
                mask_cfg = mask.float()

            
            # MAE encoder
            x = self.forward_mae_encoder(tokens_cfg, mask_cfg, class_embedding_cfg)
            # MAE decoder
            z = self.forward_mae_decoder(x, mask_cfg)

            # Update the mask, mask transfer according to probability
            random_vals = torch.rand(self.seq_len).unsqueeze(0).expand(bsz, -1).cuda()

            # Shuffle random_vals for each sample individually
            shuffled_vals = random_vals.clone()
            for i in range(1, bsz):
                perm = torch.randperm(self.seq_len).cuda()
                shuffled_vals[i] = random_vals[i, perm]

            # Get the first row of random_vals and mask
            first_row_vals = shuffled_vals[0]
            for i in range(1, bsz):
                zero_indices = torch.where(mask[i] == 0)[0]
                one_indices = torch.where(mask[i] == 1)[0]

                zero_perm = zero_indices[torch.randperm(zero_indices.size(0))]
                one_perm = one_indices[torch.randperm(one_indices.size(0))]

                shuffled_vals[i][zero_perm] = first_row_vals[torch.where(mask[0] == 0)[0]]
                shuffled_vals[i][one_perm] = first_row_vals[torch.where(mask[0] == 1)[0]]

            random_vals = shuffled_vals

            unmask_prob = (mask == 1) & (random_vals < p_transfer)

            if unmask_prob.sum() == 0:
                for i in range(bsz):
                    one_indices = torch.nonzero(mask[i] == 1).squeeze()
                    if one_indices.numel() > 0:
                        if one_indices.ndim == 0:
                            unmask_prob[i, one_indices] = True
                        else:
                            random_idx = torch.randint(0, one_indices.numel(), (1,)).item()
                            unmask_prob[i, one_indices[random_idx]] = True

            mask_next = mask.clone()
            mask_next[unmask_prob] = 0  # Change from masked to unmasked

            # In the last step, make sure all mask positions are predicted
            if step >= num_iter - 1:
                mask_next = torch.zeros_like(mask)

            # Calculate the tokens that need to be predicted in this iteration
            mask_to_pred = mask & (~mask_next)
            if mask_to_pred.sum() == 0:
                print("No tokens to predict in this iteration, breaking the loop.")
                break
            
            mask = mask_next

            if (not cfg == 1.0) and (s <= self.cfg_t_max and s > self.cfg_t_min):
                mask_to_pred_cfg = torch.cat([mask_to_pred, mask_to_pred], dim=0)
            else:
                mask_to_pred_cfg = mask_to_pred
            
            
            # Get the features of the positions that need to be predicted
            indices_to_pred = mask_to_pred_cfg.nonzero(as_tuple=True)
            z_pred = z[indices_to_pred]

            # CFG scheduling
            if cfg_schedule == "constant" and (s <= self.cfg_t_max and s > self.cfg_t_min):
                cfg_iter = cfg
            elif (s <= self.cfg_t_max and s > self.cfg_t_min):
                cfg_t = (self.seq_len - mask.sum(dim=-1)[0]) / self.seq_len
                if cfg_schedule == "linear":
                    cfg_t = cfg_t
                elif cfg_schedule == 'power':
                    cfg_t = cfg_t ** 2
                elif cfg_schedule == 'cubic':
                    cfg_t = cfg_t ** 3
                elif cfg_schedule == 'exp':
                    cfg_t = 1 - torch.exp(-5 * cfg_t)
                elif cfg_schedule == 'cosine':
                    cfg_t = torch.cos(math.pi / 2 * (1 - cfg_t))
                else:
                    raise NotImplementedError
                cfg_iter = 1 + (cfg - 1) * cfg_t
            else: 
                cfg_iter = 1.0
            sampled_token_latent = self.diffloss.sample(z_pred, temperature, cfg_iter)
            if (not cfg == 1.0) and (s <= self.cfg_t_max and s > self.cfg_t_min):
                if self.sampling_algo == 'ddpm':
                    sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
                elif self.sampling_algo == 'dpm-solver':
                    pass
                # For DDPM and DPM-Solver, both need to process indices_to_pred
                # Split each tensor in the indices_to_pred list along the first dimension into two halves and take the first half
                indices_to_pred = [idx.chunk(2)[0] for idx in indices_to_pred]

            # Update tokens
            cur_tokens[indices_to_pred] = sampled_token_latent
            tokens = cur_tokens.clone()

        # unpatchify
        tokens = self.unpatchify(tokens)
        return tokens

def emigm_xsmall(**kwargs):
    model = eMIGM(
        encoder_embed_dim=448, encoder_depth=10, encoder_num_heads=8,
        decoder_embed_dim=448, decoder_depth=10, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def emigm_small(**kwargs):
    model = eMIGM(
        encoder_embed_dim=512, encoder_depth=12, encoder_num_heads=8,
        decoder_embed_dim=512, decoder_depth=12, decoder_num_heads=8,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def emigm_base(**kwargs):
    model = eMIGM(
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=768, decoder_depth=12, decoder_num_heads=12,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def emigm_large(**kwargs):
    model = eMIGM(
        encoder_embed_dim=1024, encoder_depth=16, encoder_num_heads=16,
        decoder_embed_dim=1024, decoder_depth=16, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def emigm_huge(**kwargs):
    model = eMIGM(
        encoder_embed_dim=1280, encoder_depth=20, encoder_num_heads=16,
        decoder_embed_dim=1280, decoder_depth=20, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model