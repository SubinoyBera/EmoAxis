import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    """
    Emotion-Aware Encoder:
    - Takes in tokenized text (from tokenizer)
    - Uses precomputed emotion label description embeddings + expressiveness weights
    - Produces an emotion-aware latent representation for the posterior network
    """

    def __init__(self, label_embedding_dict, base_encoder):
        super().__init__()
        self.encoder = base_encoder
        self.hidden_size = self.encoder.config.hidden_size
        self.label_embedding_dict = {
            k: v.clone().detach() for k, v in label_embedding_dict.items()
        }

        # Multi-head cross-attention block
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=6,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(self.hidden_size)


    def forward(self, inputs, emotion_labels, expressiveness):
        """
        inputs: tokenizer output dict (input_ids, attention_mask)
        emotion_labels: list[list[str]] — known emotion label descriptions
        expressiveness: list[list[float]] — expressiveness weights per emotion
        """
        device = next(self.parameters()).device

        outputs = self.encoder(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]                                                      # [B, T, H]

        atten_mask = inputs['attention_mask']                                                          # [B, T]
        # Mean Pooling text_embeddings
        mask = atten_mask.unsqueeze(-1).float()
        text_emb = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)                                                                  # [B, H]

        # Weighted emotion embedding for each sample
        emotion_emb = []
        for labels, weights in zip(emotion_labels, expressiveness):
            # stack embeddings of all emotion descriptions
            emb_list = [self.label_embedding_dict[lbl].to(device) for lbl in labels]
            emb_stack = torch.stack(emb_list, dim=0)                                                       # [num_labels, H]

            # normalize weights → weighted mean
            w = torch.tensor(weights, dtype=torch.float32, device=device).unsqueeze(1)
            w = w / w.sum()
            weighted_emb = (emb_stack * w).sum(dim=0)                                                      # [H]
            emotion_emb.append(weighted_emb)

        emotion_emb = torch.stack(emotion_emb, dim=0).unsqueeze(1)                                         # [B, 1, H]

        # cross-attention (query=text, key/value=emotion)
        attn_out, _ = self.cross_attn(
            query=last_hidden_state,                     # [B, T, H]   
            key=emotion_emb,                             # [B, 1, H]
            value=emotion_emb                            # [B, 1, H]
        )

        # Fuse and pool
        fused_hidden_state = last_hidden_state + attn_out
        fused_emo_text_emb = (fused_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        fused_emo_text_emb = self.layer_norm(fused_emo_text_emb)

        return text_emb, fused_emo_text_emb


class PosteriorNetwork(nn.Module):
    """
    Learns an emotion-aware posterior distribution q(z | x, e)
    over latent space using fused encoder output.
    """

    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.mu_posterior = nn.Linear(256, latent_dim)
        self.logvar_posterior = nn.Linear(256, latent_dim)

    def forward(self, fused_emo_text_emb):
        h = self.mlp(fused_emo_text_emb)
        mu_post = self.mu_posterior(h)
        logvar_post = torch.clamp(self.logvar_posterior(h), min=-10, max=10)

        std = torch.exp(0.5 * logvar_post)
        eps = torch.randn_like(std)
        z = mu_post + eps * std
        
        return z, mu_post, logvar_post
    

class PriorNetwork(nn.Module):
    """
    Learns a prior distribution p(z | x)
    based only on text (without emotion labels).
    """

    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256)
        )

        self.mu_prior = nn.Linear(256, latent_dim)
        self.logvar_prior = nn.Linear(256, latent_dim)

    def forward(self, text_emb):
        h = self.mlp(text_emb)
        mu_prior = self.mu_prior(h)
        logvar_prior = torch.clamp(self.logvar_prior(h), min=-10, max=10)
        
        return mu_prior, logvar_prior
    

class Classifier(nn.Module):
    """
    Shared classifier network for emotion classification
    """
    def __init__(self, latent_dim=128, num_classes=28):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.GELU(),
            nn.LayerNorm(64),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, z):
        return self.mlp(z)
    

class CVAE_EmotionModel(nn.Module):
    def __init__(self, encoder_block, posterior_net, prior_net, emotion_classifier):
        """
        Combines:
        - EncoderBlock   → produces text and fused(text+emotion) embeddings
        - PosteriorNet   → q(z|x,e)
        - PriorNet       → p(z|x)
        - EmotionClassifier → predicts emotions from latent z
        """
        super().__init__()
        self.encoder = encoder_block
        self.posterior = posterior_net
        self.prior = prior_net
        self.emotion_classifier = emotion_classifier

    def forward(self, input_ids, attention_mask, emotion_labels, expressiveness):
        """
        Returns a dictionary containing:
        - posterior z, mu, logvar
        - prior mu, logvar
        - emotion logits from posterior z
        - emotion logits from prior mu (for KL & MSE alignment)
        """

        encoder_outputs = self.encoder(
            inputs={"input_ids": input_ids, "attention_mask": attention_mask},
            emotion_labels=emotion_labels,
            expressiveness=expressiveness
        )

        text_emb = encoder_outputs["text_embedding"]                  # [B, H]
        fused_emb = encoder_outputs["fused_text_emo_embedding"]       # [B, H]

        z_post, mu_post, logvar_post = self.posterior(fused_emb)

        mu_prior, logvar_prior = self.prior(text_emb)

        logits_post = self.emotion_classifier(z_post)                # from sampled posterior
        logits_prior = self.emotion_classifier(mu_prior)             # from prior mean

        return {
            "z_post": z_post,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "mu_prior": mu_prior,
            "logvar_prior": logvar_prior,
            "logits_post": logits_post,
            "logits_prior": logits_prior
        }