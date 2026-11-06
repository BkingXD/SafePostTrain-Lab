"""Small, explicit causal Transformer. No external weights or datasets."""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


TOKENS = ["<pad>", "<bos>", "<user>", "<assistant>", "<eos>", "repeat", "copy", "please", "answer", *list("ABCDEFGH")]
TOKEN_TO_ID = {token: i for i, token in enumerate(TOKENS)}
PAD, BOS, USER, ASSISTANT, EOS = range(5)
SYMBOLS = list("ABCDEFGH")


@dataclass
class ModelConfig:
    vocab_size: int = len(TOKENS)
    width: int = 32
    heads: int = 4
    layers: int = 2
    max_length: int = 32


class CausalBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.norm1 = nn.LayerNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.proj = nn.Linear(config.width, config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.ff = nn.Sequential(nn.Linear(config.width, config.width * 2), nn.GELU(), nn.Linear(config.width * 2, config.width))

    def forward(self, x, padding):
        batch, length, width = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(batch, length, 3, self.heads, width // self.heads).permute(2, 0, 3, 1, 4)
        scores = q @ k.transpose(-2, -1) / math.sqrt(width // self.heads)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future, float("-inf"))
        scores = scores.masked_fill(padding[:, None, None, :], float("-inf"))
        attention = scores.softmax(dim=-1)
        attended = (attention @ v).transpose(1, 2).reshape(batch, length, width)
        x = x + self.proj(attended)
        return x + self.ff(self.norm2(x))


class TinyLM(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or ModelConfig()
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.width)
        self.position = nn.Embedding(self.config.max_length, self.config.width)
        self.blocks = nn.ModuleList([CausalBlock(self.config) for _ in range(self.config.layers)])
        self.norm = nn.LayerNorm(self.config.width)
        self.lm_head = nn.Linear(self.config.width, self.config.vocab_size)

    def features(self, tokens):
        if tokens.ndim != 2 or tokens.size(1) > self.config.max_length:
            raise ValueError("expected [batch, time] within the model context window")
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.embedding(tokens) + self.position(positions)
        for block in self.blocks:
            x = block(x, tokens.eq(PAD))
        return self.norm(x)

    def forward(self, tokens):
        return self.lm_head(self.features(tokens))


def model_device(model):
    """The model's parameters, rather than a global default, own placement."""
    return next(model.parameters()).device


class LoRALinear(nn.Module):
    def __init__(self, base, rank=4, alpha=4):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        factory = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.a = nn.Parameter(torch.randn(rank, base.in_features, **factory) * .02)
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, **factory))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def apply_lora(model, rank=4):
    """Freeze base, then adapt all Linear layers; embeddings/norms stay frozen."""
    model.requires_grad_(False)
    def replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                setattr(module, name, LoRALinear(child, rank, rank))
            else:
                replace(child)
    replace(model)
    return model


class ScoreModel(nn.Module):
    """A learned scalar head on causal hidden states, not oracle input features."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.backbone.lm_head.requires_grad_(False)  # Unused by the scalar head.
        parameter = next(backbone.parameters())
        self.head = nn.Linear(backbone.config.width, 1, device=parameter.device, dtype=parameter.dtype)

    def token_values(self, tokens):
        return self.head(self.backbone.features(tokens)).squeeze(-1)

    def forward(self, tokens):
        end = tokens.ne(PAD).sum(dim=1) - 1
        return self.token_values(tokens)[torch.arange(len(tokens), device=tokens.device), end]


def encode(tokens):
    return [TOKEN_TO_ID[t] for t in tokens]


def collate(records, responses=None, device=None):
    """Shift ONCE: logits at t predict token t+1; only assistant labels count."""
    if not records:
        raise ValueError("empty batch")
    rows, labels = [], []
    for i, row in enumerate(records):
        answer = responses[i] if responses is not None else row["response"]
        prefix, response = encode(row["prompt"]), encode(answer)
        if not response:
            raise ValueError("an answer needs at least one supervised token")
        rows.append(prefix + response)
        labels.append([-100] * len(prefix) + response)
    length = max(map(len, rows))
    ids = torch.tensor([r + [PAD] * (length - len(r)) for r in rows], dtype=torch.long, device=device)
    targets = torch.tensor([r + [-100] * (length - len(r)) for r in labels], dtype=torch.long, device=device)
    return ids, targets


def token_statistics(model, ids, labels):
    logits = model(ids[:, :-1])
    log_distribution = F.log_softmax(logits, dim=-1)
    targets = labels[:, 1:]
    mask = targets.ne(-100)
    selected = log_distribution.gather(-1, targets.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    selected = selected * mask
    lengths = mask.sum(dim=1)
    if (lengths == 0).any():
        raise ValueError("empty assistant supervision")
    return {"logps": selected, "mask": mask, "lengths": lengths,
            "sums": selected.sum(dim=1), "log_distribution": log_distribution}


def sft_loss(model, ids, labels):
    stats = token_statistics(model, ids, labels)
    return -stats["logps"].sum() / stats["mask"].sum()


@torch.no_grad()
def generate(model, prompt, sample=False, max_new_tokens=4, temperature=1., generator=None):
    if not math.isfinite(temperature) or temperature <= 0 or max_new_tokens < 1:
        raise ValueError("temperature must be positive and generation budget nonzero")
    if len(prompt) + max_new_tokens - 1 > model.config.max_length:
        raise ValueError("generation budget exceeds the context window")
    device = model_device(model)
    if sample and generator is not None and torch.device(generator.device) != device:
        raise ValueError("sampling generator and model must use the same device")
    model.eval()
    ids = encode(prompt)
    generated = []
    for _ in range(max_new_tokens):
        logits = model(torch.tensor([ids], dtype=torch.long, device=device))[0, -1] / temperature
        token = torch.multinomial(logits.softmax(-1), 1, generator=generator).item() if sample else logits.argmax().item()
        generated.append(TOKENS[token])
        ids.append(token)
        if token == EOS:
            break
    return generated


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def checkpoint(path, model, optimizer=None, step=0, extra=None):
    state = {"config": asdict(model.config), "model": model.state_dict(), "step": step,
             "torch_rng": torch.get_rng_state(), "extra": extra or {}}
    device = model_device(model)
    if device.type == "cuda":
        state["torch_cuda_rng"] = {"device": str(device), "state": torch.cuda.get_rng_state(device)}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(path, device="cpu", restore_rng=False):
    """Load on the requested device; optionally restore RNG after construction.

    Optimizer restoration remains explicit at the call site. A saved CUDA RNG
    is restored to the requested CUDA device, allowing device-index remapping.
    Explicit rollout Generators and PPO critic state are not in this checkpoint.
    """
    state = torch.load(path, map_location="cpu", weights_only=True)
    model = TinyLM(ModelConfig(**state["config"]))
    if state["extra"].get("lora_rank"):
        apply_lora(model, state["extra"]["lora_rank"])
    model.load_state_dict(state["model"])
    model.to(device)
    if restore_rng:
        torch.set_rng_state(state["torch_rng"])
        if model_device(model).type == "cuda" and "torch_cuda_rng" in state:
            torch.cuda.set_rng_state(state["torch_cuda_rng"]["state"], model_device(model))
    return model, state
