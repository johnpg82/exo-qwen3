from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.qwen2_moe import Qwen2MoeDecoderLayer, ModelArgs

from ...shard import Shard
from .base import IdentityBlock

@dataclass
class ModelArgs(ModelArgs):
  shard: Shard = field(default_factory=lambda: Shard("", 0, 0, 0))

  def __post_init__(self):
    super().__post_init__()

    if isinstance(self.shard, Shard):
      return
    if not isinstance(self.shard, dict):
      raise TypeError(f"Expected shard to be a Shard instance or a dict, got {type(self.shard)} instead")

    self.shard = Shard(**self.shard)

class Qwen2MoeModel(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.vocab_size = args.vocab_size
    self.num_hidden_layers = args.num_hidden_layers
    assert self.vocab_size > 0

    if self.args.shard.is_first_layer() or (self.args.shard.is_last_layer() and args.tie_word_embeddings):
      self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)

    self.layers = []
    for i in range(self.num_hidden_layers):
      if self.args.shard.start_layer <= i <= self.args.shard.end_layer:
        self.layers.append(Qwen2MoeDecoderLayer(args=args))
      else:
        self.layers.append(IdentityBlock())

    if self.args.shard.is_last_layer():
      self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    if self.args.shard.is_first_layer():
      h = self.embed_tokens(inputs)
    else:
      h = inputs

    mask = None
    if h.shape[1] > 1:
      mask = create_attention_mask(h, cache)

    if cache is None:
      cache = [None]*len(self.layers)

    for layer, c in zip(self.layers, cache):
      h = layer(h, mask, c)

    if self.args.shard.is_last_layer():
      h = self.norm(h)
    return h


class Model(nn.Module):
  def __init__(self, args: ModelArgs):
    super().__init__()
    self.args = args
    self.model_type = args.model_type
    self.model = Qwen2MoeModel(args)
    if self.args.shard.is_last_layer():
      self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

  def __call__(
    self,
    inputs: mx.array,
    cache=None,
  ):
    out = self.model(inputs, cache)
    if self.args.shard.is_last_layer():
      out = self.lm_head(out)
    return out

  def sanitize(self, weights):
    shard_state_dict = {}

    for key, value in weights.items():
      if "self_attn.rotary_emb.inv_freq" in key:
        continue
      if key.startswith('model.layers.'):
        layer_num = int(key.split('.')[2])
        if self.args.shard.start_layer <= layer_num <= self.args.shard.end_layer:
          shard_state_dict[key] = value
      elif self.args.shard.is_first_layer() and key.startswith('model.embed_tokens'):
        shard_state_dict[key] = value
      elif (self.args.shard.is_last_layer() and self.args.tie_word_embeddings) and key.startswith('model.embed_tokens'):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and key.startswith('lm_head'):
        shard_state_dict[key] = value
      elif self.args.shard.is_last_layer() and (key.startswith('model.norm')):
        shard_state_dict[key] = value

    # Handle MoE expert weight reorganization - check if individual expert weights exist
    has_individual_experts = any(
      "mlp.experts." in key and (".up_proj.weight" in key or ".down_proj.weight" in key or ".gate_proj.weight" in key)
      for key in weights.keys()
    )
    
    if has_individual_experts:
      # Convert individual expert weights to stacked format
      for l in range(self.args.num_hidden_layers):
        if not (self.args.shard.start_layer <= l <= self.args.shard.end_layer):
          continue
        prefix = f"model.layers.{l}"
        for n in ["up_proj", "down_proj", "gate_proj"]:
          for k in ["weight", "scales", "biases"]:
            expert_weights = []
            for e in range(self.args.num_experts):
              expert_key = f"{prefix}.mlp.experts.{e}.{n}.{k}"
              if expert_key in weights:
                expert_weights.append(weights[expert_key])
                # Remove from shard_state_dict if it was already added
                shard_state_dict.pop(expert_key, None)
            
            if expert_weights:
              shard_state_dict[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(expert_weights)

    return shard_state_dict

  @property
  def layers(self):
    return self.model.layers

  @property
  def head_dim(self):
    return self.args.hidden_size // self.args.num_attention_heads

  @property
  def n_kv_heads(self):
    return self.args.num_key_value_heads