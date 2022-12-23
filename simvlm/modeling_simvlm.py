from typing import Dict, List, Optional, Set, Tuple, Union
import math
import torch
import torch.nn as  nn
import torch.nn.functional as F
from einops import rearrange
from transformers import activations, GenerationMixin
from transformers.modeling_outputs import BaseModelOutput
from .resblock import Conv2d, BottleneckBlock
from .prune_linear import prune_linear_layer, find_pruneable_heads_and_indices
from .configuration_simvlm import SimVLMConfig

class ResBlock(nn.Sequential):
    def __init__(
        self,
        kernel_size: int, 
        stride: int,
        in_channels: int = 3, 
        out_channels: int = 256, 
    ):
        super().__init__(
            Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=kernel_size,
                stride=stride,
                bias=True
            ),
            BottleneckBlock(
                in_channels=64,
                out_channels=256,
                bottleneck_channels=64
            ),
            BottleneckBlock(
                in_channels=256,
                out_channels=out_channels,
                bottleneck_channels=128
            )
        )

class SimVLMPatchEmbeddings(nn.Module):
    def __init__(
        self,
        config: SimVLMConfig
    ):
        if config.image_size % config.patch_size != 0:
            raise ValueError(
                f"The image size {config.image_size} is not multiple of "
                f"the patch size {config.patch_size}"
            )

        super().__init__()
        self.image_size = (config.image_size, config.image_size)
        self.patch_size = (
            config.patch_size if isinstance(config.patch_size, tuple) 
            else (config.patch_size, config.patch_size)
        )
        self.feat_map_size = [self.image_size[0] // self.patch_size[0] for _ in range(2)]
        self.num_patches = self.feat_map_size[0] * self.feat_map_size[1]
        self.hidden_size = config.hidden_size
        self.ResBlock = (
                            ResBlock(
                            in_channels=config.num_channels, 
                            out_channels=config.hidden_size,
                            kernel_size=config.patch_size,
                            stride=config.patch_size
                            )
                        )

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        feat_map = self.ResBlock(img)
        embeddings = rearrange(feat_map, 'b c h w -> b (h w) c')
        # embeddings += self.position_embeddings(feat_map)

        return embeddings

# 'text' in modeling_simvlm all refer to [token_ids] because tokenization is done in dataloader
class SimVLMTextEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size,
                                            padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

    def forward(
        self, 
        prefix_text: torch.Tensor, 
        tgt_text: Optional[torch.Tensor]=None,
        position_ids: Optional[torch.Tensor]=None
    ) -> torch.Tensor:

        prefix_text_embeds = self.word_embeddings(prefix_text)

        # Add position embeddings to prefix text embeddings, yet no need for target text embeddings
        if position_ids is None:
            position_ids = self.position_ids[:, :prefix_text_embeds.shape[1]]
        position_embeddings = self.position_embeddings(position_ids)
        prefix_text_embeds += position_embeddings
        prefix_text_embeds = self.dropout(self.layer_norm(prefix_text_embeds))

        if tgt_text is not None:
            tgt_text_embeds = self.word_embeddings(tgt_text)
            return prefix_text_embeds, tgt_text_embeds

        return prefix_text_embeds


class SimVLMEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config: SimVLMConfig):
        super().__init__()

        # text embeddings
        self.text_embeddings = SimVLMTextEmbeddings(config)
        # patch embeddings
        self.patch_embeddings = SimVLMPatchEmbeddings(config)
        
        # position embedding for image modality
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(
            nn.init.trunc_normal_(
                torch.zeros(1, num_patches, config.hidden_size),
                mean=0.0,
                std=config.initializer_range
            )
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config


    def interpolate_pos_encding(self, embeds):
        pass

    def visual_embedding(
        self,
        image: torch.Tensor,
        interpolate_pos_encoding: bool = False
    ) -> torch.Tensor:
        embeds = self.patch_embeddings(image)
    
        if interpolate_pos_encoding:
            embeds = embeds + self.interpolate_pos_encding(embeds, )
        else:
            embeds = embeds + self.position_embeddings

        embeds = self.dropout(embeds)
        return embeds

    def forward(
        self,
        image: torch.Tensor,
        prefix_text: Optional[torch.Tensor],
        tgt_text: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        
        embeds = {}
        # PART 1: Patch embeddings (with interpolated position encodings)
        image_embeds = self.visual_embedding(image)
        embeds['image_embeds'] = image_embeds

        if self.training:
            # PART 2: Text embeddings
            assert prefix_text is not None and tgt_text is not None
            prefix_text_embeds, tgt_text_embeds = self.text_embeddings(prefix_text, tgt_text)
            embeds['prefix_text_embeds'], embeds['tgt_embeds'] = (
                prefix_text_embeds, tgt_text_embeds
            )

            # PART 3: concatenate image embeds and prefix text embeds 
            co_embeds = torch.cat([image_embeds, prefix_text_embeds], dim=1)
            embeds['co_embeds'] = co_embeds
        
        return embeds


class SimVLMModel(nn.Module):
    def __init__(
        self,
        config: SimVLMConfig = SimVLMConfig(),
    ):
        super(SimVLMModel, self).__init__()
        self.embeddings = SimVLMEmbeddings(config)
        self.encoder = SimVLMEncoder(config)
        self.decoder = SimVLMDecoder(config)
        # self.output = SimVLMOutput(config)
        # self.generator = SimVLMGenerator(config)
        self.to_logits = SimVLMToLogits(config)
        self.config = config

    def forward_loss(self, logits: torch.Tensor, tgt_txt: torch.Tensor):
        assert logits.dim() == 3 and tgt_txt.dim() == 2
        assert logits.shape[:2] == tgt_txt.shape[:2]

        return F.cross_entropy(
            logits.contiguous().view(-1, logits.shape[-1]),
            tgt_txt.contiguous().view(-1)
        )
    
    def forward_encoder(self, co_embeds, prefix_text: Optional[torch.Tensor]):
        device=co_embeds.device
        # mask padding token_ids to make sure they do not contribute to gratitude
        memory_mask = torch.ones(*co_embeds.shape[:2], device=device)
        if prefix_text is not None:
            memory_mask[:, self.config.image_token_len:] = (prefix_text == self.config.pad_token_id)
        else:
            memory_mask = torch.ones(1, 1, co_embeds.shape[1], device=device)

        memory_mask = memory_mask.unsqueeze(-2)
        # memory_mask = (prefix_text == self.config.pad_token_id).unsqueeze(-2)

        memory = self.encoder(co_embeds, memory_mask)

        return memory, memory_mask

    def forward(
        self, 
        image: torch.Tensor, 
        prefix_text: torch.Tensor,
        decoder_input_text: torch.Tensor,
        label_text: Optional[torch.Tensor]=None,
    ) -> torch.Tensor:

        embeds = self.embeddings(image, prefix_text, decoder_input_text)
        image_embeds, co_embeds, tgt_embeds = (
            embeds['image_embeds'], embeds['co_embeds'], embeds['tgt_embeds']
        )
        self.config.image_token_len = image_embeds.shape[1]

        memory, memory_mask = self.forward_encoder(co_embeds, prefix_text)

        tgt_mask = self.make_std_mask(decoder_input_text, pad=self.config.pad_token_id)
        decoder_output = self.decoder(tgt_embeds, memory, memory_mask, tgt_mask)
        # output = self.generator(decoder_output.last_hidden_state)
        logits = self.to_logits(decoder_output.last_hidden_state)
        if self.training:
            loss = self.forward_loss(logits, label_text)

            return loss
        
        return logits
    
    # TODO integrate into forward based on whether model is on train or eval
    @torch.no_grad()
    def generate(
        self,
        image: torch.Tensor,
        prompt_text: Optional[torch.Tensor]=None,
        generate_mode='greedy',
        max_generate_len=11
    ):
        self.eval()
        self.embeddings.eval()

        device = image.device
        if prompt_text is not None:
            prompt_text = batch_input(prompt_text)
            if prompt_text.shape[1] != self.config.prefix_text_len:
                prompt_text = pad_with_id(prompt_text, max_len=self.config.max_prefix_text_len, pad_token_id=self.config.pad_token_id)
        else:
            prompt_text = (
                torch.empty(image.shape[0],
                self.config.max_prefix_text_len, device=device, dtype=torch.int64).fill_(self.config.pad_token_id)
            )
        prompt_text_embeds = self.embeddings.text_embeddings(prompt_text)
        image_embeds = self.embeddings.visual_embedding(batch_input(image))

        co_embeds = torch.cat([image_embeds, prompt_text_embeds], dim=1)
        assert co_embeds.shape[1] == self.config.encoder_input_len

        memory, memory_mask = self.forward_encoder(co_embeds, prompt_text)
        
        
        if generate_mode == 'greedy':
            return self.greedy_search(memory, memory_mask, max_generate_len)
        if generate_mode == 'top':
            pass

    # TODO modify this
    def greedy_search(self, memory, memory_mask, max_generate_len):
        # auto-regressive generation
        device = memory.last_hidden_state.device
        batch_size = memory.last_hidden_state.shape[0]
        decoder_input_text = torch.zeros(batch_size, 1, dtype=torch.int64).fill_(self.config.bos_token_id).to(device)

        for i in range(max_generate_len - 1):
            decoder_input_mask = subsequent_mask(decoder_input_text.shape[1]).to(device)
            decoder_input_embeds = self.embeddings.text_embeddings(decoder_input_text)
            decoder_output = self.decoder(decoder_input_embeds, memory, memory_mask, decoder_input_mask)
            logits = self.to_logits(decoder_output.last_hidden_state)
        
            assert logits.dim() == 3
            next_word_scores = F.log_softmax(logits[:, -1], dim=-1)
            _, next_word = torch.max(next_word_scores, dim=1)
            next_word = next_word.detach()[0]
            decoder_input_text = torch.cat(
                [decoder_input_text, torch.zeros(batch_size, 1, dtype=torch.int64).fill_(next_word).to(device)], dim=1
            )

        return decoder_input_text
        

    @staticmethod
    def make_std_mask(tgt: torch.Tensor, pad: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.detach()
        )
        return tgt_mask
        

class SimVLMGenerator(nn.Sequential):
    def __init__(self, config: SimVLMConfig):
        super().__init__(
            nn.Linear(config.hidden_size, config.vocab_size),
            nn.LogSoftmax(dim=-1)
        )


class SimVLMToLogits(nn.Sequential):
    def __init__(self, config: SimVLMConfig):
        super().__init__(
            nn.Linear(config.hidden_size, config.vocab_size),
        )


# TODO Reconstruct class 'Attention's
class SimVLMCrossAttention(nn.Module):
    def __init__(self, config: SimVLMConfig):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of"
                f"the number of the attention heads"
            )
        super().__init__()
        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: Optional[torch.Tensor] = None, 
        mask: torch.Tensor = None,
        mask_value: float = 1e-9,
        head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        
        mixed_query_layer = self.query(query)

        if value is None:
            value = key
        key_layer = self._transpose_for_scores(self.key(key))
        value_layer = self._transpose_for_scores(self.value(value))
        query_layer = self._transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Mask pad tokens to make sure thy make no contributions to gradients
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs


class SimVLMSelfAttention(nn.Module):
    def __init__(self, config: SimVLMConfig):
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of"
                f"the number of the attention heads"
            )
        super().__init__()
        self.num_attention_heads = config.num_attention_heads

        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  

    def forward(
        self,
        hidden_states, 
        mask: torch.Tensor,
        mask_value: float = 1e-9,
        head_mask: Optional[torch.Tensor] = None, 
        output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self._transpose_for_scores(self.key(hidden_states))
        value_layer = self._transpose_for_scores(self.value(hidden_states))
        query_layer = self._transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Mask pad tokens to make sure they make no contributions to gradients
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
            attention_scores = attention_scores.masked_fill(mask == 0, mask_value)

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
        
    

class SimVLMSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: SimVLMConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:

        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class SimVLMAttention(nn.Module):
    def __init__(self, config: SimVLMConfig) -> None:
        super().__init__()
        self.attention = SimVLMSelfAttention(config)
        self.output = SimVLMSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        src_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, src_mask)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class SimVLMIntermediate(nn.Module):
    def __init__(self, config: SimVLMConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = activations.ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        return self.intermediate_act_fn(self.dense(hidden_states))


class SimVLMOutput(nn.Module):
    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
        
        return input + self.dropout(self.dense(hidden_states))



class   SimVLMEncoderLayer(nn.Module):
    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.attention = SimVLMAttention(config)
        self.intermediate = SimVLMIntermediate(config)
        self.output = SimVLMOutput(config)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        src_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attention: bool = False,
    ):
        # self attention block
        attention_output = self.attention(
            self.layernorm(hidden_states),
            src_mask,
            head_mask,
            output_attention
        )
        # skip connection
        attention_output = self.layernorm(attention_output[0] + hidden_states)

        # FeedForward block
        intermediate_output = self.intermediate(
            self.layernorm(attention_output)
        )
        # skip connection
        output = self.output(intermediate_output, attention_output)

        return output

class SimVLMEncoder(nn.Module):
    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SimVLMEncoderLayer(config) for _ in range(config.num_encoder_layers)
        ])
        self.gradient_checkpointing = False
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    layer_head_mask,
                )
            else:
                layer_outputs = layer_module(hidden_states, mask, layer_head_mask, output_attentions)

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )

class SimVLMDecoder(nn.Module):
    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            SimVLMDecoderLayer(config) for _ in range(config.num_decoder_layers)
        ])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory: torch.Tensor,
        memmory_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    memory,
                    memmory_mask,
                    tgt_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states, 
                    memory,
                    memmory_mask,
                    tgt_mask,
                    )

            hidden_states = layer_outputs

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        


class SimVLMDecoderLayer(nn.Module):
    def __init__(self, config: SimVLMConfig):
        super().__init__()
        self.config = config
        self.cross_attention = SimVLMCrossAttention(config)
        self.self_attention = SimVLMSelfAttention(config)
        self.intermediate = SimVLMIntermediate(config)
        self.output = SimVLMOutput(config)
        self.layernorm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ):
        # Self attention block
        self_attention_output = self.self_attention(
            self.layernorm(x),
            tgt_mask,
        )
        # skip connection
        self_attention_output = self.layernorm(self_attention_output[0] + x)

        # Cross attention block
        cross_attention_output = self.cross_attention(
            self.layernorm(self_attention_output),
            memory.last_hidden_state,
            mask=memory_mask,
        )
        # skip connection
        cross_attention_output = self.layernorm(cross_attention_output[0] + self_attention_output)
        
        # Feed-Forward block
        intermediate_output = self.intermediate(cross_attention_output)
        output = self.output(intermediate_output, cross_attention_output)

        return output

def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
            torch.uint8
        )
        return subsequent_mask == 0

def batch_input(input_tensor):
    if input_tensor.dim() in [1, 3]:
        _input_tensor = torch.clone(input_tensor).unsqueeze(0)
    elif input_tensor.dim() in [2, 4]:
        _input_tensor = torch.clone(input_tensor)
    else:
        raise ValueError(f'Invalid shape of images or token_ids {input_tensor.shape}, it has dim {input_tensor.dim()}')
    return _input_tensor


def pad_with_id(input_ids, max_len, pad_token_id):
    input_ids = batch_input(input_ids)
    seq_len = input_ids.shape[1]
    if seq_len > max_len:
        return ValueError("Input length is beyongd expected length")

    padded_ids = torch.empty(input_ids.shape[0], max_len, device=input_ids.device, dtype=torch.int64)
    padded_ids[:, :seq_len] = input_ids
    padded_ids[:, seq_len:].fill_(pad_token_id)
    return padded_ids