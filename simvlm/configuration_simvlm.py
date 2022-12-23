from transformers.configuration_utils import PretrainedConfig
from transformers import ViltConfig

class SimVLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTModel`]. It is used to instantiate an ViT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the ViT
    [google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_encoder_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer decoder.
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        image_size (`int`, *optional*, defaults to `224`):
            The size (resolution) of each image.
        patch_size (`int`, *optional*, defaults to `16`):
            The size (resolution) of each patch.
        num_channels (`int`, *optional*, defaults to `3`):
            The number of input channels.
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        encoder_stride (`int`, `optional`, defaults to 16):
           Factor to increase the spatial resolution by in the decoder head for masked image modeling.

    Example:

    ```python
    >>> from transformers import ViTModel, ViTConfig

    >>> # Initializing a ViT vit-base-patch16-224 style configuration
    >>> configuration = ViTConfig()

    >>> # Initializing a model from the vit-base-patch16-224 style configuration
    >>> model = ViTModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    model_type = "simvlm"

    def __init__(
        self,
        hidden_size=768,
        num_encoder_layers=12,
        num_decoder_layers=12,
        num_attention_heads=12,
        intermediate_size=768,
        hidden_act="gelu",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_encoder_decoder=True,
        image_size=224,
        patch_size=16,
        num_channels=3,
        qkv_bias=True,
        encoder_stride=16,
        image_token_len=196,
        encoder_input_len=256,
        decoder_input_len=256,
        prefix_text_len=5,
        max_prefix_text_len=60,
        vocab_size=50265,
        max_position_embeddings=60,
        padding_mode="max_length",
        bos_token_id=0,
        pad_token_id=1,
        eos_token_id=2,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.hidden_size = hidden_size
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_encoder_decoder = is_encoder_decoder
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.encoder_stride = encoder_stride
        self.image_token_len = image_token_len
        self.encoder_input_len = encoder_input_len
        self.decoder_input_len = decoder_input_len
        self.prefix_text_len = prefix_text_len
        self.max_prefix_text_len = max_prefix_text_len
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.padding_mode = padding_mode
        self.max_tgt_text_len = decoder_input_len
