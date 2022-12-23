from typing import Union, Optional, List
from transformers.utils import TensorType

from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import (
    BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
)


class SimVLMProcessor(ProcessorMixin):
    r"""
    Constructs a ViLT processor which wraps a BERT tokenizer and ViLT feature extractor into a single processor.

    [`ViltProcessor`] offers all the functionalities of [`ViltFeatureExtractor`] and [`BertTokenizerFast`]. See the
    docstring of [`~ViltProcessor.__call__`] and [`~ViltProcessor.decode`] for more information.

    Args:
        feature_extractor (`ViltFeatureExtractor`):
            An instance of [`ViltFeatureExtractor`]. The feature extractor is a required input.
        tokenizer (`BertTokenizerFast`):
            An instance of ['BertTokenizerFast`]. The tokenizer is a required input.
    """
    attributes = ['tokenizer']
    tokenizer_class = ("BartTokenizer", 'BertGenerationTokenizer', 'RobertaTokenizer', 'BartTokenizerFast')

    def __init__(self, tokenizer):
        super().__init__(tokenizer)

    def __call__(
        self,
        images: Optional[TensorType] = None,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        stride: int = 0,
        pad_to_multiple_of: Optional[int] = None,
        return_token_type_ids: Optional[bool] = None,
        return_attention_mask: Optional[bool] = None,
        return_overflowing_tokens: bool = False,
        return_special_tokens_mask: bool = False,
        return_offsets_mapping: bool = False,
        return_length: bool = False,
        verbose: bool = True,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs
    ) -> BatchEncoding:
        """
        This method uses [`ViltFeatureExtractor.__call__`] method to prepare image(s) for the model, and
        [`BertTokenizerFast.__call__`] to prepare text for the model.

        Please refer to the docstring of the above two methods for more information.
        """
        encoding = self.tokenizer(
            text=text,
            add_special_tokens=add_special_tokens,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            stride=stride,
            pad_to_multiple_of=pad_to_multiple_of,
            return_token_type_ids=return_token_type_ids,
            return_attention_mask=return_attention_mask,
            return_overflowing_tokens=return_overflowing_tokens,
            return_special_tokens_mask=return_special_tokens_mask,
            return_offsets_mapping=return_offsets_mapping,
            return_length=return_length,
            verbose=verbose,
            return_tensors=return_tensors,
            **kwargs,
        )
        
        return encoding

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to BertTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    