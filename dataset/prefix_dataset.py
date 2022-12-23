import pickle
import random
from PIL import Image, ImageFile
from tqdm import tqdm
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

# TODO add: random select prefix length
#      fix: vg dataset has captions too short, leading to invalid prefix 
class PrefixDataset(Dataset):
    def __init__(
        self, 
        data_root: str,
        # json_with_parent: str,
        transforms,
        processor,
        config,
        split: str = 'train',
        random_prefix_len=False,
        seed=0
    ):

        super().__init__()
        self.data_root = data_root
        self.transforms = transforms(config.image_size)
        self.processor = processor
        self.random_prefix_len = random_prefix_len
        self.prefix_text_len = config.prefix_text_len if not random_prefix_len else None
        self.max_prefix_text_len = config.max_prefix_text_len
        self.max_tgt_text_len = config.max_tgt_text_len
        self.padding_mode = config.padding_mode
        
        self.split = split
        self.is_train = (split == 'train')
        self.seed = seed
        
        fn2captions_pickle, split2fns_pickle = self._assign_data()

        with open(fn2captions_pickle, 'rb') as f:
            self.fn2captions = pickle.load(f)

        with open(split2fns_pickle, 'rb') as f:
            self.split2fns = pickle.load(f)
        
        index_mapper = list()
        img_files = self.split2fns[split]
        for img_file in tqdm(img_files):
            if self.is_train:
                for caption in self.fn2captions[img_file]:
                    index_mapper.append({
                        'image': img_file,
                        'caption': caption,
                    })
            else:
                index_mapper.append({
                    'image': img_file,
                    'caption': self.fn2captions[img_file]
                })

        self.index_mapper = index_mapper

        # passing some attributes to config
        # get actual token_id
        config.pad_token_id = getattr(self.processor.tokenizer, 'pad_token_id')
        config.bos_token_id = getattr(self.processor.tokenizer, 'bos_token_id')
        config.eos_token_id = getattr(self.processor.tokenizer, 'eos_token_id')
        # get actual vocab_size
        config.vocab_size = getattr(self.processor.tokenizer, 'vocab_size')


    def __len__(self) -> int:
        return len(self.index_mapper)
    

    def __getitem__(self, index):

        img_cap_pair = self.index_mapper[index]
        caption = img_cap_pair['caption']
        caption = caption if isinstance(caption, list) else [caption]

        # TODO random select prefix_text_len
        prefix_text_len = self._random_prefix_text_len() if not self.prefix_text_len else self.prefix_text_len

        image = self._get_transformed_image(img_cap_pair['image'])
        # tokenize
        assert self.processor is not None
        prefix_ids, tgt_ids = self._get_tokenized_text(caption[0], prefix_text_len)
        return image, prefix_ids, tgt_ids, caption


    def _get_absolute_image_path(self, file):
        return NotImplementedError("Absolute image paths depend on dataset")

    def _get_transformed_image(self, file):
        if not file.startswith('/'):
            file = self._get_absolute_image_path(file)
            
        image = Image.open(file).convert('RGB')
        if self.transforms:
            # image = [tr(image) for tr in (self.transforms)]
            image = self.transforms(image)
        return image


    def _get_tokenized_text(self, text, prefix_text_len):

        tokenizer = self.processor.tokenizer
        tokenized = tokenizer.tokenize(text)

        prefix_text = tokenized[: prefix_text_len]
        tgt_text = tokenized[prefix_text_len: ]

        prefix_text_encoding = tokenizer(
            text=' '.join(prefix_text),
            # stick to origin model architecture, no eos or bos tokens are added to prefix text
            add_special_tokens=False,
            padding=self.padding_mode,
            truncation=True,
            max_length=self.max_prefix_text_len,
            return_tensors='pt'
        )

        tgt_text_encoding = tokenizer(
            text=' '.join(tgt_text),
            add_special_tokens=True,
            padding=self.padding_mode,
            truncation=True,
            # pad one more for later droppig bos of input and eos of label
            max_length=self.max_tgt_text_len + 1,
            return_tensors='pt'
        )
        return (
            prefix_text_encoding['input_ids'].squeeze(0),
            tgt_text_encoding['input_ids'].squeeze(0)
        )
        


    def _random_prefix_text_len(self):
        raise NotImplementedError("Random setting prefix length depends on dataset")

    # @staticmethod
    def _assign_data(self, *args, **kwargs):
        raise NotImplementedError("Assigning split and image-captions pairs depends on dataset")
