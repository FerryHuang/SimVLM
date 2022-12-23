import json
import os
import pickle
import random
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from dataset.prefix_dataset import PrefixDataset

class CocoCaptionKarpathyDataset(PrefixDataset):
    def __init__(self, split, **kwargs) -> None:
        split = 'test' if split == 'val' else split
        if not 'data_root' in kwargs:
            kwargs['data_root'] = "/share/dataset/vl_dataset/coco_captions/"

        super().__init__(split=split, **kwargs)
        self.prefix_text_len = 1

    def _random_prefix_text_len(self):
        assert self.random_prefix_len
        
        random.seed(self.seed)
        return random.choice(range(2, 8))

    def _get_absolute_image_path(self, file):
        parent = 'train2014' if 'train' in file else 'val2014'
        return os.path.join(self.data_root, parent, file)


    def _assign_data(
        self,
        cache_dir='./.cache/coco_assigned/',
        json_with_parent= "karpathy/dataset_coco.json",
        cat_train_restval=True
    ):

        fn2captions_pickle = os.path.join(cache_dir, 'fn2captions.pickle')
        split2fns_pickle = os.path.join(cache_dir, 'split2fns.pickle')
        
        if os.path.exists(fn2captions_pickle) and os.path.exists(split2fns_pickle):
            print('Using cache')
            return fn2captions_pickle, split2fns_pickle

        with open(
            os.path.join(self.data_root, json_with_parent), "r"
        ) as fp:
            captions = json.load(fp)

        all_captions = captions["images"]
        # list of dicts {'filepath'}
        fn2captions = defaultdict(list)
        split2fns = {
            'train': [],
            'val': [],
            'restval': [],
            'test': []
        }

        for caption in tqdm(all_captions):

            fn = caption['filename']
            split = caption['split']
            split2fns[split].append(fn)

            sentences = caption['sentences']
            fn2captions[fn] = []
            for sent in sentences:
                # ignore 'token' sentence for we will tokenize it later ourselves 
                fn2captions[fn].append(sent['raw'])
        
        if cat_train_restval:
            split2fns['train'].extend(split2fns['restval'])
            del split2fns['restval']

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        with open(fn2captions_pickle, 'wb') as f:
            pickle.dump(fn2captions, f)

        with open(split2fns_pickle, 'wb') as f:
            pickle.dump(split2fns, f)

        return fn2captions_pickle, split2fns_pickle
        


