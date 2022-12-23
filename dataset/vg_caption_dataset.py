import json
import os
from glob import glob
from pathlib import Path
import pickle
import random
from tqdm import tqdm
from collections import defaultdict
from dataset.prefix_dataset import PrefixDataset

class VisualGenomeCaptionDataset(PrefixDataset):
    def __init__(
        self,
        split='',
        **kwargs,
    ) -> None:

        if split != 'train':
            raise ValueError('Invalid split argument')
        if not 'data_root' in kwargs:
            kwargs['data_root'] = '/share/dataset/vl_dataset/visual_genome/'

        super().__init__(**kwargs, split=split)
        self.prefix_text_len = 2


    def _assign_data(self, cache_dir='./.cache/vg_assigned/',
                     json_with_parent="annotations/region_descriptions.json"):

        fn2captions_pickle = os.path.join(cache_dir, 'fn2captions.pickle')
        split2fns_pickle = os.path.join(cache_dir, 'split2fns.pickle')
        if os.path.exists(fn2captions_pickle) and os.path.exists(split2fns_pickle):
            return fn2captions_pickle, split2fns_pickle

        data_root = self.data_root
        with open(
            os.path.join(data_root, json_with_parent), "r"
        ) as fp:
            captions = json.load(fp)
            iid2captions = defaultdict(list)
            for caption in tqdm(captions):
                caption = caption['regions']
                for each in caption:
                    iid2captions[each['image_id']].append(each['phrase'])
        
        paths = list(glob(f"{data_root}/images/VG_100K/*.jpg")) + list(
            glob(f"{data_root}/images/VG_100K_2/*.jpg")
        )
        random.shuffle(paths)
        
        fn2captions = defaultdict(list)
        for path in paths:
            iid = int(path.split("/")[-1][:-4])
            if iid in iid2captions:
                fn2captions[path] = iid2captions[iid]

        split2fns = {
            'train': list(fn2captions.keys())
        }

        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        with open(fn2captions_pickle, 'wb') as f:
            pickle.dump(fn2captions, f)

        with open(split2fns_pickle, 'wb') as f:
            pickle.dump(split2fns, f)

        return fn2captions_pickle, split2fns_pickle