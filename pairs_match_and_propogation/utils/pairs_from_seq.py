import os.path as osp
import natsort
from loguru import logger

def compare(img_name):
    key = img_name.split('/')[-1].split('.')[0]
    return int(key)


def pairs_from_seq(img_lists, num_match=5, start_interval=1, gap=1):
    """Get covis images by image id."""
    pairs = []
    img_lists = natsort.natsorted(img_lists)
    img_ids = range(len(img_lists))

    for i in img_ids:
        count = 0
        j = i + start_interval
        
        while j < len(img_ids) and count < num_match:
            if (j - i) % gap == 0:
                count += 1
                pairs.append(" ".join([str(img_lists[i]), str(img_lists[j])])) 
            j += 1
    
    # with open(covis_pairs_out, 'w') as f:
    #     f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
    logger.info(f"Total:{len(pairs)} pairs")
    return pairs