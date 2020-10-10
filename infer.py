import os
import numpy as np

from tqdm import tqdm

import sys
sys.path.append('./mmdetection/')
from mmdet import __version__
from mmdet.apis import init_detector, inference_detector


def infer(config, checkpoint, img_file_dir, output_dir, json_name='bbox_score.json', show_score_thr=0.3):

    model = init_detector(config, checkpoint, device='cuda:0')
    img_dir = img_file_dir
    file_name_list = os.listdir(img_dir)
    file_name_list.sort()
    img_dir2 = img_dir.replace('_p', '')
    results = {}
    ik = 0
    for i in tqdm(range(len(file_name_list))):
        file_name = file_name_list[i]
        if os.path.splitext(file_name)[1] not in ['.jpg', '.png', '.bmp', '.gif']:
            continue
        #if file_name != '1013.png':
            #continue
        
        result_p = inference_detector(model, img_dir + file_name)
        result_c = inference_detector(model, img_dir2 + file_name)
        if isinstance(result_p, tuple):
            bbox_results, _ = result_p
            result_p = bbox_results
            bbox_results, _ = result_c
            result_c = bbox_results
        result_above_confidence_num_p = 0
        result_above_confidence_num_c = 0
        '''
        result_p = np.concatenate(result_p)
        result_c = np.concatenate(result_c)
        '''
        #for ir in range(len(result_p)):
        for ir in range(result_p.shape[0]):
            if result_p[ir, 4] > show_score_thr:
                result_above_confidence_num_p = result_above_confidence_num_p + 1
        #for ir in range(len(result_c)):
        for ir in range(result_c.shape[0]):
            if result_c[ir, 4] > show_score_thr:
                result_above_confidence_num_c = result_above_confidence_num_c + 1
        if result_above_confidence_num_c == 0:  # can't find any object in clean img
            bb_score = 0
            print('i=', ik)
            print(file_name)
            ik += 1
        else:
            bb_score = 1 - min(result_above_confidence_num_c,
                               result_above_confidence_num_p) / result_above_confidence_num_c
        results[file_name] = bb_score
    import json
    with open(os.path.join(output_dir, json_name), 'w') as f_obj:
        json.dump(results, f_obj)
    return results
