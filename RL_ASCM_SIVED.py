import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import torch
import torchvision
import torchvision.models as Models
import torchvision.transforms as Transforms
from PIL import Image
import urllib.request
import json
from functools import reduce

import PatchAttack.PatchAttack_attackersMSTAR as PA

from PatchAttack.PatchAttack_config import configure_PA


from SMGAA_folder.utils_test_MSTAR import *
from PIL import Image
from yolo3_main.Yolov3_attack_mstar_single import *

from PatchAttack import utils
# f = open('/media/lenovo/新加卷/DatasetSave1/MSTAR/MSTAR_Detection/MSTAR_detection_val.txt')
f = open('/media/lenovo/新加卷/DatasetSave1/MSTAR/MSTAR_Detection/MSTAR_detection_val_fineselect.txt')
val_lines = f.readlines()


iters = 0; index = 0
# over_list = os.listdir('/media/lenovo/新加卷/ASCMAttack_SARAIRcrafts/ASCMPA2')
for i in range(len(val_lines)):

    iters += 1
    # tar_tag = 6
    # set_seed(iter)

    image_path = val_lines[i].split()[0].replace('MSTAR_Detection','MSTAR_Detection/val')
    image_id = os.path.basename(image_path).split('.')[0]
    # if image_id in over_list:
    #     continue
    input_img = Image.open(image_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=input_shape)  # input_shape with the form of w, h
    x_min_shift, y_min_shift, x_max_shift, y_max_shift, scale = x_meta
    # x_query.show()
    w, h = input_img.size
    bbbox, bbbox_nor, mask = read_assigned_target_attackv2(val_lines[i], x_meta, h, w, input_shape)


    data = np.array(x_query) / 255.
    data = torch.from_numpy(data[None].transpose((0, 3, 1, 2))).float().cuda()
    detections_ori = train_util.detect_image_attackv2(data, confidence=0.5, nms_iou=0.3)
    if len(detections_ori) != 0:
        detections_ori = detections_ori[-1]
    else:
        continue
    areas = utils.calculate_iou(torch.from_numpy(bbbox[:, :4][:, [1, 0, 3, 2]]), torch.from_numpy(detections_ori[:, :4]))
    detec_ious_max, detec_idx = torch.max(areas, dim=0)
    # visualize_detections_nor({'begine image': (np.array(x_query), detections_ori.copy(), class_names)})
    detec_idx = detec_idx.numpy().copy()
    detec_ious_max = detec_ious_max.numpy().copy()
    detec_idx_iou = detec_idx[detec_ious_max > 0.5]

    ori_cls = bbbox[detec_idx][:,-1]
    pred_cls = detections_ori[:,-1]
    detec_idx_re = detec_idx[ori_cls == pred_cls]

    unique_element, count = np.unique(detec_idx_iou, return_counts=True)
    detec_idx_unique = unique_element[count==1]


    detec_idx_final = reduce(np.intersect1d, (detec_idx_iou, detec_idx_re, detec_idx_unique))

    visualize_detections_nor({'ground truth': (np.array(x_query), bbbox[:, [1, 0, 3, 2, -1]], class_names)})
    # visualize_detections_nor({'begine image': (np.array(x_query), detections_ori, class_names)})
    adv_data_ = data

    query_num_all = 0
    T_Attack, F_Attack = 0, 0
    d_line_dict = {}
    if len(detec_idx_final) == 0:
        continue

    for tar_tag  in detec_idx_final:
        mask = np.zeros(input_shape)
        # mask[int(bbbox[tar_tag][1]):int(bbbox[tar_tag][3]), int(bbbox[tar_tag][0]):int(bbbox[tar_tag][2])] = 1.0

        y_center, x_center = int((int(bbbox[tar_tag][1]) + int(bbbox[tar_tag][3])) / 2), int((int(bbbox[tar_tag][0]) + int(bbbox[tar_tag][2]))/ 2)
        h_2, w_2 = int(3 * (bbbox[tar_tag][3] - bbbox[tar_tag][1]) / 2), int(3 * (bbbox[tar_tag][2] - bbbox[tar_tag][0]) / 2)
        y_min1,y_max1,x_min1,x_max1 = np.clip(y_center - h_2, 0, 512),np.clip(y_center + h_2, 0, 512),np.clip(x_center - w_2, 0, 512),np.clip(x_center + w_2, 0, 512)
        mask[y_min1:y_max1, x_min1:x_max1] = 1.0

        mask = torch.from_numpy(mask)

        dir_title = 'PatchAttack_SMGAA_Object_MSTAR_{}_target_{}'.format(image_id,tar_tag)  # used to form the path where the results are saved
        # configure PA_cfg
        configure_PA(
            t_name='',
            t_labels=np.arange(10).tolist(), # all the possible labels, start from 0 and continuous
            target=False, # targetted or non-targetted attack
            n_occlu=4, # number of patches superimposed whose size is optimized
            rl_batch=10, steps=1000,
            MPA_color=True, # MPA_Gray or MPA_RGB
        )

        label_tensor = torch.LongTensor([0.0]).cuda()

        ASCMPA = PA.ASCMPA(image_id, dir_title, ran_name='ASCMPA',firname='ASCMAttack_MSTAR')
        rcd = ASCMPA.attack(
            model=train_util,
            input_tensor=adv_data_,
            label_tensor=label_tensor,
            target=45, # For non-targetted attack, random number
            input_name='{}'.format(index),
            mask_roi =mask,
            bbboxs = bbbox,
            tar_tag = tar_tag,
        )

        # area = ASCMPA.calculate_area(adv_image, rcd.combos[0])
        query_num_all += rcd.queries[0]
        d_line_dict[str(tar_tag)] = rcd.queries[0]
        d_line_dict['sum'] = query_num_all
        print('Area used: {:.4f}'.format(rcd.areas[0].item()))

        # utils.data_agent.show_image_from_tensor(rcd.Adv_datas[0], inv=False)
        if not rcd.non_target_success[0]:
            print('---------------------------attack_fail---------------------------')
            F_Attack += 1
            continue
        T_Attack += 1
        utils.data_agent.show_image_from_tensor(rcd.Adv_Perturbs[0], inv=False)
        adv_data_ = adv_data_ + rcd.suc_att.floating_advpert[None]

        attacked_arr = []
        if rcd.suc_att.detect_arr is not None:
            attacked_arr = rcd.suc_att.detect_arr

        visualize_detections_nor({'Attacked': (adv_data_[0].detach().cpu().numpy().transpose(1, 2, 0), attacked_arr, class_names)})

    print('The image {} --contain objects {} --query_sum_all {}'.format(image_id, len(detec_idx_final), query_num_all))
    print('T_Attack--{} and F_Attack--{}'.format(T_Attack, F_Attack))
    d_line_dict['T_Attack'] = T_Attack; d_line_dict['F_Attack'] = F_Attack
    info_json = json.dumps(d_line_dict,indent=4,separators=(',',':'),ensure_ascii=False)
    f = open(os.path.join('/media/lenovo/新加卷/ASCMAttack/ASCMAttack_MSTAR/ASCMPA',image_id,image_id+'.json'), 'w')
    f.write(info_json)
    f.close()
    break


