import os
import shutil
import cv2
import numpy as np
import random
interesting_results = [
    # tiny river, oem is best, road shwoing
    '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-05-15_ColoradoRiver/flight3/thermal-30531.png',
    # big river, all good, oduse is okay.
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-05-15_ColoradoRiver/flight3/thermal-14688.png',
    # same as above, odise not as good
    '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-05-15_ColoradoRiver/flight3/thermal-18108.png',
    # oem best, can see a road that is hard to see
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-05-15_ColoradoRiver/flight3/thermal-27291.png',
    # dw is best, all good; odise got bridge but not robust at trees

    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2021-09-09-KentuckyRiver/flight1-1/thermal-04619.png',
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2021-09-09-KentuckyRiver/flight2-1/thermal-00275.png',
    
    # duck pier highlighted oem is best
    '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-09-59-39/thermal-25310.png',


    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-14-06-04/thermal-07338.png',  # duck, dw/oem is good
    # duck, dw/oem good, a bit boring
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-19-55-11/thermal-05242.png',
    # duck, dw/oem good, unclear why dw is different
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-19-55-11/thermal-06502.png',
    # duck, dw-planet decent, shows houses.
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-19-55-11/thermal-56524.png',
    # duck, oem best, dw good, odise is okay.
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2023-03-XX_Duck/ONR_2023-03-21-09-59-39/thermal-06380.png',
    # dw best, odise/oem not good
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-12-20_Castaic_Lake/flight4/thermal-02778.png',
    # all good, oem best, odise bad
    '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-12-20_Castaic_Lake/flight4/thermal-25192.png',

    '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-12-20_Castaic_Lake/flight4/thermal-42266.png',
    # oem best, shows road where we expect it to be hard; small water
    # '/home/connor/repos/aerial-auto-segment/figures/results/test_strips/common/2022-12-20_Castaic_Lake/flight4/thermal-43185.png',
]

# Strong temporal shift
# /home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs/most_common/open_sam_boxnms_0p50/dynamicworld/dem/1.0/crf_planet/2022-12-20_Castaic_Lake/flight4/thermal-03858_strip.jpg 
def add_hspace(d):
    for key in d:
        h, w, c = d[key][0].shape
        d[key].append(np.ones((h, 10, c), dtype=np.uint8)*255)

zip_folder = 'interesting_results'
os.makedirs(zip_folder, exist_ok=True)
data_dict = dict(gt=[], image=[], dw_planet=[], dw_none=[], odise=[], oem_naip=[])
for result in interesting_results:
    split_path = result.split('/')
    label_set, place, trajectory, name = split_path[-4], split_path[-3], split_path[-2], split_path[-1]

    
    for label_set in ['common', 'most_common']:

        save_path = os.path.join(zip_folder, '{}_{}_{}_{}'.format(label_set, place, trajectory, name))
        # os.makedirs(save_path, exist_ok=True)
        # shutil.copy(result.replace('common', label_set), save_path)

        # shutil.copytree(result.replace('common', label_set).replace('test_strips', 'test_individuals').replace('.png', ''), save_path.replace('.png', ''))
        image_dir = result.replace('common', label_set).replace('test_strips', 'test_individuals').replace('.png', '')
        img = cv2.imread(os.path.join(image_dir, 'image.png'), 1)
        gt = cv2.imread(os.path.join(image_dir, 'gt.png'), 1)
        dw_none = cv2.imread(os.path.join(image_dir, 'dw_none.png'), 1)
        dw_planet = cv2.imread(os.path.join(image_dir, 'dw_planet.png'), 1)
        odise = cv2.imread(os.path.join(image_dir, 'odise.png'), 1)
        oem_naip = cv2.imread(os.path.join(image_dir, 'oem_naip.png'), 1)

        data_dict['gt'].append(gt)
        data_dict['image'].append(img)
        data_dict['dw_planet'].append(dw_planet)
        data_dict['dw_none'].append(dw_none)
        data_dict['odise'].append(odise)
        data_dict['oem_naip'].append(oem_naip)
    
        add_hspace(data_dict)

for key in data_dict:
    data_dict[key].pop()


img_row = np.hstack(data_dict['image'])
dw_planet_row = np.hstack(data_dict['dw_planet'])
dw_none_row = np.hstack(data_dict['dw_none'])
oem_naip_row = np.hstack(data_dict['oem_naip'])
odise_row = np.hstack(data_dict['odise'])
gt_row = np.hstack(data_dict['gt'])
rows = [img_row, odise_row, dw_planet_row, dw_none_row, oem_naip_row, gt_row]

vspace = np.ones((10, img_row.shape[1], 3), dtype=np.uint8)*255
new_rows = []
for row in rows:
    new_rows.append(row)
    new_rows.append(vspace)
new_rows.pop()

img = np.vstack(new_rows)
cv2.imwrite('result.png', img)