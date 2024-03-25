import os

train_split = 'cartd/train.txt'
val_split = 'cartd/val.txt'
test_split = 'cartd/test.txt'

thermal_root_dir = '/data/onr-thermal/cogito-annotation/converted-cogito-annotations-final'
cm6_root_dir = '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs_v2/common/open_sam_boxnms_0p50/dynamicworld/dem/1.0/none/'
cm5_root_dir = '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs_v2/more_common/open_sam_boxnms_0p50/dynamicworld/dem/1.0/none/'
cm3_root_dir = '/home/connor/repos/aerial-auto-segment/autoseg_refinement/outputs_v2/most_common/open_sam_boxnms_0p50/dynamicworld/dem/1.0/none/'

cm3_gt_dir = '/home/connor/repos/aerial-auto-segment/autoseg_eval/cartd_cm/cm3'
cm5_gt_dir = '/home/connor/repos/aerial-auto-segment/autoseg_eval/cartd_cm/cm5'
cm6_gt_dir = '/home/connor/repos/aerial-auto-segment/autoseg_eval/cartd_cm/cm6'

def create_new_split(split_path, thermal_root_dir, label_root_dir, out_dir, cm_gt_dir=None):
    split_lines = []
    place_count = {}
    with open(split_path, 'r') as f:

        lines = f.readlines()
        for line in lines:
            img_path, label_path = line.strip().split(',')

            place, trajectory, _, name = label_path.split('/')
            if place == 'caltech_duck':
                new_place = '2023-03-XX_Duck'
            elif place == 'kentucky_river':
                new_place = '2021-09-09-KentuckyRiver'
            else:
                new_place = place
            
            # This is the refined labels
            new_label_path = os.path.join(new_place, trajectory, name.replace('pair', 'thermal'))
            label_img_path = os.path.join(label_root_dir, new_label_path)
            
            thermal_img_path = os.path.join(thermal_root_dir, img_path)
            
            if not os.path.exists(thermal_img_path):
                continue
            if not os.path.exists(label_img_path):
                continue

            if cm_gt_dir is not None:
                label_img_path = os.path.join(cm_gt_dir, label_path)
                assert os.path.exists(label_img_path), f"Label image not found: {label_img_path}"
            
            split_lines.append('{},{}'.format(thermal_img_path, label_img_path))
            if new_place in place_count:
                place_count[new_place] += 1
            else:
                place_count[new_place] = 1

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, os.path.basename(split_path)), 'w') as f:
        for line in split_lines:
            x1, x2 = line.split(',')
            assert os.path.exists(x1), f"Thermal image not found: {x1}"
            assert os.path.exists(x2), f"Label image not found: {x2}"
            print(line, file=f)

    print(len(split_lines))
    print(place_count)

create_new_split(train_split, thermal_root_dir, cm6_root_dir, 'cm6')
create_new_split(val_split, thermal_root_dir, cm6_root_dir, 'cm6')
create_new_split(test_split, thermal_root_dir, cm6_root_dir, 'cm6', cm6_gt_dir)

create_new_split(train_split, thermal_root_dir, cm5_root_dir, 'cm5')
create_new_split(val_split, thermal_root_dir, cm5_root_dir, 'cm5')
create_new_split(test_split, thermal_root_dir, cm5_root_dir, 'cm5', cm5_gt_dir)

create_new_split(train_split, thermal_root_dir, cm3_root_dir, 'cm3')
create_new_split(val_split, thermal_root_dir, cm3_root_dir, 'cm3')
create_new_split(test_split, thermal_root_dir, cm3_root_dir, 'cm3', cm3_gt_dir)


create_new_split(train_split, thermal_root_dir, cm6_root_dir, 'cm6_gt', cm6_gt_dir)
create_new_split(val_split, thermal_root_dir, cm6_root_dir, 'cm6_gt', cm6_gt_dir)
create_new_split(test_split, thermal_root_dir, cm6_root_dir, 'cm6_gt', cm6_gt_dir)

create_new_split(train_split, thermal_root_dir, cm5_root_dir, 'cm5_gt', cm5_gt_dir)
create_new_split(val_split, thermal_root_dir, cm5_root_dir, 'cm5_gt', cm5_gt_dir)
create_new_split(test_split, thermal_root_dir, cm5_root_dir, 'cm5_gt', cm5_gt_dir)

create_new_split(train_split, thermal_root_dir, cm3_root_dir, 'cm3_gt', cm3_gt_dir)
create_new_split(val_split, thermal_root_dir, cm3_root_dir, 'cm3_gt', cm3_gt_dir)
create_new_split(test_split, thermal_root_dir, cm3_root_dir, 'cm3_gt', cm3_gt_dir)

