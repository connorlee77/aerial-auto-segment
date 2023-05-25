import cv2
import numpy as np
import glob
import os

FRAMES_DIR = 'video_frames'
files = sorted(glob.glob('outputs/thermal-*_pts.png'))
for i, fp in enumerate(files):
    orig_fp = fp.replace('_pts', '')
    autoseg_fp = fp.replace('_pts', '_autoseg')
    autoseg_refined_fp = fp.replace('_pts', '_autoseg_refined')

    orig = cv2.imread(orig_fp, 1)
    pts = cv2.imread(fp, 1)
    autoseg = cv2.imread(autoseg_fp, 1)
    autoseg_ref = cv2.imread(autoseg_refined_fp, 1)

    img1 = np.hstack([orig, pts])
    img2 = np.hstack([autoseg, autoseg_ref])
    img = np.vstack([img1, img2])
    cv2.imwrite(os.path.join(FRAMES_DIR, '{}.png'.format(str(i).zfill(4))), img)