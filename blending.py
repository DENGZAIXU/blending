import cv2
import os
import json
import time
import numpy as np

def extract_contour(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (_, thresh) = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY)
    _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print(contours)
    cnts = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    #print(cnts)


def save_ann(ann_file, anns, img_fname, shape):
    ann_dict = {}
    ann_dict['version'] = '3.16.1'
    ann_dict['flags'] = {}
    ann_dict['shapes'] = anns
    ann_dict['lineColor'] = [0, 255, 0, 128]
    ann_dict['fillColor'] = [255, 255, 0, 64]
    ann_dict['imagePath'] = img_fname
    ann_dict['imageData'] = None
    ann_dict['imageHeight'] = shape[0]
    ann_dict['imageWidth'] = shape[1]
    with open(ann_file, 'w') as f:
        json.dump(ann_dict, f, indent=4, ensure_ascii=False)
 

def intersection(bboxes, bbox):
    bboxes = np.array(bboxes)
    lt = np.maximum(bboxes[:, :2], bbox[:2])
    rb = np.minimum(bboxes[:, 2:], bbox[2:])
    inter_wh = np.maximum(rb - lt, 0)
    areas = inter_wh[:, 0] * inter_wh[:, 1]
    if areas.sum() == 0:
        return False
    else:
        return True


if __name__ == '__main__':
    eps = 1e-5
    ann_dir = 'ann'
    out_dir = 'out'
    assert(not os.path.exists(out_dir))
    os.mkdir(out_dir)
    for img_dir in ['1', '2']:
        fnames = sorted(os.listdir(img_dir))
        drinks = []
        for fn in fnames:
            ann_path = os.path.join(ann_dir, fn[:-4] + '.json')
            ann = json.load(open(ann_path, 'r'))
            bbox = ann['shapes'][0]['points']
            x1, y1, x2, y2 = int(bbox[0][0]), int(bbox[0][1]), int(bbox[1][0]), int(bbox[1][1])
     
            img_path = os.path.join(img_dir, fn)
            drinks.append((img_path, \
                    np.array(bbox, dtype=np.int64).reshape(-1), ann['shapes'][0]))
    
        gen_num = 150
        max_beverage_num = 10
        for gi in range(gen_num):
            base_idx = np.random.randint(len(drinks))
            base_path, base_bbox, base_ann = drinks[base_idx]
            base_img = cv2.imread(base_path)
            bboxes = [base_bbox]
            anns = [base_ann]
            bnum = np.random.randint(max_beverage_num)
    
            try_times = 0
            MAX_TRY_TIMES = 200
            while bnum:
                try_times += 1
                if try_times > MAX_TRY_TIMES:
                    break
                sel_idx = np.random.randint(len(drinks))
                sel_path, sel_bbox, sel_ann = drinks[sel_idx]
                if not intersection(bboxes, sel_bbox):
                    bboxes.append(sel_bbox)
                    anns.append(sel_ann)
                    sel_img = cv2.imread(sel_path)
                    x1, y1, x2, y2 = sel_bbox
                    base_part = base_img[y1:y2, x1:x2]
                    sel_part = sel_img[y1:y2, x1:x2]
                    #cnt = extract_contour(sel_part)
                    w = np.minimum(3000 / (np.sum(np.abs(base_part.astype(np.float32) - \
                            sel_part + eps), axis=2)**2), 1)
                    w = w[..., np.newaxis]
                    base_img[y1:y2, x1:x2] = (base_part * w + sel_part * (1-w)).astype(np.uint8)
                    bnum -= 1

            fname = 'blending_%d' % int(time.time()*1000000)
            img_file = os.path.join(out_dir, fname + '.jpg')
            ann_file = os.path.join(out_dir, fname + '.json')
            cv2.imwrite(img_file, base_img)
            save_ann(ann_file, anns, fname + '.jpg', base_img.shape)
    
            cv2.imshow('show', base_img)
            key = cv2.waitKey(1)
            if key == 27:
                exit()
