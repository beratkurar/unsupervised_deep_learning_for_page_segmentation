import numpy as np
import cv2
import os

SHOW_RESULTS=False
grid = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

counter=1

def sample_rotated_patch(img, pos, angle, patch_size):

    rotate = np.asarray([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)],])
    _dst = np.asarray(
        [[-patch_size // 2, -patch_size // 2], [-patch_size // 2, patch_size // 2], 
         [patch_size // 2, patch_size // 2], [patch_size // 2, -patch_size // 2]], dtype=np.float32)

    src = np.asarray([[pos[0],pos[1]], [pos[0],pos[1]+patch_size], [pos[0]+patch_size,pos[1]+patch_size], [pos[0]+patch_size, pos[1]]])

    tran = src[0] - _dst[0]

    src = np.asarray([rotate.dot(_dst[i])+tran for i in range(src.shape[0])], dtype=np.float32)

    dst = np.asarray([[0,0], [0,patch_size-1], [patch_size-1,patch_size-1], [patch_size-1, 0]], dtype=np.float32)

    theta, _ = cv2.estimateAffine2D(src, dst)

    patch = cv2.warpAffine(img, theta, (patch_size, patch_size))

    return patch

def get_rotated_patches(img, thresh):
    x_min_margin=10
    x_max_margin=50
    y_min_margin=10
    y_max_margin=50
    patch_size=200
    i_x, i_y = 0, 0
    x_margin, y_margin = x_max_margin, y_max_margin
    
    while True:
    
        angle = np.random.uniform(0.2,1.8) * np.pi

        pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size,img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
               np.random.randint((i_y + 1) * patch_size + 2 * y_margin,img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        
        p1_pos = pos
        i, j = grid[np.random.randint(0, len(grid))]
        p2_pos = [pos[0] + i * np.random.randint(x_min_margin + patch_size, patch_size + x_max_margin),
                  pos[1] + j * np.random.randint(y_min_margin + patch_size, patch_size + y_max_margin)]

        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh=thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p2_thresh=thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
     
        if (cv2.countNonZero(p1_thresh) < 4000) or (cv2.countNonZero(p2_thresh) < 4000):
            continue
        else:
            p2 = sample_rotated_patch(img, p2_pos, angle, patch_size)
            label = 1
            break

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 255, 0),5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 255, 0),5, lineType=cv2.LINE_AA)

        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1  

    return p1, p2, label

def get_backpaired_patches(img, thresh):
    patch_size=200
    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        
        if (cv2.countNonZero(p1_thresh) < 2000) != (cv2.countNonZero(p2_thresh) < 2000):
            label = 1
            break
        else:
            continue

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0),5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255), 5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label

def get_nearby_patches(img,thresh):
    patch_size=200
    x_min_margin=-50
    x_max_margin=0
    y_min_margin=-50
    y_max_margin=0
    i_x, i_y = 0, 0
    x_margin, y_margin = x_max_margin, y_max_margin
    while True:
        pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
               np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]

        p1_pos = pos
        i, j = grid[np.random.randint(0, len(grid))]
        p2_pos = [pos[0] + i * np.random.randint(x_min_margin + patch_size, patch_size + x_max_margin),
                  pos[1] + j * np.random.randint(y_min_margin + patch_size, patch_size + y_max_margin)]

        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        
        if (cv2.countNonZero(p1_thresh) < 4000) or (cv2.countNonZero(p2_thresh) < 4000):
            continue
        else:
            label = 0
            break

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0),5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255),5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1  
    
    return p1, p2, label
def calculate_patch_average(cc_stats, p_unique_labels):
    min_height=30
    max_height=150
    min_width=30
    max_width=150
    h_sum, w_sum = 0, 0
    cc_sum = 0
    for cc_label in p_unique_labels:
        if cc_stats[cc_label][cv2.CC_STAT_HEIGHT]>max_height or cc_stats[cc_label][cv2.CC_STAT_WIDTH]>max_width:
            h_sum+=min(cc_stats[cc_label][cv2.CC_STAT_HEIGHT],max_height)  
            w_sum+=min(cc_stats[cc_label][cv2.CC_STAT_WIDTH],max_width)
            cc_sum+=1
        elif cc_stats[cc_label][cv2.CC_STAT_HEIGHT]<min_height or cc_stats[cc_label][cv2.CC_STAT_WIDTH]<min_width:
            continue
        else:
            h_sum+=cc_stats[cc_label][cv2.CC_STAT_HEIGHT] 
            w_sum+=cc_stats[cc_label][cv2.CC_STAT_WIDTH]
            cc_sum+=1  
    if cc_sum==0:
       cc_sum=1
     
    h_avg=h_sum/cc_sum
    w_avg=w_sum/cc_sum
    return h_avg, w_avg
     
def get_different_size_patches(img, thresh):
    patch_size=200
    epsilon = 0.0001
    ccs = cv2.connectedComponentsWithStatsWithAlgorithm(thresh, 4, cv2.CV_32S, cv2.CCL_GRANA)
    cc_number=ccs[0]
    cc_labels=ccs[1]
    cc_stats=ccs[2]

    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_labels = cc_labels[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_unique_labels = np.unique(p1_labels)[1:]

        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_labels = cc_labels[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_unique_labels = np.unique(p2_labels)[1:]
        
        if (cv2.countNonZero(p1_thresh) < 2000) or (cv2.countNonZero(p2_thresh) < 2000):
            continue

        h1, w1 = calculate_patch_average(cc_stats, p1_unique_labels)
        h2, w2 = calculate_patch_average(cc_stats, p2_unique_labels)
        
        hw1=h1*w1
        hw2=h2*w2

        hw1, hw2 = hw1 + epsilon, hw2 + epsilon
        

        if (min(hw1, hw2)/max(hw1, hw2) < 0.5) :  # todo: improve on this condition
            label = 1
            break
        else:
            continue

    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        for cc_i in range(1,cc_number):
            x, y, w, h = cc_stats[cc_i][cv2.CC_STAT_LEFT], cc_stats[cc_i][cv2.CC_STAT_TOP], cc_stats[cc_i][cv2.CC_STAT_WIDTH], cc_stats[cc_i][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(disp_img, (int(x), int(y)), (int(x + w), int(y + h)), (60, 200, 200), 1, lineType=cv2.LINE_AA)
        for cc_i in p1_unique_labels:
            x, y, w, h = cc_stats[cc_i][cv2.CC_STAT_LEFT], cc_stats[cc_i][cv2.CC_STAT_TOP], cc_stats[cc_i][cv2.CC_STAT_WIDTH], cc_stats[cc_i][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(disp_img, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 3, lineType=cv2.LINE_AA)
        for cc_i in p2_unique_labels:
            x, y, w, h = cc_stats[cc_i][cv2.CC_STAT_LEFT], cc_stats[cc_i][cv2.CC_STAT_TOP], cc_stats[cc_i][cv2.CC_STAT_WIDTH], cc_stats[cc_i][cv2.CC_STAT_HEIGHT]
            cv2.rectangle(disp_img, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 3, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0), 5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255),5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label
def get_different_area_patches(img, thresh):
    patch_size=200
    i_x, i_y = 0, 0
    x_margin, y_margin = 5,5
    
    while True:
        p1_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        p2_pos = [np.random.randint(2 * x_margin + (i_x + 1) * patch_size, img.shape[1] - (i_x + 2) * patch_size - 2 * x_margin),
                  np.random.randint((i_y + 1) * patch_size + 2 * y_margin, img.shape[0] - (i_y + 2) * patch_size - 2 * y_margin)]
        
        p1 = img[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]
        p1_thresh = thresh[p1_pos[1]:p1_pos[1] + patch_size, p1_pos[0]:p1_pos[0] + patch_size]

        p2 = img[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        p2_thresh = thresh[p2_pos[1]:p2_pos[1] + patch_size, p2_pos[0]:p2_pos[0] + patch_size]
        
        a1 = cv2.countNonZero(p1_thresh)
        a2 = cv2.countNonZero(p2_thresh)
        
        if (a1 < 2000) or (a2 < 2000):
            continue


        if (min(a1, a2)/max(a1, a2) < 0.5) : 
            label = 1
            break
        else:
            continue
            
    if SHOW_RESULTS:
        global counter
        disp_img = cv2.cvtColor(thresh.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(disp_img,  (int(p1_pos[0]), int(p1_pos[1])), (int(p1_pos[0] + patch_size), int(p1_pos[1] + patch_size)), (255, 0, 0), 5, lineType=cv2.LINE_AA)
        cv2.rectangle(disp_img, (int(p2_pos[0]), int(p2_pos[1])), (int(p2_pos[0] + patch_size), int(p2_pos[1] + patch_size)), (0, 0, 255),5, lineType=cv2.LINE_AA)
        
        cv2.imwrite('sample_pairs/'+str(counter)+'page'+str(label)+'.png',disp_img)
        cv2.imwrite('sample_pairs/'+str(counter)+'firstpair'+str(label)+'.png',p1)
        cv2.imwrite('sample_pairs/'+str(counter)+'secondpair'+str(label)+'.png',p2)
        counter=counter+1        

    return p1, p2, label


def get_random_pair(images_path):
    images = os.listdir(images_path)
    image_name = np.random.choice(images)
    img = cv2.imread(os.path.join(images_path, image_name), 0)
    ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    gen_func = np.random.choice([get_nearby_patches, get_backpaired_patches, 
                                 get_nearby_patches, get_different_area_patches,
                                 get_nearby_patches, get_different_size_patches,
                                 get_nearby_patches, get_rotated_patches])
    p1, p2, label = gen_func(img,thresh)
    return p1, p2, label


# images_path='complex_test_train'
# for i in range(10):
    # get_random_pair(images_path)
    
    
    