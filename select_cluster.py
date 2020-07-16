import cv2
import numpy as np
import os

def select_cluster(org_folder,vis_folder,page_name,maintext_folder,overlap_folder):
    vis_img=cv2.imread(os.path.join(vis_folder,page_name))
    org_img=cv2.imread(os.path.join(org_folder,page_name),0)
    ret, thresh = cv2.threshold(org_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows,cols,ch=vis_img.shape 
    red=vis_img[:,:,2]
    green=vis_img[:,:,1]
    blue=vis_img[:,:,0]
    
    # book1_page1.png
    # maintext=np.zeros((rows,cols),dtype=np.uint8)
    # s=(blue<18)
    # maintext[s]=255
    # cv2.imwrite(os.path.join(maintext_folder,page_name),maintext)
    # cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

  
    # book1_page11.png
    # maintext=np.zeros((rows,cols),dtype=np.uint8)
    # s=(blue<19)&(green>37)
    # maintext[s]=255
    # cv2.imwrite(os.path.join(maintext_folder,page_name),maintext)
    # cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

    
    # book1_page16.png
    # maintext=np.zeros((rows,cols),dtype=np.uint8)
    # s=(blue<29)
    # maintext[s]=255
    # cv2.imwrite(os.path.join(maintext_folder,page_name),maintext)
    # cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)
    
    # book1_page18.png
    # maintext=np.zeros((rows,cols),dtype=np.uint8)
    # s=(blue<118)
    # maintext[s]=255
    # cv2.imwrite(os.path.join(maintext_folder,page_name),maintext)
    # cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

    # book1_page7.png
    # maintext=np.zeros((rows,cols),dtype=np.uint8)
    # s=(blue<59)
    # maintext[s]=255
    # cv2.imwrite(os.path.join(maintext_folder,page_name),maintext)
    # cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

    # book1_page8.png
    maintext=np.zeros((rows,cols),dtype=np.uint8)
    s=(blue<59)
    maintext[s]=255
    cv2.imwrite(os.path.join(maintext_folder,page_name[:-4]+'.png'),maintext)
    cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

    # book2_page2.png
#    maintext=np.zeros((rows,cols),dtype=np.uint8)
#    s=(blue<220)
#    maintext[s]=255
#    cv2.imwrite(os.path.join(maintext_folder,page_name[:-4]+'.png'),maintext)
#    cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

    # book3_page1.jpg
#    maintext=np.zeros((rows,cols),dtype=np.uint8)
#    s=(blue<20)&(green>40)
#    maintext[s]=255
#    cv2.imwrite(os.path.join(maintext_folder,page_name[:-4]+'.png'),maintext)
#    cv2.imwrite(os.path.join(overlap_folder,page_name),maintext&thresh)

vis_folder='output_10_10/cv2_vis2'
org_folder='complex_test'
maintext_folder='complex_maintext_4'
overlap_folder='complex_maintext_overlap_4'
#page_name='book1_page1.png'
#page_name='book1_page11.png'
#page_name='book1_page16.png'
#page_name='book1_page18.png'
#page_name='book1_page7.png'
page_name='book1_page8.jpg'
#page_name='book2_page2.jpg'
#page_name='book3_page1.jpg'


select_cluster(org_folder,vis_folder,page_name,maintext_folder,overlap_folder)



