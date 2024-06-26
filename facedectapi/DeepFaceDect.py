import cv2
from deepface import DeepFace
import os
import PIL as image
import numpy as np

model_name = ["VGG-FACE"]

def get_face_feature(face_pic_path):
    return DeepFace.represent(face_pic_path)

'''
def upload_face_from_user(face_pic,user_name,faceInPic_name):
    # path of save images
    # using user_name and name of picture
    image_save_folder = '../pics/'+user_name+'/'+faceInPic_name+'/'

    # create one folder to save pic
    if not os.path.exists(image_save_folder):
        os.makedirs(image_save_folder)

    # count pic number in folder
    files = os.listdir(image_save_folder)
    file_count = sum(os.path.isfile(os.path.join(image_save_folder, file)) for file in files)
    file_count = file_count+1
    # create name of pic
    saved_image_name = faceInPic_name+str(file_count)+'.jpg'
    # create path to save
    image_path_to_save =  os.path.join(image_save_folder,saved_image_name)
    # use RGB to create pillow pic
    img = image.fromarray(face_pic, 'RGB')
    # save image
    img.save(image_path_to_save)
'''

# new save func
def upload_face_from_user(face_pic,user_name,faceInPic_name):
    image_save_path = '../pics/'+user_name + '/' + faceInPic_name
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    images_in_folder = os.listdir(image_save_path)
    number = len(images_in_folder)+1
    name_of_pic = image_save_path + str(number) + '.jpg'
    cv2.imwrite(name_of_pic, face_pic)

def create_tmp_pic(face_pic, user_name):
    tmp_path = '../tmp/'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)


def is_image(pic):
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tif', '.tiff']
    # 检查文件扩展名是否是图片格式
    return any(pic.lower().endswith(ext) for ext in image_extensions)
def compare_and_verify_face(face_pic, user_name):

    image_folders = [os.path.join('../pics/'+user_name, d) for d in os.listdir('../pics/'+user_name)
                     if os.path.isdir(os.path.join('../pics/'+user_name, d))
                     and any(is_image(f)) for f in os.listdir(os.path.join('../pics/'+user_name, d))]

    for image_folder in image_folders:
        images_in_folder = os.listdir(image_folder)
        for image_in_folder in images_in_folder:
            result = DeepFace.verify(face_pic, os.path.join(image_folder, image_in_folder))
            if result['verified']:
                return os.path.basename(image_folder)
    return 'not found match face'

def get_all_face_in_pic(pic):
    return DeepFace.extract_faces(pic)
