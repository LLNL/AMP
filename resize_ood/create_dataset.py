import cv2
from PIL import Image
import glob
import os

PATH_TO_LSUN='path/to/lsun'
SAVE_PATH = 'path/to/save/images'


def run_cv_resize(directory=PATH_TO_LSUN,interp_method='bilinear'):
    '''
    Resizing using OpenCV Image Library
    '''
    save_path = f'{SAVE_PATH}/cv2/{interp_method}'
    os.makedirs(save_path,exist_ok=True)
    filenames = glob.glob(f'{directory}/*.jpg')
    print(f'Using OpenCV for resize for {len(filenames)} files')

    if interp_method == 'bilinear':
        interp = cv2.INTER_LINEAR
    elif interp_method == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interp_method == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interp_method == 'lanczos':
        interp = cv2.INTER_LANCZOS4


    for i, f in enumerate(filenames):
        img = cv2.imread(os.path.join(directory,f))
        img_ = cv2.resize(img,(32,32),interpolation=interp)
        fname  = filenames[i].split('/')[-1]
        if i%100==0:
            print(f'Saving file at {save_path}/cv2_{fname}')

        cv2.imwrite(f'{save_path}/cv2_{fname}',img_)

def run_PIL_resize(directory=PATH_TO_LSUN,interp_method='bilinear'):
    save_path = f'{SAVE_PATH}/pil/{interp_method}'
    os.makedirs(save_path,exist_ok=True)
    filenames = glob.glob(f'{directory}/*.jpg')
    print(f'Using PIL for resize for {len(filenames)} files')

    if interp_method == 'bilinear':
        interp = Image.BILINEAR
    elif interp_method == 'nearest':
        interp = Image.NEAREST
    elif interp_method == 'bicubic':
        interp = Image.CUBIC
    elif interp_method == 'lanczos':
        interp = Image.LANCZOS


    for i, f in enumerate(filenames):
        img = Image.open(os.path.join(directory,f))
        img_ = img.resize((32,32),interp)
        fname  = filenames[i].split('/')[-1]
        if i%100==0:
            print(f'Saving file at {save_path}/pil_{fname}')
        img_.save(f'{save_path}/pil_{fname}')

if __name__=='__main__':
    # run_cv_resize(interp_method='lanczos')
    for interp in ['nearest','bilinear','bicubic','lanczos']:
        run_PIL_resize(interp_method=interp)
