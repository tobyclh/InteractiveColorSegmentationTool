import cv2
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
import time

import streamlit as st

def main(args):
    # Once we have the dependencies, add a selector for the app mode on the sidebar.

    st.title('Interactive image segmentation')
    images = gather_images(args.folder)
    if len(images) == 0:
        print('No image found in the folder jpg, png, jpeg')
        exit()
    frame_idx = st.sidebar.slider('Frame', min_value=0, max_value=len(images)-1, value=0)
    h_min_slider = st.sidebar.slider('H_l', min_value=0, max_value=360, value=0)
    h_max_slider = st.sidebar.slider('H_u', min_value=0, max_value=360, value=360)
    s_min_slider = st.sidebar.slider('S_l', min_value=0, max_value=100, value=0)
    s_max_slider = st.sidebar.slider('S_u', min_value=0, max_value=100, value=100)
    v_min_slider = st.sidebar.slider('V_l', min_value=0, max_value=100, value=0)
    v_max_slider = st.sidebar.slider('V_u', min_value=0, max_value=100, value=100)
    h_min_slider = min(h_max_slider, h_min_slider)
    s_min_slider = min(s_max_slider, s_min_slider)
    v_min_slider = min(v_max_slider, v_min_slider)
    h_max_slider = max(h_max_slider, h_min_slider)
    s_max_slider = max(s_max_slider, s_min_slider)
    v_max_slider = max(v_max_slider, v_min_slider)
    lower = np.array((h_min_slider, s_min_slider, v_min_slider), dtype=np.float)
    upper = np.array((h_max_slider, s_max_slider, v_max_slider), dtype=np.float)
    _HSV_range = get_hsv_range()
    lower_bound = np.floor(lower * _HSV_range).astype(np.uint8)
    upper_bound = np.ceil(upper * _HSV_range).astype(np.uint8)

    img = get_image(images[frame_idx])
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ~hsv_mask(img_hsv, lower_bound, upper_bound)
    masked = img_rgb.copy()
    masked[mask] = 0
    mask = (mask * 255).astype(np.uint8)[..., None].repeat(3, -1)
    out = np.concatenate((img_rgb, masked, mask), axis=1)

    image_slot = st.image(out)
    save_name = st.sidebar.text_input('Name')
    if st.sidebar.button('Save'):
        out_bound = np.concatenate((lower, upper))
        print(f'{save_name} : {out_bound}')
        return 
    

@st.cache
def get_hsv_range():
    _HSV_range = np.array((0.55, 2.55, 2.55))
    return _HSV_range
    
@st.cache
def get_image(path):
    return cv2.imread(str(path))

@st.cache
def gather_images(input_folder, limit=200):
    folder = Path(input_folder)
    all_images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + list(folder.glob('*.png'))
    if len(all_images) > limit:
        all_images = all_images[:limit]
    return all_images

def hsv_mask(image_hsv, lower_bound, upper_bound):
    mask = cv2.inRange(image_hsv, np.array(lower_bound), np.array(upper_bound)).astype(np.bool)
    return mask



if __name__ == '__main__':
    parser = ArgumentParser('Interactively segment images')
    parser.add_argument('--folder', help='input directory', type=str, default='image')
    args = parser.parse_args()
    main(args)