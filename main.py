import streamlit as st
import time 
import base64
import math
import requests
from io import BytesIO
import numpy as np
import cv2
import gc
import os
import pandas as pd
from collections import defaultdict
from utils import *
import time
import io
from PIL import Image, ImageOps

def read_img(img_path):
  return cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)

def draw_masks_fromList(
    image, chosen_index, contours, origin_size_mask,
    labels, colors, alpha = 0.4,
    contour_color = (0,0,0), contour_line_weight = 3):
  masked_image = image.copy()
  contour_list = []
  for i, mask_index in enumerate(chosen_index):
    contour = contours[mask_index]
    contour_list.append(contour)
    mask = cv2.drawContours(np.zeros(origin_size_mask), [contour], -1, (255), -1)
    # mask[offset_masks[i][0]:offset_masks[i][1],...] = masks_generated[i]

    if mask.shape[0]!= image.shape[0] or mask.shape[1]!= image.shape[1]:
      mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    # bbox, max_contour = mask_to_bbox(mask, return_contour=True)
    # contour_list.append(max_contour)

    masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                            np.asarray(colors[int(labels[i][-1])], dtype='uint8'),
                            masked_image)

    masked_image = masked_image.astype(np.uint8)
    del mask

  gc.collect()

  image = cv2.addWeighted(image, alpha, masked_image, (1-alpha), 0)
  image = cv2.drawContours(image, contour_list, -1, contour_color, contour_line_weight)

  return image


# =====================================================
st.set_page_config(layout='wide', page_title='Detect Anything')

if "server_url" not in st.session_state:
  st.session_state["server_url"] = st.secrets["server_url"]

with st.sidebar:
    server_url = st.text_input("Server URL",value="")
    set_button = st.button("Apply")

if set_button and server_url!="":
    st.session_state["server_url"] = server_url
    st.toast('Successfully Apply')
    time.sleep(.5)

if 'sample_folder' not in st.session_state:
    st.session_state["sample_folder"] = os.getenv("REF_FOLDER",".")

with st.expander("Send Request"):
    tab1, tab2 = st.tabs(["Object Detection", "DB Functions"])
    with tab1:
        with st.form("detect_image"):
            st.title("Detect Image")
            query_img = st.file_uploader("Image",key="query_img_key")
            if query_img is not None:
                st.image(query_img.getvalue())
                # with open(os.path.join(st.session_state["sample_folder"], "query.jpg"), "wb+") as fh:
                #     fh.write(query_img.getvalue())

            # Every form must have a submit button.
            detect_submitted = st.form_submit_button("Detect")
    with tab2:
        reset_button = st.button("Reset DB")
        init_button = st.button("Initialize DB")
        update_button = st.button("Update DB")


if init_button:
    response = requests.post(
           f'{st.session_state["server_url"]}/init_db',
            headers = {
                'ngrok-skip-browser-warning': '1'
            },
            timeout=120,
        )
    status = response.json()
    if "status" in status:
        st.toast('Initialize DB Successfully', icon='🎉')
    elif "error" in status:
        st.toast(f'Error in initializing db: {status["error"]}', icon="🥞")

if reset_button:
    
    response = requests.post(
           f'{st.session_state["server_url"]}/reset_db',
            headers = {
                'ngrok-skip-browser-warning': '1'
            },
            timeout=120,
        )
    status = response.json()
    if "status" in status:
        st.toast('Reset DB Successfully', icon='🎉')
    elif "error" in status:
        st.toast(f'Error in reseting db: {status["error"]}', icon="🥞")

if update_button:
    response = requests.post(
           f'{st.session_state["server_url"]}/update_db',
            headers = {
                'ngrok-skip-browser-warning': '1'
            },
            timeout=120,
        )
    status = response.json()
    if "status" in status:
        st.toast('Update DB Successfully', icon='🎉')
    elif "error" in status:
        st.toast(f'Error in updating db: {status["error"]}', icon="🥞")


allow_show_img = False

if detect_submitted:

    start = time.time()
    with st.status("Processing data data...") as status:
        query_img_encode = base64.b64encode(query_img.getvalue()).decode('utf-8')
        status.update(
            label="Send request to AI", state="running", expanded=False
        )

        data = {
            'query_image':query_img_encode,
        }

        status.update(
            label="AI is working", state="running", expanded=False
        )
        st.session_state["image"] = query_img_encode
        start = time.time()
        response = requests.post(
           f'{st.session_state["server_url"]}/classify',
            json = data,
            headers = {
                'ngrok-skip-browser-warning': '1'
            },
            timeout=2000,
        )

        return_data = response.json()
        st.session_state["reptime"] = time.time()-start
        
        
        during = time.time()-start
        if 'return_data' not in st.session_state:
            st.session_state["return_data"] = return_data['query_image']
            st.session_state["return_data"]['during'] = during
        else:
            st.session_state["return_data"] = return_data['query_image']
            st.session_state["return_data"]['during'] = during

        status.update(
            label="Processing complete!", state="complete", expanded=False
        )

        allow_show_img = True
            
    

if st.session_state.get("return_data") is not None:
    with st.expander("Postproccess"):
        valid_conf = st.slider("Valid Confident Score", 0.0, 1.0, 0.5, 0.01)
        minimum_conf = st.slider("Minimum Confident Score", 0.0, 1.0, 0.3, 0.01)
        assert minimum_conf < valid_conf, "Minimum Confident Score < Valid Confident Score"
        if st.button("Show Image"):
            allow_show_img = True

if allow_show_img:
    allow_show_img = False
    return_data = st.session_state.get("return_data")

    labels = return_data['label']
    conf = np.array(return_data['conf'])
    bbox = return_data['bbox']
    img_shape = return_data['img_shape']

    index_valid = [idx for idx, valid in zip(range(len(labels)), conf > minimum_conf) if valid]

    valid_labels = [ labels[idx] for idx in index_valid]
    valid_conf = [ conf[idx] for idx in index_valid]
    valid_bbox = [ bbox[idx] for idx in index_valid]

    # """
    #     {
    #         class_name:{
    #             best_entity:0,
    #             entities:[
    #                 {
    #                     "conf":...,
    #                     "bbox":...,
    #                 }
    #             ]
    #         }
    #     }
    # """
    pack = lambda conf, bbox: {
                "conf":conf,
                "bbox":bbox,
            }
    dataware = defaultdict(lambda:{
        "best_entity":-1,
        "entities":[]
    })

    for label, conf, bbox in zip(valid_labels, valid_conf, valid_bbox):

        curr_idx = len(dataware[label]["entities"])-1
        best_idx = dataware[label]["best_entity"]
        if best_idx < 0:
            best_idx = 0
        else:
            entity = dataware[label]["entities"][best_idx]
            if conf > entity["conf"]:
                best_idx = curr_idx

        dataware[label]["best_entity"] = best_idx
        dataware[label]["entities"].append(pack(conf, bbox))

    dataframe = {"Class":[], "Image":[], "Num":[]}
    # image = read_img(os.path.join(st.session_state["sample_folder"],'query.jpg'))
    img_bytes = base64.b64decode(st.session_state["image"])
 
    image =  Image.open(io.BytesIO(img_bytes))
    image = ImageOps.exif_transpose(image)
    image = np.array(image)
    current_shape = image.shape[:-1]


    # outobjs = image.copy()
    num_class = 0
    for class_name, vals in dataware.items():
        num_class +=1
        dataframe["Class"].append(class_name)
        num = len(vals["entities"])
        dataframe["Num"].append(num)

        if vals["best_entity"] > -1:
            best_entity = vals["entities"][vals["best_entity"]]

            # x1,y1,x2,y2 = convert_bbox(*best_entity["bbox"], *img_shape, *current_shape)
            x1,y1,x2,y2 = convert_bbox(*best_entity["bbox"], current_shape[1], current_shape[0])
            # print(f"current_shape: {current_shape}")
            max_shape = 60
            ratio = min(max_shape/(x2-x1), max_shape/(y2-y1))
            # print(f"xywhn: {best_entity['bbox']}")
            # print(f"xyxy: {(x1,y1,x2,y2)}")
            # print(f"src_shape: {image.shape}")
            # print(f"obj shape: {image[y1:y2, x1:x2, :].shape}")
            # cv2.rectangle(outobjs, (x1, y1), (x2, y2), (0, 255, 0), 2)
            img = cv2.resize(image[y1:y2, x1:x2, :], None, fx=ratio, fy=ratio)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dataframe["Image"].append(numpy_to_base64_cv2(img))

    # cv2.imwrite("/app/src/outobs.png", outobjs)
    col1, col2 = st.columns([2,5])
    num_col = min(num_class, 5)
    num_row = math.ceil(num_class/num_col) if num_col > 0 else 0

    # max_shape = 2000
    # ratio = min(max_shape/current_shape[0], max_shape/current_shape[1])
    # main_img = cv2.resize(image.copy(), None, fx=ratio, fy=ratio)
    main_img = image
    print(f"image shape: {image.shape}")
    output_image, color_mapping = draw_bounding_boxes(main_img, valid_bbox, valid_labels, img_shape, 0.8)
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    
    with col1:
        st.image(numpy_to_base64_cv2(output_image))

    with col2:
        st.metric("Response Time", st.session_state.get("reptime",-1))
        with st.expander("Class"):
            for row_idx in range(num_row):
                cols = st.columns(num_col)
                for col_idx in range(num_col):
                    index = (row_idx * num_col + col_idx)
                    if index >= num_class: continue
                    space = cols[col_idx].container(height = 200, border =True)
                    with space:
                        color = rgb_to_hex(*color_mapping[dataframe['Class'][index]])
                        st.color_picker(f"{dataframe['Class'][index]}: {dataframe['Num'][index]}", color, disabled =True, key=f"c{index}{color}{time.time()}")
                        st.image(dataframe['Image'][index])
    
    # st.json(dataware)

