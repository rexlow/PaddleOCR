# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tools.infer.utility import draw_ocr_box_txt
from tools.infer.utility import draw_ocr
from PIL import Image
from ppocr.utils.utility import get_image_file_list
import time
import math
import numpy as np
import copy
import tools.infer.predict_rec as predict_rec
import tools.infer.predict_det as predict_det
from tools.infer.predict_system import sorted_boxes, TextSystem
import cv2
from ppocr.utils.utility import initial_logger
import tools.infer.utility as utility
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

logger = initial_logger()

def main(args):
  image_file_list = get_image_file_list(args.image_dir)
  text_sys = TextSystem(args)
  for image_file in image_file_list:
    img = cv2.imread(image_file)
    if img is None:
      logger.info("error in loading image:{}".format(image_file))
      continue
    starttime = time.time()
    dt_boxes, rec_res = text_sys(img)
    elapse = time.time() - starttime
    print("Predict time of %s: %.3fs" % (image_file, elapse))
    dt_num = len(dt_boxes)
    dt_boxes_final = []
    for dno in range(dt_num):
      text, score = rec_res[dno]
      if score >= 0.5:
        text_str = "%s, %.3f" % (text, score)
        print(text_str)
        dt_boxes_final.append(dt_boxes[dno])


if __name__ == "__main__":
    main(utility.parse_args())
