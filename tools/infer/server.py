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
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

from ppocr.utils.utility import get_image_file_list
import time
import math
import numpy as np
import copy
import cv2
from json import loads
import simplejson as json
from bson import ObjectId
from decimal import Decimal
from ppocr.utils.utility import initial_logger
import tools.infer.utility as utility
import os
import sys
import tornado.web
import tornado.ioloop
from PIL import Image
from io import BytesIO
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '../..')))

logger = initial_logger()
from tools.infer.predict_system import TextSystem, sorted_boxes

text_sys = TextSystem(utility.parse_args())

class JSONEncoder(json.JSONEncoder):

    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, np.ndarray):
            return list(o)
        if isinstance(o, np.float32):
            return float(o)
        if isinstance(o, Decimal):
            return float(o)
        if isinstance(o, np.int64):
            return int(o)
        return json.JSONEncoder.default(self, o)

class MainHandler(tornado.web.RequestHandler):

    def get(self):
        self.write("Hello, world")

    def post(self):
      bytesData = self.request.files['file'][0]['body']
      image = Image.open(BytesIO(bytesData))
      npImage = np.array(image)
      dt_boxes, rec_res = text_sys(npImage)
      self.finish(loads(JSONEncoder().encode({
        "boxes": dt_boxes,
        "data": rec_res
      })))


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    port = 5000
    app = make_app()
    app.listen(port)
    print("Paddle server running at port ", port)
    tornado.ioloop.IOLoop.current().start()
