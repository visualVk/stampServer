import tornado.ioloop
import tornado.web
import cv2
import tempfile
import os
import uuid
from tornado_swagger_ui import get_tornado_handler
import numpy as np
import py_eureka_client.eureka_client as eureka_client
from skimage.measure import compare_ssim
import json

SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
# Our API url (can of course be a local resource)
API_URL = 'http://localhost:8888/swagger.json'

swagger_handlers = get_tornado_handler(
    base_url=SWAGGER_URL,
    api_url=API_URL,
    config={
        "app_name": "Test application"
    },
    # oauth_config={ # OAuth config. See https://github.com/swagger-api/swagger-ui#oauth2-configuration .
    # 'clientId': "your-client-id",
    # 'clientSecret': "your-client-secret-if-required",
    # 'realm': "your-realms",
    # 'appName': "your-app-name",
    # 'scopeSeparator': " ",
    # 'additionalQueryStringParams': {'test': "hello"}
    # }
)

def get_real_stamp(file_name):
    files = os.listdir('realStampImage')
    for file in files:
        if not os.path.isdir(file):
            if file_name == file:
                return 'realStampImage/'+file_name
    return None

def get_stamp(file):
    np.set_printoptions(threshold=np.inf)
    # upload_path = os.path.dirname(__file__)
    # filename = file.filename
    file_1 = tempfile.mktemp(suffix=file.filename,prefix=str(uuid.uuid1()))
    f = None
    try:
        f = open(file_1,"wb")
        f.write(file.body)
        image = cv2.imread(file_1)

        hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        low_range = np.array([150, 103, 100])
        high_range = np.array([180, 255, 255])
        th = cv2.inRange(hue_image, low_range, high_range)
        index1 = th == 255

        img = np.zeros(image.shape, np.uint8)
        img[:, :] = (255, 255, 255)
        img[index1] = image[index1]
        return img
    except Exception:
        print(Exception)
    finally:
        if f != None:
            f.close()
            os.remove(file_1)
    return None



def test_compare_pic(imageA):
    imageB = cv2.imread(get_real_stamp("extract_img.png"))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    h1, w1 = grayA.shape
    h2, w2 = grayB.shape
    if grayA.shape < grayB.shape:
        grayA = cv2.resize(grayA, (w2, h2))
    else:
        grayB = cv2.resize(grayB, (w1, h1), interpolation=cv2.INTER_CUBIC)
    score, diff = compare_ssim(grayA, grayB, full=True)
    print("SSIM: {}".format(score))
    return score >= 0.90


class VertifyStampHandler(tornado.web.RequestHandler):
    def post(self):
        file = self.request.files.get("file")[0]
        result_stamp = get_stamp(file)
        result = test_compare_pic(result_stamp)
        self.write(json.dumps({'result': np.bool(result)}))


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.finish("Tornado Swagger UI")


handlers = [
    (r"/", IndexHandler),
    (r"/stamp/verify", VertifyStampHandler)
]

handlers.extend(swagger_handlers)


def make_app():
    return tornado.web.Application(handlers)

if __name__ == "__main__":
    eureka_client.init_registry_client(eureka_server="http://localhost:50101/eureka/,http://localhost:50102/eureka/",
                                       app_name="python-stamp-server",
                                       instance_port=8888)
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.instance().start()
