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
from tornado.options import define, options, parse_command_line
import json

define("port", default=35001, help="port to listen on")
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


def file2pic(file):
    file_1 = tempfile.mktemp(suffix=file.filename, prefix=str(uuid.uuid1()))
    f = None
    try:
        f = open(file_1, "wb")
        f.write(file.body)
        image = cv2.imread(file_1)
        return image
    except Exception:
        print(Exception)
    finally:
        if f is not None:
            f.close()
            os.remove(file_1)
    return None


# 剪裁图片
def cut_image(img_org):
    # 转成hsv
    hue_image = cv2.cvtColor(img_org, cv2.COLOR_BGR2HSV)
    # 红色边界
    low_range = np.array([150, 103, 100])
    high_range = np.array([180, 255, 255])
    th = cv2.inRange(hue_image, low_range, high_range)
    # 二值化
    ret, binary = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    # 膨胀操作
    dilation = cv2.dilate(binary, kernel, iterations=1)
    # 获取外围轮廓
    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    if len(contours) > 0:
        # cv2.boundingRect()返回轮廓矩阵的坐标值，四个值为x, y, w, h， 其中x, y为左上角坐标，w,h为矩阵的宽和高
        boxes = [cv2.boundingRect(c) for c in contours]
        # 获取最外围的坐标信息
        box = boxes[-1]
        x, y, w, h = box
        cut_pic_1 = img_org[y:y + h, x:x + w]
        return cut_pic_1, contours
    return None,None


# 旋转图片
def rotate2upright(original_img, index):
    # original_img = cv2.imread('a2.jpg')
    # 基础处理
    gray_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (9, 9), 0)  # 高斯模糊去噪（设定卷积核大小影响效果）
    _, RedThresh = cv2.threshold(blurred, 165, 255, cv2.THRESH_BINARY)  # 设定阈值165（阈值影响开闭运算效果）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 定义矩形结构元素
    closed = cv2.morphologyEx(RedThresh, cv2.MORPH_CLOSE, kernel)  # 闭运算（链接块）
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)  # 开运算（去噪点）

    image, contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    aim_rect = sorted(contours, key=cv2.contourArea, reverse=True)
    if len(aim_rect) >= index:
        return False
    c = aim_rect[index]
    rect = cv2.minAreaRect(c)
    angle = rect[2]
    box = np.int0(cv2.boxPoints(rect))
    r, c = original_img.shape[:2]
    M = cv2.getRotationMatrix2D((c / 2, r / 2), angle, 1)
    result_img = cv2.warpAffine(original_img, M, (c, r))
    return result_img


# 获取本地样章
def get_real_stamp(file_name):
    files = os.listdir('realStampImage')
    for file in files:
        if not os.path.isdir(file):
            if file_name == file:
                return 'realStampImage/' + file_name
    return None


# 图章提取
def get_stamp(image):
    np.set_printoptions(threshold=np.inf)

    hue_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    low_range = np.array([150, 103, 100])
    high_range = np.array([180, 255, 255])
    th = cv2.inRange(hue_image, low_range, high_range)
    index1 = th == 255

    img = np.zeros(image.shape, np.uint8)
    img[:, :] = (255, 255, 255)
    img[index1] = image[index1]
    return img


# 图章对比
def test_compare_pic(imageA):
    imageB = cv2.imread(get_real_stamp("dilatie_img.png"))

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
    return score >= 0.82


# 图章识别
class VertifyStampHandler(tornado.web.RequestHandler):
    def post(self):
        file = self.request.files.get("file")[0]
        org_image = file2pic(file)
        # cv2.imwrite('org_image.png', org_image)
        no_rotate_img, contours = cut_image(org_image)
        if no_rotate_img is not None:
            # cv2.imwrite('not_rotate_img.png', no_rotate_img)
            not_rotate_result_img = get_stamp(no_rotate_img)
            # cv2.imwrite('not_rotate_result_img.png', not_rotate_result_img)
            not_rotate_result = test_compare_pic(not_rotate_result_img)
            if (not_rotate_result == True):
                self.write(json.dumps({'result': np.bool(not_rotate_result)}))
                return
        for i in range(0, 2):
            rotate2upright_img = rotate2upright(org_image, i)
            if rotate2upright_img == False:
                break
            # cv2.imwrite('rotate.png', rotate2upright_img)
            image, _ = cut_image(rotate2upright_img)
            if image == False:
                break
            # cv2.imwrite('image.png', image)
            result_stamp = get_stamp(image)
            # cv2.imwrite('result.png', result_stamp)
            result = test_compare_pic(result_stamp)
            if result == True:
                self.write(json.dumps({'result': np.bool(result)}))
                return
        self.write(json.dumps({'result': np.bool(False)}))


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
    parse_command_line()
    eureka_client.init_registry_client(eureka_server="http://localhost:50101/eureka/,http://localhost:50102/eureka/",
                                       app_name="python-stamp-server",
                                       instance_port=options.port,
                                       instance_ip="personnel-manage.top")
    app = make_app()
    app.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()
