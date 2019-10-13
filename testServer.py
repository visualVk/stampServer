import tornado.web
import tornado.ioloop
import os

abs_path = os.path.dirname(os.path.abspath("__file__"))
print(abs_path)


class IndexHandler(tornado.web.RequestHandler):
    def get(self, *args, **kwargs):
        para = self.get_argument('test', None)
        self.write("test tornado {}".format(para))


if __name__ == '__main__':
    app = tornado.web.Application([
        (r'/GET', IndexHandler)
    ],
        static_path=os.path.join(abs_path, 'static')
    )
    # 监听端口
    app.listen(8000)
    tornado.ioloop.IOLoop.current().start()
