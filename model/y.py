import logging
if __name__ == '__main__':
    logger = logging.getLogger("main-logger")
    logger.setLevel(logging.INFO)

    # 移除默认的处理程序
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # 添加自定义的处理程序
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    logger.info("123344")