import multiprocessing


def fun(msg, cls):
    print(msg)


if __name__ == '__main__':
    is_clf = True
    if is_clf:
        from Resnet import ResClassifier
        classifier = ResClassifier()
        classifier.load()
    else:
        classifier = None

    pool = multiprocessing.Pool(processes=4)

    for i in range(4):
        pool.apply_async(fun, (i, classifier, ))
    pool.close()
    pool.join()