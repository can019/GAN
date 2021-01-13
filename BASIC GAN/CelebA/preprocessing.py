import os
import zipfile
import numpy as np
import sys
import cv2
import tqdm
from sklearn.utils import shuffle

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 100):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


class PreProcessor:
    target_root = None
    target_zip = None
    target_file = None

    def __init__(self):
        self.target_root = """.\\dataset"""
        self.target_zip = """archive.zip"""

    def run(self):
        self.unzip()
        imgs = self.ready_data()  # 218*178
        x_train = imgs[:8000].astype('float32')/255.
        x_test = imgs[8000:].astype('float32')/255.
        print(np.shape(x_test))
        # x_valid =
        y_train = np.ones([np.size(x_train, axis=0),1]).astype('uint8')
        y_test = np.ones([np.size(x_test, axis=0),1]).astype('uint8')
        # y_valid =
        return x_train, x_test, y_train, y_test

    def unzip(self):
        if not os.path.exists(self.target_root):
            raise Exception("No such dir name '{}'".format(self.target_root))
        img_folder = os.path.join(self.target_root, "img")
        if not os.path.exists(img_folder):
            target_zip = os.path.join(self.target_root, self.target_zip)
            if not os.path.exists(target_zip):
                raise Exception("No such file name '{}'".format(target_zip))
            os.mkdir(img_folder)
            with zipfile.ZipFile(target_zip) as zf:
                for member in tqdm.tqdm(zf.infolist()[:10000], desc='Extracting '):
                    try:
                        zf.extract(member, img_folder)
                    except zipfile.error as e:
                        print(e)
                        pass

    def ready_data(self):
        target_folder = os.path.join(self.target_root, "img", "img_align_celeba", "img_align_celeba")
        if not os.path.exists(target_folder):
            raise Exception("Exception on line 34 in preprocessing.py No such file name '{}'".format(target_folder))
        imgs = []
        lists = os.listdir(target_folder)
        print("shuffle...")
        lists = shuffle(lists[:10000])
        print("done")
        i = 0
        print("read imgs...")
        for file in lists:
            i += 1
            current_file_name = os.path.join(target_folder, file)
            img = cv2.imread(current_file_name, cv2.IMREAD_GRAYSCALE)
            imgs.append(cv2.resize(img, (256, 256)))
            printProgress(i, len(lists))
        print("done")
        return np.array(imgs)
g