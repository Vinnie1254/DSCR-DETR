import warnings, os
warnings.filterwarnings('ignore')
from ultralytics import RTDETR

if __name__ == '__main__':
    model = RTDETR('yaml')

    model.train(data='',
                cache=False,
                imgsz=640,
                epochs=1,
                batch=1,
                workers=1,

                project='runs/',
                name='',
                )