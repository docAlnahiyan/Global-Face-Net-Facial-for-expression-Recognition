# GlobalFaceNet: facial expression recognition

GlobalFaceNet is a CNN + Vision Transformer hybrid for recognising emotion from a single face image (unimodal). A convolutional path captures fine-grained facial detail; its feature maps are split into patches and passed through Transformer blocks (multi-head self-attention) to model global spatial context, and the two are combined before classification.

It is trained and evaluated on the [RAF-DB](http://www.whdeng.cn/raf/model1.html) dataset across seven emotions: surprise, fear, disgust, happy, sad, angry and neutral. Reported result: **85.0% accuracy** with about **3.2 ms** inference per image.

More about this project: [docalnahiyan.github.io/projects/globalfacenet](https://docalnahiyan.github.io/projects/globalfacenet/).

## Files

| File | Purpose |
|---|---|
| `global_facenet.py` | The full pipeline, exported from a Kaggle notebook: data loading, class balancing and augmentation, the GlobalFaceNet model, training, and evaluation plots |

The original notebook is on Kaggle: [kaggle.com/code/docnahiyan/globalfacenet](https://www.kaggle.com/code/docnahiyan/globalfacenet).

## Data

RAF-DB is available to researchers on request from its authors and is not included here.

## Requirements

Python 3 with TensorFlow/Keras, OpenCV, scikit-learn, NumPy, pandas, Matplotlib, seaborn and tqdm.
