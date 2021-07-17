import cv2
import pickle
from data_prep import img_prep
import numpy as np

def allowed_file(filename, ALLOWED_EXTENSIONS):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def predict_receipt(file_path, model_path):
    classes = []
    with open('./models/stores_list.txt', 'r') as f:
        for c in f:
            classes.append(c.rstrip())

    with open(model_path, mode ='rb') as fp:
        clf = pickle.load(fp)

    img = cv2.imread(file_path)
    img = img_prep(img, gray_scale=True)
    X_sample = np.array(img)
    X_sample = X_sample.flatten()

    prediction = clf.predict(X_sample.reshape(1, -1))[0]

    return classes[prediction]




# from statistics import mode
# ans = ["ローソン", "ローソン", "セブン", "セブン"]
#
#
# print(mode(ans))

# import numpy as np
# def check(l):
#     arr =np.array([0,0,0])
#     for i in l:
#         if i=="セブン":
#             arr[0]+=1
#         elif i=="ローソン":
#             arr[1]+=1
#         else:
#             arr[2]+=1
#     return np.random.choice([i for i, x in enumerate(arr) if x == max(arr)] ,1)
# l=["セブン","セブン","ローソン","ローソン"]
# check(l)
# print(check(l))
