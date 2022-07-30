import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import numpy as np

def getPrediction(filename):
    model1 = load_model('model_saved.h5')
    image = load_img('static/images/'+filename, target_size=(224, 224))
    img = np.array(image)
    img = img / 255.0
    img = img.reshape(1,224,224,3)
    label = (model1.predict(img) > 0.5).astype('int32')
    result="Pneumonia" if label else "Normal"
    print(result)
    return result

