from tensorflow.keras.models import load_model
from tinymlgen import port

model = load_model('model.h5')
c_code = port(model, optimize=False, pretty_print=True)
text_file = open("ECGModel.h", "w")
n = text_file.write(c_code)
text_file.close()