import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model('model.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.allow_custom_ops = True
# converter.experimental_new_converter = True
# converter.target_spec.supported_ops = [
#   tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
#   tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
# ]
tflite_model = converter.convert()
with open("model.tflite", "wb") as fp:
    fp.write(tflite_model)