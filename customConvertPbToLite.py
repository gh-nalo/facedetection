import tensorflow as tf

MODEL_NAME = "08_ssd_mobilenet_v1"
IMAGE_SHAPE = 300

model = tf.saved_model.load(
    f'./__models/{MODEL_NAME}/saved_model'
)

concrete_func = model.signatures[
    tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

concrete_func.inputs[0].set_shape([1, IMAGE_SHAPE, IMAGE_SHAPE, 3])

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open(f'{MODEL_NAME}_{IMAGE_SHAPE}.tflite', 'wb') as f:
    f.write(tflite_model)
