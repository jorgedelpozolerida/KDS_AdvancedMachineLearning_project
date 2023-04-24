import tensorflow as tf

# FIRST:
# pip install pycuda


print("\nTEST RESULTS:\n\nTensorFlow version:", tf.__version__)

# Try to load the CUDA toolkit version if available
try:
    import pycuda.driver as cuda
    print("\nCUDA version:", ".".join(str(x) for x in cuda.get_version()))
except:
    print("\nCUDA not available")

print('')

# Check if TensorFlow can access a GPU
devices = tf.config.list_physical_devices()

if len(devices) == 0:
    print("No devices found.")
else:
    for device in devices:
        print("Device name:", device.name)
        print("Device type:", device.device_type)
        print()

# Check if TensorFlow can access a GPU
if tf.test.gpu_device_name():
    print('\nGPU device found: {}'.format(tf.test.gpu_device_name()))
else:
    print("\nNo GPU found. Please make sure that TensorFlow is installed with GPU support and that your GPU is properly configured.")
