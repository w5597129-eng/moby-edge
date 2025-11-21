import onnx
import onnxruntime as ort
import numpy as np

onnx_path = 'models/mlp_classifier.onnx'
print('Loading', onnx_path)
model = onnx.load(onnx_path)
opset = None
for ops in model.opset_import:
    print('opset domain:', ops.domain, 'version:', ops.version)
    if ops.domain == '' or ops.domain == 'ai.onnx':
        opset = ops.version
print('Detected opset:', opset)

sess = ort.InferenceSession(onnx_path)
inputs = sess.get_inputs()
outputs = sess.get_outputs()
print('Inputs:')
for i in inputs:
    print(' ', i.name, 'shape', i.shape, 'type', i.type)
print('Outputs:')
for o in outputs:
    print(' ', o.name, 'shape', o.shape, 'type', o.type)

# Try to infer input size from ONNX graph if possible
try:
    shape = model.graph.input[0].type.tensor_type.shape
    dim = None
    if len(shape.dim) >= 2:
        last = shape.dim[-1]
        if last.dim_value > 0:
            dim = last.dim_value
    if dim is None:
        # fallback to output's last dim if available
        if len(model.graph.output) > 0:
            outshape = model.graph.output[0].type.tensor_type.shape
            if len(outshape.dim) >= 2 and outshape.dim[-1].dim_value > 0:
                dim = outshape.dim[-1].dim_value
    if dim is None:
        dim = 111
    print('Using input dim =', dim)
    x = np.random.randn(1, dim).astype(np.float32)
    res = sess.run([outputs[0].name], {inputs[0].name: x})
    print('Run OK, output shape:', np.asarray(res).shape)
except Exception as e:
    print('Runtime run failed:', e)
