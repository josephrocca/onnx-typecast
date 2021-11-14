A fork of this: https://github.com/aadhithya/onnx-typecast I've added a script to convert ONNX float16 models to float32, since [ONNX Runtime Web](https://github.com/microsoft/onnxruntime/tree/master/js/web) doesn't currently support float16. It's very haphazard and I've only tested it with one model. If it misses some nodes, the ONNX runtime should tell you their names (in the error message), and then you can open your model file in [netron.app](https://netron.app/) to investigate. You'll want to add another `if` statement somewhere around [here](https://github.com/josephrocca/onnx-typecast/blob/fc1173d5a1755ad2ee4bc102b4963c154000444c/convert-float16-to-float32.py#L102) to handle the currently-unhandled node.

---

# onnx-typecast
A simple python script to typecast ONNX model parameters from INT64 to INT32, and FLOAT16 to FLOAT(32).

## Why?
I wanted to play around with [ONNX.js](https://github.com/microsoft/onnxjs) and soon figured out that it doesn't support onnx models with INT64 parameters. Also, OpenCV doesn't seem to support INT64 parameters.

## What does this script do?
- The script goes through the parameters of each node of the onnx model and blindly converts INT64 parameters to INT32.
- It also converts the constant parameters from INT64 to INT32.
- Finally, creates a new model and saves it.

## What's the catch?
- **The script does not handle overflows and blindly converts all INT64 parameters to INT32.**
- So ops that require `>INT32.max` or `<INT32.min` values may not perform as expected.
- Please feel free to modify the script to account for this.

## Alright. How do I use it?
 - simple.
 - Install requirements: `pip install -r requirements.txt`
 - run: `python convert-int64-to-int32.py path/to/int64_model.onnx path/to/converted_int32_model.onnx`
 - or: `python convert-float16-to-float32.py path/to/float16_model.onnx path/to/converted_float32_model.onnx`


## Also Checkout
- https://github.com/microsoft/onnxjs/issues/168#issuecomment-727219050
