# I'm not sure why this script works, but it fixes this problem: https://github.com/microsoft/onnxruntime/issues/9760#issue-1053052192
# without causing this int64 error: https://github.com/microsoft/onnxruntime/issues/9760#issuecomment-968967562
# I just commented out all the actual conversion stuff (except  for the model input/output conversion, but that should
# actually be doing anything here anyway) as a way to begin debugging, but then upon testing the output model it actually
# worked fine. So I'm guessing it has something to do with the way onnx adjusts the model when it "re-packages" it. I really
# have no idea though...
# To use this, just open a notebook/Colab and paste these lines:

#   !git clone https://github.com/josephrocca/onnx-typecast
#   %cd onnx-typecast
#   !pip install -r requirements.txt
#   !python3 ./fix-clip-text-vit-32-float32---scratch.py ./clip-text-vit-32-float32.onnx ./clip-text-vit-32-float32-int32.onnx

# Where the `clip-text-vit-32-float32.onnx` file is the file that you get from:
#   1. Exporting CLIP PyTorch model to ONNX using this: https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/Export_CLIP_to_ONNX_tflite_tfjs_tf_saved_model.ipynb
#   2. Then putting the output (clip-text-vit-32.onnx) through this: https://colab.research.google.com/github/josephrocca/openai-clip-js/blob/main/ONNX_float16_to_float32.ipynb

import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto, AttributeProto
from onnx import numpy_helper as nph

import numpy as np
from collections import OrderedDict

from logger import log
import typer


def make_param_dictionary(initializer):
    params = OrderedDict()
    for data in initializer:
        params[data.name] = data
    return params


def convert_params_to_int32(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        # if data.data_type == TensorProto.INT64:
        #     data_cvt = nph.to_array(data).astype(np.int32)
        #     data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params


def convert_constant_nodes_to_int32(nodes):
    """
    convert_constant_nodes_to_int32 Convert Constant nodes to INT32. If a constant node has data type INT64, a new version of the
    node is created with INT32 data type and stored.

    Args:
        nodes (list): list of nodes

    Returns:
        list: list of new nodes all with INT32 constants.
    """
    new_nodes = []
    for node in nodes:
      # if (
      #     node.op_type == "Constant"
      #     and node.attribute[0].t.data_type == TensorProto.INT64
      # ):
      #   data = nph.to_array(node.attribute[0].t).astype(np.int32)
      #   new_t = nph.from_array(data)
      #   new_node = h.make_node(
      #       "Constant",
      #       inputs=[],
      #       outputs=node.output,
      #       name=node.name,
      #       value=new_t,
      #   )
      #   new_nodes += [new_node]
      # else:
      #   new_nodes += [node]

      # I replaced the above code with the line below because the above code does conversions that we don't want. E.g it converts the Constant input the Div_27 here: https://github.com/microsoft/onnxruntime/issues/9760#issuecomment-968967562
      # and that was the cause of the bug in that comment, because the Gather_25 outputs int64 (and I'm not sure how to change that).
      new_nodes += [node]

    return new_nodes


def convert_model_to_int32(model_path: str, out_path: str):
    """
    convert_model_to_int32 Converts ONNX model with INT64 params to INT32 params.\n

    Args:\n
        model_path (str): path to original ONNX model.\n
        out_path (str): path to save converted model.
    """
    log.info("ONNX INT64 --> INT32 Converter")
    log.info(f"Loading Model: {model_path}")
    # * load model.
    model = onnx.load_model(model_path)
    ch.check_model(model)
    # * get model opset version.
    opset_version = model.opset_import[0].version
    graph = model.graph
    # * The initializer holds all non-constant weights.
    init = graph.initializer
    # * collect model params in a dictionary.
    params_dict = make_param_dictionary(init)
    log.info("Converting INT64 model params to INT32...")
    # * convert all INT64 aprams to INT32.
    converted_params = convert_params_to_int32(params_dict)

    log.info("Converting constant INT64 nodes to INT32...")
    new_nodes = convert_constant_nodes_to_int32(graph.node)

    # convert input and output to INT32:
    input_type = graph.input[0].type.tensor_type.elem_type
    output_type = graph.output[0].type.tensor_type.elem_type
    if input_type == TensorProto.INT64:
      graph.input[0].type.tensor_type.elem_type = TensorProto.INT32
    if output_type == TensorProto.INT64:
      graph.output[0].type.tensor_type.elem_type = TensorProto.INT32

    # convert node attributes to INT32:
    # for node in new_nodes:
      # if node.name == "Gather_25":
      #   print(node)
      #   print(node.attribute[0].t.data_type)
      #   print("\n\n")

      # if node.name == "Div_27":
      #   print(node)
      #   print("\n\n")

      # if node.name == "Constant_26":
      #   print(node)
      #   print(TensorProto.INT64)
      #   print(TensorProto.INT32)
      #   print("\n\n")

      # if node.op_type == "Constant":
      #   print(node)
      #   print("\n\n")

      # for index, attribute in enumerate(node.attribute):
        # if attribute.name == "to" and attribute.i == TensorProto.INT64:  # for op_type=="Cast"
        #   attribute.i = TensorProto.INT32
          
        # if hasattr(attribute, "type"):
        #   if attribute.type == AttributeProto.TENSOR:
        #     if attribute.t.data_type == TensorProto.INT64:
        #       attribute.t.CopyFrom( nph.from_array( nph.to_array(attribute.t).astype(np.int32) ) )

    graph_name = f"{graph.name}-int32"
    log.info("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_int32 = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    log.info("Creating new int32 model...")
    model_int32 = h.make_model(graph_int32, producer_name="onnx-typecast")
    model_int32.opset_import[0].version = opset_version
    ch.check_model(model_int32)
    log.info(f"Saving converted model as: {out_path}")
    onnx.save_model(model_int32, out_path)
    log.info(f"Done Done London. 🎉")
    return


if __name__ == "__main__":
    typer.run(convert_model_to_int32)
