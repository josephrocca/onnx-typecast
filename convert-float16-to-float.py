import onnx

from onnx import helper as h
from onnx import checker as ch
from onnx import TensorProto, GraphProto
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


def convert_params_to_float(params_dict):
    converted_params = []
    for param in params_dict:
        data = params_dict[param]
        if data.data_type == TensorProto.FLOAT16:
            data_cvt = nph.to_array(data).astype(np.float)
            data = nph.from_array(data_cvt, data.name)
        converted_params += [data]
    return converted_params


def convert_constant_nodes_to_float(nodes):
    """
    convert_constant_nodes_to_float Convert Constant nodes to FLOAT. If a constant node has data type FLOAT16, a new version of the
    node is created with FLOAT data type and stored.

    Args:
        nodes (list): list of nodes

    Returns:
        list: list of new nodes all with FLOAT constants.
    """
    new_nodes = []
    for node in nodes:
        if (
            node.op_type == "Constant"
            and node.attribute[0].t.data_type == TensorProto.FLOAT16
        ):
            data = nph.to_array(node.attribute[0].t).astype(np.float)
            new_t = nph.from_array(data)
            new_node = h.make_node(
                "Constant",
                inputs=[],
                outputs=node.output,
                name=node.name,
                value=new_t,
            )
            new_nodes += [new_node]
        else:
            new_nodes += [node]

    return new_nodes


def convert_model_to_float(model_path: str, out_path: str):
    """
    convert_model_to_float Converts ONNX model with FLOAT16 params to FLOAT params.\n

    Args:\n
        model_path (str): path to original ONNX model.\n
        out_path (str): path to save converted model.
    """
    log.info("ONNX FLOAT16 --> FLOAT Converter")
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
    log.info("Converting FLOAT16 model params to FLOAT...")
    # * convert all FLOAT16 aprams to FLOAT.
    converted_params = convert_params_to_float(params_dict)
    log.info("Converting constant FLOAT16 nodes to FLOAT...")
    new_nodes = convert_constant_nodes_to_float(graph.node)

    graph_name = f"{graph.name}-float"
    log.info("Creating new graph...")
    # * create a new graph with converted params and new nodes.
    graph_float = h.make_graph(
        new_nodes,
        graph_name,
        graph.input,
        graph.output,
        initializer=converted_params,
    )
    log.info("Creating new float model...")
    model_float = h.make_model(graph_float, producer_name="onnx-typecast")
    model_float.opset_import[0].version = opset_version
    ch.check_model(model_float)
    log.info(f"Saving converted model as: {out_path}")
    onnx.save_model(model_float, out_path)
    log.info(f"Done Done London. 🎉")
    return


if __name__ == "__main__":
    typer.run(convert_model_to_float)
