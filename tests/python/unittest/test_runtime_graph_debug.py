import os
import tvm
import numpy as np
import json
from tvm.contrib.debugger import debug_runtime as graph_runtime
def test_graph_simple():
    n = 4
    A = tvm.placeholder((n,), name='A')
    B = tvm.compute(A.shape, lambda *i: A(*i) + 1.0, name='B')
    s = tvm.create_schedule(B.op)

    node0 = {"op": "null", "name": "x", "inputs": []}
    node1 = {"op": "tvm_op", "name": "add",
             "inputs": [[0, 0, 0]],
             "attrs": {"func_name": "myadd",
                       "flatten_data": "1",
                       "num_inputs" : "1",
                    "num_outputs" : "1"}}
    nodes = [node0, node1]
    arg_nodes = [0]
    node_row_ptr = [0, 1, 2]
    outputs = [[1, 0, 0]]
    shape = (4,)
    attrs = {
        "shape" : ["list_shape", [shape, shape]],
        "dltype" : ["list_str", ["float32", "float32"]],
        "storage_id" : ["list_int", [0, 1]],
    }
    graph = {"nodes": nodes,
             "arg_nodes": arg_nodes,
             "node_row_ptr": node_row_ptr,
             "heads": outputs,
             "attrs": attrs}
    graph = json.dumps(graph)

    def check_verify():
        if not tvm.module.enabled("llvm"):
            print("Skip because llvm is not enabled")
            return
        mlib = tvm.build(s, [A, B], "llvm", name="myadd")
        mod = graph_runtime.create(graph, mlib, tvm.cpu(0))
        a = np.random.uniform(size=(n,)).astype(A.dtype)
        mod.set_input(x=a)
        #verify dumproot created
        path = mod.ui_obj.curses_obj._dump_root + mod.ui_obj.curses_obj.dump_folder()
        directory = os.path.dirname(path)
        assert(os.path.exists(directory))
        #verify graph is there
        assert(len(os.listdir(directory)) > 0)
        #verify dump root delete after cleanup
        mod.ui_obj.curses_obj.exit()
        assert(not os.path.exists(directory))

    check_verify()

if __name__ == "__main__":
    test_graph_simple()
