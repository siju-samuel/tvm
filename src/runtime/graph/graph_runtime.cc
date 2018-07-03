/*!
 *  Copyright (c) 2017 by Contributors
 * \file graph_runtime.cc
 */
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <numeric>
#include "./graph_runtime.h"

namespace tvm {
namespace runtime {


/*!
 * \brief Tiny graph runtime.
 *
 *  This runtime can be acccesibly in various language via
 *  TVM runtime PackedFunc API.
 */
void GraphRuntime::Run() {
  // setup the array and requirements.
  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
  }
}

/*!
 * \brief Initialize the graph executor with graph and context.
 * \param graph_json The execution graph.
 * \param module The module containing the compiled functions.
 * \param ctx The context where the graph should sit on
 */
void GraphRuntime::Init(const std::string& graph_json,
          tvm::runtime::Module module,
          TVMContext ctx) {
#ifndef _LIBCPP_SGX_NO_IOSTREAMS
  std::istringstream is(graph_json);
#else
  std::string is = graph_json;
#endif
  dmlc::JSONReader reader(&is);
  this->Load(&reader);
  module_ = module;
  ctx_ = ctx;
  this->SetupStorage();
  this->SetupOpExecs();
}
/*!
 * \brief Get the input index given the name of input.
 * \param name The name of the input.
 * \return The index of input.
 */
int GraphRuntime::GetInputIndex(const std::string& name) {
  for (size_t i = 0; i< input_nodes_.size(); ++i) {
    uint32_t nid = input_nodes_[i];
    if (nodes_[nid].name == name) {
      return static_cast<int>(i);
    }
  }
  LOG(WARNING) << "Warning: cannot find \"" << name << "\" among input";
  return -1;
}

/*!
 * \brief set index-th input to the graph.
 * \param index The input index.
 * \param data_in The input data.
 */
void GraphRuntime::SetInput(int index, DLTensor* data_in) {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->GetEntryId(input_nodes_[index], 0);
  TVM_CCALL(TVMArrayCopyFromTo(data_in, &data_entry_[eid], nullptr));
}

/*!
 * \brief Copy index-th input to data_out
 * \param index The input index.
 * \param data_out The output
 */
void GraphRuntime::GetInput(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), input_nodes_.size());
  uint32_t eid = this->GetEntryId(input_nodes_[index], 0);
  TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
}

/*!
 * \brief Copy index-th output to data_out.
 * \param index The output index.
 * \param data_out the output data.
 */
void GraphRuntime::GetOutput(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), outputs_.size());
  uint32_t eid = this->entry_id(outputs_[index]);
  TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
}
#ifdef TVM_GRAPH_RUNTIME_DEBUG
/*!
 * \brief Get the node index given the name of node.
 * \param name The name of the node.
 * \return The index of node.
 */
int GraphRuntime::GetNodeIndex(const std::string& name) {
  for (uint32_t nid = 0; nid< nodes_.size(); ++nid) {
    if (nodes_[nid].name == name) {
      return static_cast<int>(nid);
    }
  }
  LOG(FATAL) << "cannot find " << name << " among nodex";
  return -1;
}

/*!
 * \brief Copy index-th node to data_out.
 *
 * This method will do a partial run of the the graph
 * from begining upto the index-th node and return output of index-th node.
 * This is costly operation and suggest to use only for debug porpose.
 *
 * \param index: The  index of the node.
 * \param data_out the node data.
 */
void GraphRuntime::DebugGetNodeOutput(int index, DLTensor* data_out) {
  CHECK_LT(static_cast<size_t>(index), nodes_.size());
  uint32_t eid = index;

  for (size_t i = 0; i < op_execs_.size(); ++i) {
    if (op_execs_[i]) op_execs_[i]();
    if (static_cast<int>(i) == index) break;
  }

  TVM_CCALL(TVMArrayCopyFromTo(&data_entry_[eid], data_out, nullptr));
}
#endif

/*!
 * \brief Load parameters from parameter blob.
 * \param param_blob A binary blob of parameter.
 */
void GraphRuntime::LoadParams(const std::string& param_blob) {
  dmlc::MemoryStringStream strm(const_cast<std::string*>(&param_blob));
  this->LoadParams(&strm);
}

void GraphRuntime::LoadDLTensor(dmlc::Stream* strm, DLTensor* dst) {
  // always use strm->Read to maintain endianness conversion
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&reserved))
      << "Invalid DLTensor file format";
  CHECK(header == kTVMNDArrayMagic)
      << "Invalid DLTensor file format";

  DLTensor tensor;
  CHECK(strm->Read(&(tensor.ctx)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&(tensor.ndim)))
      << "Invalid DLTensor file format";
  CHECK(strm->Read(&(tensor.dtype)))
      << "Invalid DLTensor file format";
  std::vector<int64_t> shape(tensor.ndim);
  if (tensor.ndim != 0) {
    CHECK(strm->ReadArray(&shape[0], tensor.ndim))
        << "Invalid DLTensor file format";
  }
  CHECK_EQ(tensor.ndim, dst->ndim) << "param dimension mismatch";
  CHECK(tensor.dtype.bits == dst->dtype.bits &&
        tensor.dtype.code == dst->dtype.code &&
        tensor.dtype.lanes == dst->dtype.lanes) << "param type mismatch";
  for (int i = 0; i < tensor.ndim; ++i) {
    CHECK_EQ(shape[i], dst->shape[i]) << "param shape mismatch";
  }
  size_t bits = dst->dtype.bits * dst->dtype.lanes;
  size_t elem_bytes = (bits + 7) / 8;
  size_t num_elems = 1;
  for (int i = 0; i < dst->ndim; ++i) {
    num_elems *= dst->shape[i];
  }
  uint64_t data_byte_size;
  CHECK(strm->Read(&data_byte_size))
      << "Invalid DLTensor file format";
  CHECK_EQ(data_byte_size, elem_bytes * num_elems)
      << "Invalid DLTensor file format";
  std::vector<uint8_t> bytes(data_byte_size + 1);
  CHECK(strm->Read(&bytes[0], data_byte_size))
      << "Invalid DLTensor file format";
  // explicitly swap endian when necessary.
  if (!DMLC_IO_NO_ENDIAN_SWAP) {
    dmlc::ByteSwap(&bytes[0], elem_bytes, num_elems);
  }
  TVM_CCALL(TVMArrayCopyFromBytes(dst, &bytes[0], data_byte_size));
}

void GraphRuntime::LoadParams(dmlc::Stream* strm) {
  uint64_t header, reserved;
  CHECK(strm->Read(&header))
      << "Invalid parameters file format";
  CHECK(header == kTVMNDArrayListMagic)
      << "Invalid parameters file format";
  CHECK(strm->Read(&reserved))
      << "Invalid parameters file format";

  std::vector<std::string> names;
  CHECK(strm->Read(&names))
      << "Invalid parameters file format";
  uint64_t sz;
  strm->Read(&sz);
  size_t size = static_cast<size_t>(sz);
  CHECK(size == names.size())
      << "Invalid parameters file format";
  for (size_t i = 0; i < size; ++i) {
    int in_idx = GetInputIndex(names[i]);
    CHECK_GE(in_idx, 0) << "Found param for non-existent input: " << names[i];
    uint32_t eid = this->GetEntryId(input_nodes_[in_idx], 0);
    CHECK_LT(eid, data_entry_.size());
    LoadDLTensor(strm, &data_entry_[eid]);
  }
}

void GraphRuntime::SetupStorage() {
  // Grab saved optimization plan from graph.
  std::vector<TVMType> vtype;
  for (const std::string& s_type : attrs_.dltype) {
    vtype.push_back(tvm::runtime::String2TVMType(s_type));
  }
  data_entry_.resize(num_node_entries());
  // size of each storage pool entry
  std::vector<size_t> pool_entry_bytes;
  // Find the maximum space size.
  for (size_t i = 0; i < attrs_.shape.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    size_t size = 1;
    for (int64_t sz : attrs_.shape[i]) {
      size *= static_cast<size_t>(sz);
    }
    CHECK_GE(storage_id, 0) << "Do not support runtime shape op";
    DLDataType t = vtype[i];
    size_t bits = t.bits * t.lanes;
    CHECK_EQ(bits % 8U, 0U);
    size_t bytes = (bits / 8U) * size;

    size_t sid = static_cast<size_t>(storage_id);
    if (sid >= pool_entry_bytes.size()) {
      pool_entry_bytes.resize(sid + 1, 0);
    }
    pool_entry_bytes[sid] = std::max(pool_entry_bytes[sid], bytes);
  }
  // Allocate the space.
  for (size_t i = 0; i < pool_entry_bytes.size(); ++i) {
    int64_t shape[] = {static_cast<int64_t>(pool_entry_bytes[i] + 3) / 4};
    DLTensor* tensor;
    TVM_CCALL(TVMArrayAlloc(
        shape, 1, kDLFloat, 32, 1, ctx_.device_type, ctx_.device_id, &tensor));
    storage_pool_.push_back(tensor);
  }
  // Assign the pooled entries.
  for (size_t i = 0; i < data_entry_.size(); ++i) {
    int storage_id = attrs_.storage_id[i];
    CHECK_LT(static_cast<size_t>(storage_id), storage_pool_.size());
    data_entry_[i] = *storage_pool_[storage_id];
    data_entry_[i].shape = const_cast<int64_t*>(attrs_.shape[i].data());
    data_entry_[i].ndim = static_cast<int>(attrs_.shape[i].size());
    data_entry_[i].dtype = vtype[i];
  }
}

void GraphRuntime::SetupOpExecs() {
  op_execs_.resize(this->num_nodes());
  // setup the array and requirements.
  for (uint32_t nid = 0; nid < this->num_nodes(); ++nid) {
    const auto& inode = nodes_[nid];
    if (inode.op_type == "null") continue;
    std::vector<DLTensor> args;
    for (const auto& e : inode.inputs) {
      args.push_back(data_entry_[this->entry_id(e)]);
    }
    for (uint32_t index = 0; index < inode.param.num_outputs; ++index) {
      uint32_t eid = this->GetEntryId(nid, index);
      args.push_back(data_entry_[eid]);
    }
    CHECK_EQ(inode.op_type, "tvm_op")
        << "Can only take tvm_op as op";
    op_execs_[nid] = CreateTVMOp(inode.param, args, inode.inputs.size());
  }
}

std::function<void()> GraphRuntime::CreateTVMOp(
    const TVMOpParam& param,
    const std::vector<DLTensor>& args,
    size_t num_inputs) {
  struct OpArgs {
    std::vector<DLTensor> args;
    std::vector<TVMValue> arg_values;
    std::vector<int> arg_tcodes;
    std::vector<int64_t> shape_data;
  };
  std::shared_ptr<OpArgs> arg_ptr = std::make_shared<OpArgs>();
  // setup address.
  arg_ptr->args = std::move(args);
  if (param.flatten_data) {
    arg_ptr->shape_data.resize(arg_ptr->args.size());
  }
  for (size_t i = 0; i < arg_ptr->args.size(); ++i) {
    TVMValue v;
    DLTensor* t = &(arg_ptr->args[i]);
    v.v_handle = t;
    arg_ptr->arg_values.push_back(v);
    arg_ptr->arg_tcodes.push_back(kArrayHandle);
    if (param.flatten_data) {
      arg_ptr->shape_data[i] = std::accumulate(
          t->shape, t->shape + t->ndim, 1, std::multiplies<int64_t>());
      t->ndim = 1;
      t->shape = &(arg_ptr->shape_data[i]);
    }
  }
  if (param.func_name == "__nop") {
    return [](){};
  }
  // get compiled function from module.
  tvm::runtime::PackedFunc pf = module_.GetFunction(param.func_name, false);
  CHECK(pf != nullptr) << "no such function in module: " << param.func_name;
  auto fexec = [arg_ptr, pf] () {
    TVMRetValue rv;
    TVMArgs targs(arg_ptr->arg_values.data(),
                  arg_ptr->arg_tcodes.data(),
                  static_cast<int>(arg_ptr->arg_values.size()));
    pf.CallPacked(targs, &rv);
  };
  return fexec;
}

PackedFunc GraphRuntime::GetFunction(
    const std::string& name,
    const std::shared_ptr<ModuleNode>& sptr_to_self) {
  // return member functions during query.
  if (name == "set_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          if (in_idx >= 0) this->SetInput(in_idx, args[1]);
        } else {
          this->SetInput(args[0], args[1]);
        }
      });
  } else if (name == "get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->GetOutput(args[0], args[1]);
      });
  } else if (name == "get_input") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          int in_idx = this->GetInputIndex(args[0]);
          CHECK_GE(in_idx, 0);
          this->GetInput(in_idx, args[1]);
        } else {
          this->GetInput(args[0], args[1]);
        }
      });
#ifdef TVM_GRAPH_RUNTIME_DEBUG
  } else if (name == "debug_get_output") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        if (args[0].type_code() == kStr) {
          this->DebugGetNodeOutput(this->GetNodeIndex(args[0]), args[1]);
        } else {
          this->DebugGetNodeOutput(args[0], args[1]);
        }
      });
#endif
  } else if (name == "run") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->Run();
      });
  } else if (name == "load_params") {
    return PackedFunc([sptr_to_self, this](TVMArgs args, TVMRetValue* rv) {
        this->LoadParams(args[0].operator std::string());
      });
  } else {
    return PackedFunc();
  }
}

Module GraphRuntimeCreate(std::string sym_json,
                          tvm::runtime::Module m,
                          int device_type,
                          int device_id) {
  TVMContext ctx;
  ctx.device_type = static_cast<DLDeviceType>(device_type);
  ctx.device_id   = device_id;
  std::shared_ptr<GraphRuntime> exec = std::make_shared<GraphRuntime>();
  exec->Init(sym_json, m, ctx);
  return Module(exec);
}

TVM_REGISTER_GLOBAL("tvm.graph_runtime.create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    *rv = GraphRuntimeCreate(args[0], args[1], args[2], args[3]);
  });

TVM_REGISTER_GLOBAL("tvm.graph_runtime.remote_create")
.set_body([](TVMArgs args, TVMRetValue *rv) {
    void* mhandle = args[1];
    *rv = GraphRuntimeCreate(args[0],
                             *static_cast<tvm::runtime::Module*>(mhandle),
                             args[2], args[3]);
  });
}  // namespace runtime
}  // namespace tvm
