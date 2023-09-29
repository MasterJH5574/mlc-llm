/*!
 *  Copyright (c) 2023 by Contributors
 * \file common.h
 * \brief Header of common utilities.
 */

#include "common.h"

#include <tokenizers_cpp.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/disco/session.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/relax_vm/memory_manager.h>

#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <list>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

namespace mlc {
namespace llm {

//----------------------------
// Tokenizers
//----------------------------
std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  ICHECK(!fs.fail()) << "Cannot open " << path;
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path) {
  std::filesystem::path path(_path);
  std::filesystem::path sentencepiece;
  std::filesystem::path huggingface;
  std::filesystem::path rwkvworld;
  CHECK(std::filesystem::exists(path)) << "Cannot find tokenizer via path: " << _path;
  if (std::filesystem::is_directory(path)) {
    sentencepiece = path / "tokenizer.model";
    huggingface = path / "tokenizer.json";
    rwkvworld = path / "tokenizer_model";
    // Check ByteLevelBPE
    {
      std::filesystem::path merges_path = path / "merges.txt";
      std::filesystem::path vocab_path = path / "vocab.json";
      std::filesystem::path added_tokens_path = path / "added_tokens.json";
      if (std::filesystem::exists(merges_path) && std::filesystem::exists(vocab_path) &&
          std::filesystem::exists(added_tokens_path)) {
        std::string vocab = LoadBytesFromFile(vocab_path.string());
        std::string merges = LoadBytesFromFile(merges_path.string());
        std::string added_tokens = LoadBytesFromFile(added_tokens_path.string());
        return Tokenizer::FromBlobByteLevelBPE(vocab, merges, added_tokens);
      }
    }
  } else {
    sentencepiece = path.parent_path() / "tokenizer.model";
    huggingface = path.parent_path() / "tokenizer.json";
    rwkvworld = path.parent_path() / "tokenizer_model";
  }
  if (std::filesystem::exists(sentencepiece)) {
    return Tokenizer::FromBlobSentencePiece(LoadBytesFromFile(sentencepiece.string()));
  }
  if (std::filesystem::exists(huggingface)) {
    return Tokenizer::FromBlobJSON(LoadBytesFromFile(huggingface.string()));
  }
  if (std::filesystem::exists(rwkvworld)) {
    return Tokenizer::FromBlobRWKVWorld(rwkvworld.string());
  }
  LOG(FATAL) << "Cannot find any tokenizer under: " << _path;
}

//----------------------------
// FunctionTable
//----------------------------

PackedFunc FunctionTable::SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name) {
  return PackedFunc([sess, func = std::move(sess_func), name = std::move(name)](
                        TVMArgs args, TVMRetValue* rv) -> void {
    std::vector<TVMValue> tvm_values(args.num_args + 3);
    std::vector<int> tvm_type_codes(args.num_args + 3);
    TVMArgsSetter setter(tvm_values.data(), tvm_type_codes.data());
    setter(0, static_cast<int>(DiscoAction::kCallPacked));
    setter(1, 0);
    setter(2, func);
    for (int i = 0; i < args.num_args; ++i) {
      tvm_values[i + 3] = args.values[i];
      tvm_type_codes[i + 3] = args.type_codes[i];
    }
    *rv =
        sess->CallWithPacked(TVMArgs(tvm_values.data(), tvm_type_codes.data(), args.num_args + 3));
  });
}

void FunctionTable::Init(TVMArgValue reload_lib, Device device, int num_shards) {
  Device null_device{DLDeviceType(0), 0};
  if (num_shards > 1) {
    String lib_path{nullptr};
    try {
      lib_path = reload_lib.operator String();
    } catch (...) {
      LOG(FATAL)
          << "ValueError: In multi-GPU inference, we expect the first argument to Reload to be a "
             "string path to the model library (.so on Linux or .dll on Windows), but got: "
          << ArgTypeCode2Str(reload_lib.type_code());
    }
    constexpr const char* f_create_process_pool = "runtime.disco.create_process_pool";
    if (Registry::Get(f_create_process_pool) == nullptr) {
      LOG(FATAL) << "Cannot find process launcher `" << f_create_process_pool << "`. "
                 << "Multi-GPU inference depends on MLC LLM Python API to launch process.";
    }
    std::string ccl;
    if (device.device_type == kDLCUDA) {
      ccl = "nccl";
    } else if (device.device_type == kDLROCM) {
      ccl = "rccl";
    } else {
      LOG(FATAL) << "ValueError: Multi-GPU on device " << DLDeviceType2Str(device.device_type)
                 << " is not supported. Currently, only NCCL and RCCL are integrated.";
    }
    std::vector<int64_t> device_ids(num_shards);
    for (int i = 0; i < num_shards; ++i) {
      device_ids[i] = i;
    }
    this->use_disco = true;
    this->sess = Session::ProcessSession(num_shards, f_create_process_pool);
    this->sess->InitCCL(ccl, ShapeTuple(device_ids));
    this->disco_mod = sess->CallPacked(sess->GetGlobalFunc("runtime.disco.load_vm_module"),
                                       lib_path, null_device);
    this->mod_get_func = [this,
                          fmodule_get_function = sess->GetGlobalFunc("runtime.ModuleGetFunction")](
                             const std::string& name) -> PackedFunc {
      DRef func = sess->CallPacked(fmodule_get_function, this->disco_mod, name, false);
      bool exists = (func->DebugGetFromRemote(0).operator PackedFunc()) != nullptr;
      if (!exists) {
        return PackedFunc(nullptr);
      }
      return SessionFuncAsPackedFunc(sess, func, name);
    };
    this->get_global_func = [this](const std::string& name) -> PackedFunc {
      return SessionFuncAsPackedFunc(sess, sess->GetGlobalFunc(name), name);
    };
    this->_InitFunctions();
    {
      Module mod = this->disco_mod->DebugGetFromRemote(0);
      this->softmax_func_ = mod->GetFunction("softmax_with_temperature");
    }
  } else {
    Module executable{nullptr};
    if (reload_lib.type_code() == kTVMModuleHandle) {
      executable = reload_lib.operator Module();
    } else {
      String lib_path = reload_lib.operator String();
      executable = tvm::runtime::Module::LoadFromFile(lib_path);
    }
    this->use_disco = false;
    auto fload_exec = executable->GetFunction("vm_load_executable");
    ICHECK(fload_exec.defined()) << "TVM runtime cannot find vm_load_executable";
    this->local_vm = fload_exec();
    this->local_vm->GetFunction("vm_initialization")(
        static_cast<int>(device.device_type), device.device_id,
        static_cast<int>(relax_vm::AllocatorType::kPooled), static_cast<int>(kDLCPU), 0,
        static_cast<int>(relax_vm::AllocatorType::kPooled));
    this->mod_get_func = [this](const std::string& name) -> PackedFunc {
      return this->local_vm->GetFunction(name, false);
    };
    this->get_global_func = [](const std::string& name) -> PackedFunc {
      const auto* f = tvm::runtime::Registry::Get(name);
      CHECK(f != nullptr) << "ValueError: Cannot find function " << name;
      return *f;
    };
    this->_InitFunctions();
  }
}

ObjectRef FunctionTable::LoadParams(const std::string& model_path, Device device) {
  if (this->use_disco) {
    std::filesystem::path fs_model_path = model_path;
    std::string metadata_path = (fs_model_path / "ndarray-cache.json").string();
    std::string ndarray_cache_metadata = LoadBytesFromFile(metadata_path);
    PackedFunc loader_create = this->get_global_func("runtime.disco.ShardLoader");
    PackedFunc loader_load_all = this->get_global_func("runtime.disco.ShardLoaderLoadAll");
    CHECK(loader_create != nullptr);
    CHECK(loader_load_all != nullptr);
    DRef loader = loader_create(metadata_path, ndarray_cache_metadata, "", this->disco_mod);
    DRef params = loader_load_all(loader);
    return params;
  } else {
    const PackedFunc* fload_cache = tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.load");
    ICHECK(fload_cache) << "TVM runtime cannot find vm.builtin.ndarray_cache.load";
    (*fload_cache)(model_path, static_cast<int32_t>(device.device_type), device.device_id);
    const PackedFunc* fload_params =
        tvm::runtime::Registry::Get("vm.builtin.param_array_from_cache");
    ICHECK(fload_params) << "Cannot find env function vm.builtin.param_array_from_cache";
    Array<NDArray> params = (*fload_params)("param", -1);
    // after we get params, it is safe to simply clear the cached version
    // as these params are referenced by params_
    const PackedFunc* fclear_ndarray_cache =
        tvm::runtime::Registry::Get("vm.builtin.ndarray_cache.clear");
    ICHECK(fclear_ndarray_cache) << "Cannot find env function vm.builtin.ndarray_cache.clear";
    (*fclear_ndarray_cache)();
    return params;
  }
}

void FunctionTable::_InitFunctions() {
  this->prefill_func_ = mod_get_func("prefill");
  this->embed_func_ = mod_get_func("embed");
  this->prefill_with_embed_func_ = mod_get_func("prefill_with_embed");
  this->decode_func_ = mod_get_func("decode");
  this->softmax_func_ = mod_get_func("softmax_with_temperature");
  this->encoding_without_cache_func_ = mod_get_func("encoding_without_cache");
  this->create_kv_cache_func_ = mod_get_func("create_kv_cache");
  this->reset_kv_cache_func_ = mod_get_func("reset_kv_cache");
  if (this->reset_kv_cache_func_ == nullptr) {
    this->reset_kv_cache_func_ = get_global_func("vm.builtin.attention_kv_cache_array_clear");
    support_backtracking_kv_ = true;
  } else {
    support_backtracking_kv_ = false;
  }
  this->fkvcache_array_popn_ = get_global_func("vm.builtin.attention_kv_cache_array_popn");
}

ObjectRef FunctionTable::Empty(ShapeTuple shape, DataType dtype, Device device) const {
  Device null_device{DLDeviceType(0), 0};
  if (this->use_disco) {
    DRef empty_func = sess->GetGlobalFunc("runtime.disco.empty");
    return sess->CallPacked(empty_func, shape, dtype, null_device);
  } else {
    return NDArray::Empty(shape, dtype, device);
  }
}

ObjectRef FunctionTable::CopyToWorker0(const NDArray& host_array) {
  Device null_device{DLDeviceType(0), 0};
  if (this->use_disco) {
    DRef array =
        Downcast<DRef>(this->Empty(host_array.Shape(), host_array.DataType(), null_device));
    sess->CopyToWorker0(host_array, array);
    return array;
  } else {
    return host_array;
  }
}

}  // namespace llm
}  // namespace mlc
