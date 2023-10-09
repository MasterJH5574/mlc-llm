/*!
 *  Copyright (c) 2023 by Contributors
 * \file common.h
 * \brief Header of common utilities.
 */

#ifndef MLC_LLM_DLL
#ifdef _WIN32
#ifdef MLC_LLM_EXPORTS
#define MLC_LLM_DLL __declspec(dllexport)
#else
#define MLC_LLM_DLL __declspec(dllimport)
#endif
#else
#define MLC_LLM_DLL __attribute__((visibility("default")))
#endif
#endif

#ifndef MLC_LLM_COMMON_H_
#define MLC_LLM_COMMON_H_

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

using tvm::Device;
using namespace tvm::runtime;
using tokenizers::Tokenizer;

// Random number generator
class RandomGenerator {
 private:
  std::mt19937 gen;
  std::uniform_real_distribution<> dis;

  RandomGenerator(int seed) : gen(seed), dis(0.0, 1.0) {}

 public:
  static RandomGenerator& GetInstance(int seed = std::random_device{}()) {
    static RandomGenerator instance(seed);
    return instance;
  }

  double GetRandomNumber() { return dis(gen); }

  void SetSeed(int seed) { gen.seed(seed); }
};

MLC_LLM_DLL std::unique_ptr<Tokenizer> TokenizerFromPath(const std::string& _path);

struct FunctionTable {
  static PackedFunc SessionFuncAsPackedFunc(Session sess, DRef sess_func, String name);

  void Init(TVMArgValue reload_lib, Device device, int num_shards, bool use_paged_kv_cache = false);

  ObjectRef LoadParams(const std::string& model_path, Device device);

  void _InitFunctions(bool use_paged_kv_cache);

  ObjectRef Empty(ShapeTuple shape, DataType dtype, Device device) const;

  ObjectRef CopyToWorker0(const NDArray& host_array);

  bool use_disco = false;
  Session sess{nullptr};
  DRef disco_mod{nullptr};
  tvm::runtime::Module local_vm{nullptr};

  TypedPackedFunc<PackedFunc(const std::string&)> mod_get_func;
  TypedPackedFunc<PackedFunc(const std::string&)> get_global_func;

  PackedFunc prefill_func_;
  PackedFunc embed_func_;
  PackedFunc prefill_with_embed_func_;
  PackedFunc decode_func_;
  PackedFunc encoding_without_cache_func_;
  PackedFunc softmax_func_;
  PackedFunc create_kv_cache_func_;
  PackedFunc reset_kv_cache_func_;
  bool support_backtracking_kv_;
  PackedFunc remove_from_kv_cache_func_;
  PackedFunc fkvcache_array_popn_;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_COMMON_H_
