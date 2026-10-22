// SPDX-License-Identifier: Apache-2.0
//
// Minimal port of the single_layer_kv_transfer path from
// third_pkgs/LMCache/csrc/cuda/mem_kernels.cu.
// Kept: single_layer_kv_transfer_kernel + its host-side dispatch wrapper,
// the EngineKVFormat guards (is_hnd / check_block_size / check_head_size)
// and get_kernel_ptr. Dropped: multi_layer* / _unilateral / _fused_ptr /
// cachegen / rocm(sgl) paths. Original license: Apache-2.0; see NOTICE.

#include <torch/extension.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <pybind11/pybind11.h>
#include <algorithm>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <limits>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#ifdef USE_ROCM
  #include <hip/hip_fp8.h>
#else
  #include <cuda_fp8.h>
#endif

#include "engine_kv_format.h"
#include "kv_transfer_types.h"

namespace py = pybind11;

namespace lmc {

// inline helper to check HND layout (callable from device and host)
__host__ __device__ __forceinline__ bool is_hnd(
    const EngineKVFormat engine_kv_format) {
  return engine_kv_format ==
             EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||  // flash attn HND
         engine_kv_format ==
             EngineKVFormat::NL_X_NB_TWO_NH_BS_HS;  // flash infer HND
}

// All paged (non-MLA) formats rely on block_size for offset computation.
// The blocked-scale indexer format is MLA-like for kv_size but its paged
// addressing is per-block, so it needs block_size too.
inline void check_block_size(const EngineKVFormat engine_kv_format,
                             const int block_size) {
  TORCH_CHECK((is_mla(engine_kv_format) &&
               engine_kv_format != EngineKVFormat::NL_X_NB_BSV_BSS) ||
                  block_size > 0,
              "block_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

// HND formats additionally need head_size to decompose scalar offsets.
inline void check_head_size(const EngineKVFormat engine_kv_format,
                            const int head_size) {
  TORCH_CHECK(!is_hnd(engine_kv_format) || head_size > 0,
              "head_size is required (must be > 0) for EngineKVFormat ",
              static_cast<int>(engine_kv_format));
}

template <typename scalar_t, EngineKVFormat format>
__global__ void single_layer_kv_transfer_kernel(
    scalar_t* __restrict__ lmc_key_value_cache,
    scalar_t* __restrict__ vllm_key_value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int vllm_block_key_stride_in_64bit, const int vllm_value_offset,
    const int lmc_stride, const int lmc_value_offset, const int num_heads,
    const int head_size_in_64bit, const int block_size,
    const TransferDirection direction) {
  constexpr bool USE_MLA = (format == EngineKVFormat::NL_X_NB_BS_HS);
  constexpr bool HND_LAYOUT = (format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS ||
                               format == EngineKVFormat::NL_X_NB_TWO_NH_BS_HS);

  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];

  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n = num_heads * head_size_in_64bit;

  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    const int64_t lmc_key_idx = token_idx * lmc_stride + i;

    const int head_idx = i / head_size_in_64bit;
    const int head_offset = i % head_size_in_64bit;

    int64_t vllm_key_idx;
    if constexpr (HND_LAYOUT) {
      // HND layout: [..., num_heads, block_size, head_size]
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     head_idx * block_size * head_size_in_64bit +
                     block_offset * head_size_in_64bit + head_offset;
    } else {
      // NHD layout: [..., block_size, num_heads, head_size]
      // (also correct for MLA where num_heads==1)
      vllm_key_idx = block_idx * vllm_block_key_stride_in_64bit +
                     block_offset * num_heads * head_size_in_64bit +
                     head_idx * head_size_in_64bit + head_offset;
    }

    if (direction == TransferDirection::D2H) {
      // GPU to LMCache (gather: paged -> staging)
      lmc_key_value_cache[lmc_key_idx] = vllm_key_value_cache[vllm_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        lmc_key_value_cache[lmc_value_idx] =
            vllm_key_value_cache[vllm_value_idx];
      }
    } else {
      // LMCache to GPU (scatter: staging -> paged)
      vllm_key_value_cache[vllm_key_idx] = lmc_key_value_cache[lmc_key_idx];
      if constexpr (!USE_MLA) {
        const int64_t lmc_value_idx = lmc_key_idx + lmc_value_offset;
        const int64_t vllm_value_idx = vllm_key_idx + vllm_value_offset;
        vllm_key_value_cache[vllm_value_idx] =
            lmc_key_value_cache[lmc_value_idx];
      }
    }
  }
}

}  // namespace lmc

template <typename T, typename TENSOR_TYPE>
T* get_kernel_ptr(TENSOR_TYPE& tensor) {
  // Get the kernel-accessible pointer of the given type T
  // Returns NULL if the tensor is on CPU and non-pinned
  torch::Device device = tensor.device();
  if (device.is_cuda()) {
    return static_cast<T*>(tensor.data_ptr());
  } else if (device.is_cpu()) {
    T* ptr;
    auto st = cudaHostGetDevicePointer(
        (void**)&ptr, static_cast<void*>(tensor.data_ptr()), 0);
    TORCH_CHECK(st == cudaSuccess,
                "Host tensor not registered/pinned (or bad ptr)");
    return ptr;
  } else {
    TORCH_CHECK(false, "Invalid device. Device must be cuda or pinned cpu.");
  }
}

void single_layer_kv_transfer(
    torch::Tensor& lmc_key_value_cache,  // [num_tokens, 2, num_heads*head_size]
                                         // or
                                         // [2, num_tokens, num_heads*head_size]
                                         // or for MLA:
                                         // [num_tokens, aligned_head_size]

    torch::Tensor&
        vllm_key_value_cache,  // NHD: [2, num_blocks, block_size, num_heads,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, block_size, num_heads,
                               //       head_size] for flash infer
                               // HND: [2, num_blocks, num_heads, block_size,
                               //       head_size] for flash attention
                               //      [num_blocks, 2, num_heads, block_size,
                               //       head_size] for flash infer
                               // MLA: [num_blocks, block_size, head_size]

    torch::Tensor& slot_mapping,  // [num_tokens]
    const TransferDirection direction, const EngineKVFormat engine_kv_format,
    const bool token_major  // true: lmc_key_value_cache is
                            // [num_tokens, 2, num_heads*head_size]
                            // false: lmc_key_value_cache is
                            // [2, num_tokens, num_heads*head_size]
) {
  int64_t* lmc_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(lmc_key_value_cache);

  int64_t* vllm_key_value_cache_ptr =
      get_kernel_ptr<int64_t, torch::Tensor>(vllm_key_value_cache);

  const int64_t* slot_mapping_ptr =
      get_kernel_ptr<const int64_t, const torch::Tensor>(slot_mapping);

  int elements_per_entry = 8 / vllm_key_value_cache.element_size();

  int num_tokens = slot_mapping.size(0);
  int num_heads;
  int head_size_in_64bit;
  int block_size;

  const bool use_mla = ::is_mla(engine_kv_format);
  const bool hnd_layout = lmc::is_hnd(engine_kv_format);

  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    num_heads = 1;
    block_size = vllm_key_value_cache.size(1);
    head_size_in_64bit = vllm_key_value_cache.size(2) / elements_per_entry;
  } else if (hnd_layout) {
    // HND format: [..., num_heads, block_size, head_size]
    num_heads = vllm_key_value_cache.size(2);
    block_size = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  } else {
    // NHD format: [..., block_size, num_heads, head_size]
    block_size = vllm_key_value_cache.size(2);
    num_heads = vllm_key_value_cache.size(3);
    head_size_in_64bit = vllm_key_value_cache.size(4) / elements_per_entry;
  }

  lmc::check_block_size(engine_kv_format, block_size);
  lmc::check_head_size(engine_kv_format, head_size_in_64bit);

  int lmc_stride;
  int lmc_value_offset;
  if (use_mla) {
    // MLA format: [num_tokens, aligned_head_size]
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = 0;  // No separate K/V for MLA
  } else if (token_major) {
    lmc_stride = lmc_key_value_cache.stride(0) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(1) / elements_per_entry;
  } else {
    lmc_stride = lmc_key_value_cache.stride(1) / elements_per_entry;
    lmc_value_offset = lmc_key_value_cache.stride(0) / elements_per_entry;
  }

  int vllm_block_key_stride_in_64bit;
  int vllm_value_offset;
  if (use_mla) {
    // MLA format: [num_blocks, block_size, head_size]
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = 0;  // No separate K/V for MLA
  } else if (engine_kv_format == EngineKVFormat::NL_X_TWO_NB_BS_NH_HS ||
             engine_kv_format == EngineKVFormat::NL_X_TWO_NB_NH_BS_HS) {
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(1) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(0) / elements_per_entry;
  } else {  // engine_kv_format == EngineKVFormat::NL_X_NB_TWO_BS_NH_HS
    vllm_block_key_stride_in_64bit =
        vllm_key_value_cache.stride(0) / elements_per_entry;
    vllm_value_offset = vllm_key_value_cache.stride(1) / elements_per_entry;
  }

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size_in_64bit, 128));
  const at::cuda::OptionalCUDAGuard device_guard(
      device_of(vllm_key_value_cache));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // Dispatch to the appropriate template specialization based on EngineKVFormat
#define LAUNCH_SINGLE_LAYER_KERNEL(FORMAT)                                     \
  lmc::single_layer_kv_transfer_kernel<int64_t, FORMAT>                        \
      <<<grid, block, 0, stream>>>(                                            \
          lmc_key_value_cache_ptr, vllm_key_value_cache_ptr, slot_mapping_ptr, \
          vllm_block_key_stride_in_64bit, vllm_value_offset, lmc_stride,       \
          lmc_value_offset, num_heads, head_size_in_64bit, block_size,         \
          direction);                                                          \
  break;

  switch (engine_kv_format) {
    case EngineKVFormat::NL_X_NB_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_BS_HS)
    case EngineKVFormat::NL_X_TWO_NB_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
    case EngineKVFormat::NL_X_NB_TWO_BS_NH_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
    case EngineKVFormat::NL_X_TWO_NB_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
    case EngineKVFormat::NL_X_NB_TWO_NH_BS_HS:
      LAUNCH_SINGLE_LAYER_KERNEL(EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
    default:
      TORCH_CHECK(false,
                  "Unsupported EngineKVFormat for single_layer_kv_transfer: ",
                  static_cast<int>(engine_kv_format));
  }
#undef LAUNCH_SINGLE_LAYER_KERNEL
}

class ReplicaIpcExport {
 public:
  explicit ReplicaIpcExport(torch::Tensor tensor) {
    TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kUInt8 &&
                    tensor.is_contiguous() && tensor.numel() > 0,
                "replica export requires contiguous CUDA uint8 storage");
    device_ = tensor.get_device();
    c10::cuda::CUDAGuard guard(device_);
    void* function = nullptr;
    C10_CUDA_CHECK(cudaGetDriverEntryPoint("cuMemGetAddressRange", &function,
                                         cudaEnableDefault));
    TORCH_CHECK(function, "CUDA allocation range query is unavailable");
    using Query = CUresult (CUDAAPI *)(CUdeviceptr*, size_t*, CUdeviceptr);
    CUdeviceptr base = 0;
    size_t bytes = 0;
    const auto pointer = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto status = reinterpret_cast<Query>(function)(&base, &bytes, pointer);
    TORCH_CHECK(status == CUDA_SUCCESS && base <= pointer &&
                    bytes <= static_cast<size_t>(std::numeric_limits<int64_t>::max()) &&
                    pointer - base <= bytes && tensor.numel() <= bytes - (pointer - base),
                "replica KV allocation is not exportable without staging (VMM is unsupported)");
    const auto exported = cudaIpcGetMemHandle(&handle_, reinterpret_cast<void*>(base));
    TORCH_CHECK(exported == cudaSuccess,
                "replica KV allocation is not exportable without staging: ",
                cudaGetErrorString(exported));
    offset_ = pointer - base;
    bytes_ = bytes;
    tensor_ = std::make_unique<torch::Tensor>(std::move(tensor));
  }
  ~ReplicaIpcExport() { if (tensor_) tensor_.release(); }
  py::bytes handle() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(tensor_, "replica IPC export is closed");
    return py::bytes(reinterpret_cast<const char*>(&handle_), sizeof(handle_));
  }
  int64_t offset() const { return offset_; }
  int64_t allocation_bytes() const { return bytes_; }
  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    c10::cuda::CUDAGuard guard(device_);
    tensor_.reset();
  }
 private:
  std::unique_ptr<torch::Tensor> tensor_;
  cudaIpcMemHandle_t handle_;
  int device_;
  int64_t offset_, bytes_;
  std::mutex mutex_;
};

class ReplicaIpcMapping {
 public:
  ReplicaIpcMapping(const std::string& blob, int64_t offset, int64_t bytes, int device)
      : offset_(offset), bytes_(bytes), device_(device) {
    TORCH_CHECK(blob.size() == sizeof(cudaIpcMemHandle_t) && offset >= 0 && bytes > 0 &&
                    offset <= std::numeric_limits<int64_t>::max() - bytes,
                "invalid replica IPC descriptor");
    cudaIpcMemHandle_t handle;
    std::memcpy(&handle, blob.data(), sizeof(handle));
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(cudaIpcOpenMemHandle(&base_, handle, cudaIpcMemLazyEnablePeerAccess)
                    == cudaSuccess, "replica IPC import failed");
  }
  uintptr_t acquire_plan(int device, int64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(base_, "replica IPC mapping is closed");
    TORCH_CHECK(device == device_ && bytes <= bytes_, "replica IPC source size or device mismatch");
    ++live_plans_;
    return reinterpret_cast<uintptr_t>(base_) + offset_;
  }
  void release_plan() {
    std::lock_guard<std::mutex> lock(mutex_);
    --live_plans_;
  }
  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!base_) return;
    TORCH_CHECK(live_plans_ == 0, "replica IPC mapping still has live scatter plans");
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(cudaIpcCloseMemHandle(base_) == cudaSuccess, "replica IPC close failed");
    base_ = nullptr;
  }
 private:
  void* base_ = nullptr;
  int64_t offset_, bytes_;
  int device_, live_plans_ = 0;
  std::mutex mutex_;
};

__global__ void replica_page_scatter_kernel(const int64_t* descriptors,
    uint8_t* destination, int64_t count, int64_t page_bytes, bool vectorized) {
  for (int64_t page = blockIdx.x; page < count; page += gridDim.x) {
    const auto* row = descriptors + page * 3;
    const auto* source = reinterpret_cast<const uint8_t*>(row[0]) + row[1];
    auto* target = destination + row[2];
    if (vectorized) {
      const auto* src = reinterpret_cast<const uint4*>(source);
      auto* dst = reinterpret_cast<uint4*>(target);
      for (int64_t i = threadIdx.x; i < page_bytes / 16; i += blockDim.x) dst[i] = src[i];
    } else {
      for (int64_t i = threadIdx.x; i < page_bytes; i += blockDim.x) target[i] = source[i];
    }
  }
}

class ReplicaPeerPagePlan {
  struct State {
    torch::Tensor destination, host, descriptors;
    std::vector<std::shared_ptr<ReplicaIpcMapping>> mappings;
    cudaEvent_t fence = nullptr;
    bool submitted = false, reliable_fence = true;
    ~State() {
      if (fence) cudaEventDestroy(fence);
      for (auto& mapping : mappings) mapping->release_plan();
    }
  };
 public:
  ReplicaPeerPagePlan(std::vector<std::shared_ptr<ReplicaIpcMapping>> peers,
      std::vector<int64_t> offsets, std::vector<int64_t> sizes,
      torch::Tensor destination, torch::Tensor pages)
      : state_(std::make_unique<State>()) {
    TORCH_CHECK(destination.is_cuda() && destination.scalar_type() == torch::kUInt8 &&
                    destination.dim() == 2 && destination.is_contiguous() && destination.numel() > 0,
                "scatter destination must be contiguous CUDA uint8 pages");
    TORCH_CHECK(pages.device().is_cpu() && pages.scalar_type() == torch::kInt64 &&
                    pages.dim() == 2 && pages.size(1) == 3 && pages.is_contiguous(),
                "scatter page table must be contiguous CPU int64 [N,3]");
    TORCH_CHECK(peers.size() < 8 && peers.size() == offsets.size() && peers.size() == sizes.size(),
                "invalid scatter source members");
    device_ = destination.get_device();
    page_bytes_ = destination.size(1);
    count_ = pages.size(0);
    state_->destination = destination;
    c10::cuda::CUDAGuard guard(device_);
    std::vector<uintptr_t> sources;
    sources.reserve(peers.size());
    state_->mappings.reserve(peers.size());
    for (size_t peer = 0; peer < peers.size(); ++peer) {
      TORCH_CHECK(peers[peer] && offsets[peer] >= 0 && sizes[peer] > 0 &&
                      sizes[peer] % page_bytes_ == 0 &&
                      offsets[peer] <= std::numeric_limits<int64_t>::max() - sizes[peer],
                  "invalid scatter source size or offset");
      state_->mappings.push_back(peers[peer]);
      uintptr_t base;
      try {
        base = peers[peer]->acquire_plan(device_, offsets[peer] + sizes[peer]);
      } catch (...) {
        state_->mappings.pop_back();
        throw;
      }
      TORCH_CHECK(base <= std::numeric_limits<uintptr_t>::max() - offsets[peer] - sizes[peer],
                  "scatter source address overflows");
      sources.push_back(base + offsets[peer]);
    }
    const auto target = reinterpret_cast<uintptr_t>(destination.data_ptr());
    TORCH_CHECK(target <= std::numeric_limits<uintptr_t>::max() - destination.numel(),
                "scatter destination address overflows");
    for (size_t peer = 0; peer < sources.size(); ++peer) {
      TORCH_CHECK(target >= sources[peer] + sizes[peer] ||
                      sources[peer] >= target + destination.numel(),
                  "scatter source and destination overlap");
    }
    state_->host = torch::empty({count_, 3}, torch::TensorOptions().dtype(torch::kInt64).pinned_memory(true));
    const auto* input = pages.data_ptr<int64_t>();
    auto* output = state_->host.data_ptr<int64_t>();
    std::set<int64_t> destinations;
    vectorized_ = page_bytes_ % 16 == 0 && target % 16 == 0;
    for (auto source : sources) vectorized_ = vectorized_ && source % 16 == 0;
    for (int64_t i = 0; i < count_; ++i) {
      const int64_t member = input[i * 3], src = input[i * 3 + 1], dst = input[i * 3 + 2];
      TORCH_CHECK(member >= 0 && member < static_cast<int64_t>(sources.size()) &&
                      src >= 0 && src < sizes[member] / page_bytes_ &&
                      dst >= 0 && dst < destination.size(0), "scatter page index out of bounds");
      TORCH_CHECK(destinations.insert(dst).second, "scatter destination pages overlap");
      output[i * 3] = static_cast<int64_t>(sources[member]);
      output[i * 3 + 1] = src * page_bytes_;
      output[i * 3 + 2] = dst * page_bytes_;
    }
    C10_CUDA_CHECK(cudaEventCreateWithFlags(&state_->fence, cudaEventDisableTiming));
    try {
      state_->submitted = true;
      state_->descriptors = state_->host.to(destination.device(), true);
      C10_CUDA_CHECK(cudaEventRecord(state_->fence, at::cuda::getCurrentCUDAStream(device_)));
    } catch (...) {
      state_.release();
      throw;
    }
  }
  ~ReplicaPeerPagePlan() {
    if (!state_) return;
    try {
      c10::cuda::CUDAGuard guard(device_);
      if (!state_->reliable_fence ||
          (state_->submitted && cudaEventQuery(state_->fence) != cudaSuccess)) state_.release();
      else state_.reset();
    } catch (...) {
      state_.release();
    }
  }
  void launch() {
    std::lock_guard<std::mutex> lock(mutex_);
    TORCH_CHECK(state_ && state_->reliable_fence, "replica scatter plan is closed or poisoned");
    c10::cuda::CUDAGuard guard(device_);
    auto stream = at::cuda::getCurrentCUDAStream(device_);
    C10_CUDA_CHECK(cudaStreamWaitEvent(stream, state_->fence, 0));
    state_->reliable_fence = false;
    if (count_) {
      replica_page_scatter_kernel<<<std::min<int64_t>(count_, 256), 128, 0, stream>>>(
          state_->descriptors.data_ptr<int64_t>(), state_->destination.data_ptr<uint8_t>(),
          count_, page_bytes_, vectorized_);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    C10_CUDA_CHECK(cudaEventRecord(state_->fence, stream));
    state_->reliable_fence = true;
  }
  void close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!state_) return;
    c10::cuda::CUDAGuard guard(device_);
    TORCH_CHECK(state_->reliable_fence && cudaEventQuery(state_->fence) == cudaSuccess,
                "replica scatter plan is busy; backing retained");
    state_.reset();
  }
 private:
  std::unique_ptr<State> state_;
  int device_;
  int64_t page_bytes_, count_;
  bool vectorized_;
  std::mutex mutex_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<ReplicaIpcMapping, std::shared_ptr<ReplicaIpcMapping>>(m, "ReplicaIpcMapping")
      .def(py::init<const std::string&, int64_t, int64_t, int>())
      .def("close", &ReplicaIpcMapping::close, py::call_guard<py::gil_scoped_release>());
  py::class_<ReplicaIpcExport>(m, "ReplicaIpcExport")
      .def(py::init<torch::Tensor>())
      .def("handle", &ReplicaIpcExport::handle)
      .def_property_readonly("offset", &ReplicaIpcExport::offset)
      .def_property_readonly("allocation_bytes", &ReplicaIpcExport::allocation_bytes)
      .def("close", &ReplicaIpcExport::close, py::call_guard<py::gil_scoped_release>());
  py::class_<ReplicaPeerPagePlan>(m, "ReplicaPeerPagePlan")
      .def(py::init<std::vector<std::shared_ptr<ReplicaIpcMapping>>, std::vector<int64_t>,
                    std::vector<int64_t>, torch::Tensor, torch::Tensor>())
      .def("launch", &ReplicaPeerPagePlan::launch, py::call_guard<py::gil_scoped_release>())
      .def("close", &ReplicaPeerPagePlan::close, py::call_guard<py::gil_scoped_release>());
  m.def(
      "single_layer_kv_transfer",
      [](torch::Tensor lmc_key_value_cache,
         torch::Tensor vllm_key_value_cache, torch::Tensor slot_mapping,
         int direction, int engine_kv_format, bool token_major) {
        return single_layer_kv_transfer(
            lmc_key_value_cache, vllm_key_value_cache, slot_mapping,
            static_cast<TransferDirection>(direction),
            static_cast<EngineKVFormat>(engine_kv_format), token_major);
      },
      py::arg("lmc_key_value_cache"), py::arg("vllm_key_value_cache"),
      py::arg("slot_mapping"), py::arg("direction"),
      py::arg("engine_kv_format"), py::arg("token_major"));
}
