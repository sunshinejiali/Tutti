"""Experimental replica IO layout, derived without importing the data plane."""
import os


def validate_layerwise_execution(vllm_config):
    if getattr(vllm_config.model_config, "enforce_eager", False):
        return
    compilation = getattr(vllm_config, "compilation_config", None)
    graph = getattr(compilation, "cudagraph_mode", None)
    mode = getattr(compilation, "mode", None)
    if getattr(graph, "name", graph) != "NONE" or mode != 3:
        raise ValueError("独立KV实体支持 enforce_eager=True 或 mode=3、cudagraph_mode=NONE；CUDA graph回调尚未支持")
    if os.environ.get("VLLM_USE_BREAKABLE_CUDAGRAPH") == "1":
        raise ValueError("独立KV实体编译要求 VLLM_USE_BREAKABLE_CUDAGRAPH=0")


def validate_replica_io(vllm_config, extra):
    enabled = extra.get("replica_io", False)
    if not isinstance(enabled, bool):
        raise ValueError("replica_io must be a bool")
    if not enabled:
        return
    parallel = vllm_config.parallel_config
    if (parallel.tensor_parallel_size != 8 or
            any(getattr(parallel, field, 1) != 1 for field in (
                "data_parallel_size", "pipeline_parallel_size",
                "decode_context_parallel_size", "prefill_context_parallel_size"))):
        raise ValueError("replica_io initially requires single-node TP8, DP/PP/DCP/PCP=1")
    model = vllm_config.model_config
    hf = getattr(model, "hf_config", None)
    text = getattr(model, "hf_text_config", None)
    if text is None:
        text = getattr(hf, "text_config", hf)
    if (getattr(hf, "model_type", "") not in ("minimax_m3", "minimax_m3_vl") or
            getattr(text, "num_key_value_heads", None) != 4 or
            getattr(text, "head_dim", None) != 128):
        raise ValueError("replica_io initially supports MiniMax M3 with 4 KV heads, head_dim=128")
    validate_layerwise_execution(vllm_config)
    if os.environ.get("TUTTI_ASYNC_COMPLETION") == "1":
        raise ValueError("replica_io requires synchronous NVMe completion")
    if extra.get("store", {}).get("type") != "tutti_nvme":
        raise ValueError("replica_io requires tutti_nvme")
    names = extra.get("kv_group_layer_names", ())
    if ("kv_layer_page_bytes" not in extra or len(names) != 117 or
            sum(name.endswith(".index_cache") for name in names) != 57):
        raise ValueError("replica_io requires the registered 60 main KV and 57 index entities")


def replica_groups_for(extra, rank):
    """Main KV stays rank-local; only replicated index entities use TP8 owners."""
    if not extra.get("replica_io", False):
        return ()
    return tuple(tuple(range(8)) if name.endswith(".index_cache") else (rank,)
                 for name in extra["kv_group_layer_names"])
