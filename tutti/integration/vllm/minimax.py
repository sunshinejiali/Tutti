"""MiniMax M3 eager attention callbacks, installed without patching vLLM."""


def install_cache_callbacks(vllm_config, connector, caches):
    context = vllm_config.compilation_config.static_forward_context
    order = tuple(connector._impl._kv_group_layer_names)
    pairs = []
    for name, layer in context.items():
        if name not in caches or not hasattr(layer, "indexer"):
            continue
        index_cache = getattr(layer.indexer, "index_cache", None)
        if index_cache is None or index_cache.prefix not in caches:
            raise ValueError("MiniMax attention index cache is missing from the KV group")
        names = tuple(key for key in order if key in (name, index_cache.prefix))
        if len(names) != 2:
            raise ValueError("MiniMax attention requires both main and index cache callbacks")
        pairs.append((layer, names))
    if not pairs:
        raise ValueError("Independent KV entities require a supported custom attention callback adapter")
    handles = []
    try:
        for layer, names in pairs:
            def before(module, args, _names=names):
                if connector.has_connector_metadata():
                    for key in _names:
                        connector.wait_for_layer_load(key)

            def after(module, args, result, _names=names):
                if connector.has_connector_metadata():
                    from vllm.forward_context import get_forward_context
                    metadata = get_forward_context().attn_metadata
                    for key in _names:
                        connector.save_kv_layer(key, caches[key], metadata)

            handles.append(layer.register_forward_pre_hook(before))
            handles.append(layer.register_forward_hook(after))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return tuple(handles)
