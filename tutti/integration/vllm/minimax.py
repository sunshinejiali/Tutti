"""MiniMax M3 runtime KV callbacks with optional compiled projections."""


class _CompiledForward:
    def __init__(self, layer, name):
        import logging
        import torch
        from torch._dynamo.backends.registry import lookup_backend

        self.layer = layer
        self.original = layer.forward
        self.name = name
        inductor = lookup_backend("inductor")
        def backend(graph, inputs):
            result = inductor(graph, inputs)
            logging.getLogger(__name__).info(
                "TUTTI_COMPILE_GRAPH layer=%s nodes=%d", name, len(list(graph.graph.nodes)))
            return result
        compiled = torch.compile(self.original, backend=backend, fullgraph=True, dynamic=True)
        def forward(*args, **kwargs):
            from vllm.compilation.wrapper import _compilation_context
            with _compilation_context():
                return compiled(*args, **kwargs)
        layer.forward = forward

    def remove(self):
        self.layer.forward = self.original


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
            if not getattr(getattr(vllm_config, "model_config", None), "enforce_eager", True):
                for projection in ("qkv_proj", "o_proj"):
                    handles.append(_CompiledForward(getattr(layer, projection),
                                                    f"{names[-1]}.{projection}"))
    except Exception:
        for handle in handles:
            handle.remove()
        raise
    return tuple(handles)
