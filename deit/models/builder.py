from fvcore.common.registry import Registry

BACKBONE_REGISTRY = Registry("BACKBONE")


def build_model(cfg):
    """
    Build the whole model architecture, defined by ``cfg.MODEL.META_ARCHITECTURE``.
    Note that it does not load any weights from ``cfg``.
    """
    model_arch = cfg.type
    model = BACKBONE_REGISTRY.get(model_arch)(**cfg.kwargs)
    return model
