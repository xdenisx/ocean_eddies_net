"""Optional model integration tests; skipped when the requested backend is absent."""
import importlib.util
import pytest
import torch
from ocean_eddy.models import ModelSpec,build_model,freeze_batchnorm


@pytest.mark.parametrize('architecture,backend,module',[
    ('deeplabv3plus_resnet101','smp','segmentation_models_pytorch'),
    ('unetpp_effb4','smp','segmentation_models_pytorch'),
    ('segformer_b0','smp','segmentation_models_pytorch'),
    ('segformer_b0','hf','transformers'),
    ('upernet_swin_t','hf','transformers'),
    ('hrnet_w32','timm','timm'),
])
def test_optional_forward_multiclass(architecture,backend,module):
    if importlib.util.find_spec(module) is None:
        pytest.skip(f'{module} is not installed')
    spec=ModelSpec(architecture,1,3,backend)
    model=build_model(spec,False)
    model.train();freeze_batchnorm(model)
    x=torch.randn(1,1,64,64)
    result=model(x)
    assert result.shape==(1,3,64,64)
    result.mean().backward()
