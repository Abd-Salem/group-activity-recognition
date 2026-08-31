from pathlib import Path
import torch, pytest
from .backbones import NonTemporalBackbone, TemporalBackbone



@pytest.fixture
def load_nontemp_data_and_model():
    def _load_nontemp_data_and_model(image_level=True):
        model = NonTemporalBackbone(image_level=image_level)
        x = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            feats = model.feature_extractor(x)
            out = model(x)

        return feats, out, model

    return _load_nontemp_data_and_model

class TestNonTemporalBackboneShapes:
    def test_output_shape_group_activity(self, load_nontemp_data_and_model):
        _, out, _= load_nontemp_data_and_model(image_level = True)

        assert out.shape == (4, 8)


    def test_output_shape_player_action(self, load_nontemp_data_and_model):
        _, out, _ = load_nontemp_data_and_model(image_level=False)

        assert out.shape == (4, 9)


    def test_feature_extractor_output_shape(self, load_nontemp_data_and_model):
        _, feats, model = load_nontemp_data_and_model(image_level = True)

        assert feats.ndim == 2
        assert feats.shape == (4, model.feat_dim)





@pytest.fixture
def load_temp_data_and_model():
    def _load_temp_data_and_model(image_level=True, hidden_size=512):
        model = TemporalBackbone(image_level=image_level, hidden_size=hidden_size)
        if image_level:
            b ,t ,ch ,h ,w = 4, 9, 3, 224, 224
            x = torch.randn(b ,t ,ch ,h ,w)

            with torch.no_grad():
                resnet_feats = model.feature_extractor(x)
        else:
            b ,t ,ch ,h ,w = 4, 9, 12, 3, 224, 224
            x = torch.randn(b, t, ch, h, w)
        model.eval()
        with torch.no_grad():
            feats = model.feature_extractor(x.view())
            out = model(x)
        return feats, out, model

    return _load_temp_data_and_model

class TestTemporalBackboneShapes:
    def test_output_shape_group_activity(self, load_temp_data_and_model):
        _, out, _ = load_temp_data_and_model(image_level=True, hidden_size=512)

        assert out.shape == (4, 8)

    def test_output_shape_player_action(self):
        _, out, _ = load_temp_data_and_model(image_level=True, hidden_size=512)

        assert out.shape == (4, 9)

