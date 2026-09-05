from pathlib import Path
import torch, pytest
from baselines.backbones import NonTemporalBackbone, TemporalBackbone



@pytest.fixture
def load_nontemp_data_and_model():
    def _load_nontemp_data_and_model(image_level=True):
        model = NonTemporalBackbone(image_level=image_level)
        x = torch.randn(4, 3, 224, 224)
        model.eval()
        with torch.no_grad():
            feats = model.feature_extractor(x)
            out = model.classifier(feats)

        return feats, out, model

    return _load_nontemp_data_and_model

class TestNonTemporalBackboneShapes:
    def test_output_shape_group_activity(self, load_nontemp_data_and_model):
        _, out, model= load_nontemp_data_and_model(image_level = True)

        assert out.shape == (4, model.num_classes)


    def test_output_shape_player_action(self, load_nontemp_data_and_model):
        _, out, model = load_nontemp_data_and_model(image_level=False)

        assert out.shape == (4, model.num_classes)


    def test_feature_extractor_output_shape(self, load_nontemp_data_and_model):
        feats, _, model = load_nontemp_data_and_model(image_level = True)

        assert feats.ndim == 2
        assert feats.shape == (4, model.feat_dim)



@pytest.fixture
def load_temp_data_and_model():
    def _load_temp_data_and_model(image_level=True, hidden_size=512):
        model = TemporalBackbone(image_level=image_level, hidden_size=hidden_size)
        model.eval()

        if image_level:
            B, T, CH, H, W = 4, 9, 3, 224, 224
            x = torch.randn(B, T, CH, H, W)
            with torch.no_grad():
                resnet_feats = model.feature_extractor(x.view(B * T, CH, H, W))
                _, (lstm_h_st, _) = model.lstm(resnet_feats.view(B, T, -1))
                lstm_h_st = lstm_h_st[-1]                 # (B, hidden_size)
                out = model.classifier(lstm_h_st)
        else:
            B, T, P, CH, H, W = 4, 9, 12, 3, 224, 224
            x = torch.randn(B, T, P, CH, H, W)
            with torch.no_grad():
                resnet_feats = model.feature_extractor(x.view(B * T * P, CH, H, W))
                resnet_feats = resnet_feats.view(B, T, P, -1).permute(0, 2, 1, 3).reshape(B * P, T, -1)
                _, (lstm_h_st, _) = model.lstm(resnet_feats)
                lstm_h_st = lstm_h_st[-1]                 # (B*P, hidden_size)
                out = model.classifier(lstm_h_st)
                out = out.view(B, P, -1)

        return (resnet_feats, lstm_h_st), out, model

    return _load_temp_data_and_model


class TestTemporalBackboneShapes:
    def test_output_shape_group_activity(self, load_temp_data_and_model):
        _, out, model = load_temp_data_and_model(image_level=True, hidden_size=512)
        assert out.shape == (4, model.num_classes)

    def test_output_shape_player_action(self, load_temp_data_and_model):
        _, out, model = load_temp_data_and_model(image_level=False, hidden_size=512)
        assert out.shape == (4, 12, model.num_classes)

    def test_temporal_full_image_feature_shape(self, load_temp_data_and_model):
        (resnet_feats, lstm_h_st), out, model = load_temp_data_and_model(image_level=True, hidden_size=512)
        assert resnet_feats.shape == (4 * 9, model.feat_dim)
        assert lstm_h_st.shape == (4, model.lstm.hidden_size)
        assert out.shape == (4, model.num_classes)

    def test_temporal_crops_feature_shape(self, load_temp_data_and_model):
        (resnet_feats, lstm_h_st), out, model = load_temp_data_and_model(image_level=False, hidden_size=512)
        assert resnet_feats.shape == (4 * 12, 9, model.feat_dim)
        assert lstm_h_st.shape == (4 * 12, model.lstm.hidden_size)
        assert out.shape == (4, 12, model.num_classes)