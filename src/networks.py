import torch
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    BertModel,
    DistilBertModel,
    GPT2Model,
    T5EncoderModel,
    BartModel,
    DebertaV2Model,
    LongformerModel,
    AlbertModel
)
from transformers import GPT2Config, GPT2Model

def remove_batch_norm_from_resnet(model):
    fuse = torch.nn.utils.fusion.fuse_conv_bn_eval
    model.eval()

    model.conv1 = fuse(model.conv1, model.bn1)
    model.bn1 = Identity()

    for name, module in model.named_modules():
        if name.startswith("layer") and len(name) == 6:
            for b, bottleneck in enumerate(module):
                for name2, module2 in bottleneck.named_modules():
                    if name2.startswith("conv"):
                        bn_name = "bn" + name2[-1]
                        setattr(bottleneck, name2,
                                fuse(module2, getattr(bottleneck, bn_name)))
                        setattr(bottleneck, bn_name, Identity())
                if isinstance(bottleneck.downsample, torch.nn.Sequential):
                    bottleneck.downsample[0] = fuse(bottleneck.downsample[0],
                                                    bottleneck.downsample[1])
                    bottleneck.downsample[1] = Identity()
    model.train()
    return model


class Identity(nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MLP(nn.Module):
    """Just  an MLP"""
    def __init__(self, n_inputs, n_outputs, hparams):
        super(MLP, self).__init__()
        self.input = nn.Linear(n_inputs, hparams['mlp_width'])
        self.dropout = nn.Dropout(hparams['mlp_dropout'])
        self.hiddens = nn.ModuleList([
            nn.Linear(hparams['mlp_width'], hparams['mlp_width'])
            for _ in range(hparams['mlp_depth']-2)])
        self.output = nn.Linear(hparams['mlp_width'], n_outputs)
        self.n_outputs = n_outputs

    def forward(self, x):
        x = self.input(x)
        x = self.dropout(x)
        x = F.relu(x)
        for hidden in self.hiddens:
            x = hidden(x)
            x = self.dropout(x)
            x = F.relu(x)
        x = self.output(x)
        return x

class TextModelFeatureWrapper(torch.nn.Module):

    def __init__(self, model, hparams):
        super().__init__()
        self.model = model
                # Try hidden_size first, then fall back to d_model
        model_hidden_dim = getattr(model.config, "hidden_size", None)
        if model_hidden_dim is None:
            model_hidden_dim = getattr(model.config, "d_model", None)

        if model_hidden_dim is None:
            raise ValueError(
                "No 'hidden_size' or 'd_model' attribute found in model config. "
                f"Check model config: {model.config}"
            )

        self.n_outputs = model_hidden_dim

        # Default dropout from config, e.g. hidden_dropout_prob (BERT) or similar
        default_dropout = getattr(model.config, "hidden_dropout_prob", 0.0)
        # Use user-provided dropout if it's non-zero, else fallback to model’s default
        classifier_dropout = (
            hparams['last_layer_dropout'] if hparams.get('last_layer_dropout', 0.0) != 0.0
            else default_dropout
        )
        self.dropout = nn.Dropout(classifier_dropout)

    def forward(self, x):
        output = self.model(**x)

        # Some models (BERT) have pooler_output, others (GPT2, DistilBERT, T5) do not
        if hasattr(output, 'pooler_output') and output.pooler_output is not None:
            return self.dropout(output.pooler_output)
        else:
            # Fallback: use the [CLS]-token (first token) representation
            # last_hidden_state shape: (batch_size, seq_len, hidden_dim)
            return self.dropout(output.last_hidden_state[:, 0, :])


class ModelLoader(torch.nn.Module):
    def __init__(self, input_shape, hparams):
        super(ModelLoader, self).__init__()
        """
        model_type: str, type of model to load ('resnet18', 'resnet50', 'densenet121', 'convnext_tiny')
        input_shape: tuple, the shape of the input (should include the transfer flag as last element)
        hparams: dict, hyperparameters such as number of output classes
        """
        self.model_type = hparams['model_arch']
        self.input_shape = input_shape
        self.transfer = hparams['transfer']  # Assuming transfer is the last element in input_shape
        self.num_classes = hparams.get('num_classes', 2)
        self.hparams = hparams
        self.network = self._load_model()

        # Freeze BatchNorm layers
        self.freeze_bn()

        self.dropout = nn.Dropout(hparams['resnet_dropout'])

    def _load_model(self):
        # text
        if self.model_type in [
            "bert-base-uncased",
            "bert-large-uncased",
            "allenai/scibert_scivocab_uncased",
            "roberta-base",
            "roberta-large",
            "dmis-lab/biobert-base-cased-v1.1",
            "nlpaueb/legal-bert-base-uncased",
            "yiyanghkust/finbert-tone"
        ]:
            # Any HF model with hidden_size can be loaded via AutoModel here
            model = AutoModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs

        elif self.model_type in ["albert-base-v2", "albert-large-v2", "albert-xlarge-v2", "albert-xxlarge-v2"]:
            model = AlbertModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["microsoft/deberta-base", "microsoft/deberta-large", "microsoft/deberta-v2-xxlarge",
                                 "microsoft/deberta-v3-base", "microsoft/deberta-v3-large", "microsoft-mdeberta-v3-base", "microsoft-deberta-base-mnli"]:
            model = DebertaV2Model.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["allenai/longformer-base-4096", "allenai/longformer-large-4096"]:
            model = LongformerModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["distilbert-base-uncased", "distilbert-base-cased", "distilbert-base-uncased-distilled-squad", "distilbert-base-cased-distilled-squad"]:
            model = DistilBertModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["t5-small", "t5-base", "t5-large", "t5-3l", "t5-11l"]:
            model = T5EncoderModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["facebook/bart-base", "facebook/bart-large", "facebook/bart-large-cnn"]:
            model = BartModel.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        elif self.model_type in ["gpt2", "gpt-neo-125m", "gpt-neo-1.3b", "gpt-neo-2.7b"]:
            model = GPT2Model.from_pretrained(self.model_type)
            model = TextModelFeatureWrapper(model, self.hparams)
            self.n_outputs = model.n_outputs
        # image
        elif self.model_type == 'resnet18':
            weights = self.hparams.get('weights', models.ResNet18_Weights.DEFAULT)
            model = models.resnet18(weights=weights)
            self.n_outputs = 512
        elif self.model_type == 'resnet34':
            weights = self.hparams.get('weights', models.ResNet34_Weights.DEFAULT)
            model = models.resnet34(weights=weights)
            self.n_outputs = 512
        elif self.model_type == 'resnet50':
            weights = self.hparams.get('weights', models.ResNet50_Weights.DEFAULT)
            model = models.resnet50(weights=weights)
            self.n_outputs = 2048
        elif self.model_type == 'resnet101':
            weights = self.hparams.get('weights', models.ResNet101_Weights.DEFAULT)
            model = models.resnet101(weights=weights)
            self.n_outputs = 2048
        elif self.model_type == 'resnet152':
            weights = self.hparams.get('weights', models.ResNet152_Weights.DEFAULT)
            model = models.resnet152(weights=weights)
            self.n_outputs = 2048

        elif self.model_type == 'densenet121':
            weights = self.hparams.get('weights', models.DenseNet121_Weights.DEFAULT)
            model = models.densenet121(weights=weights)
            self.n_outputs = 1024
        elif self.model_type == 'densenet169':
            weights = self.hparams.get('weights', models.DenseNet169_Weights.DEFAULT)
            model = models.densenet169(weights=weights)
            self.n_outputs = 1664
        elif self.model_type == 'densenet161':
            weights = self.hparams.get('weights', models.DenseNet161_Weights.DEFAULT)
            model = models.densenet161(weights=weights)
            self.n_outputs = 2208
        elif self.model_type == 'densenet201':
            weights = self.hparams.get('weights', models.DenseNet201_Weights.DEFAULT)
            model = models.densenet201(weights=weights)
            self.n_outputs = 1920

        elif self.model_type == 'mobilenet_v2':
            weights = self.hparams.get('weights', models.MobileNet_V2_Weights.DEFAULT)
            model = models.mobilenet_v2(weights=weights)
            self.n_outputs = 1280
        elif self.model_type == 'mobilenet_v3_small':
            weights = self.hparams.get('weights', models.MobileNet_V3_Small_Weights.DEFAULT)
            model = models.mobilenet_v3_small(weights=weights)
            self.n_outputs = 1000
        elif self.model_type == 'mobilenet_v3_large':
            weights = self.hparams.get('weights', models.MobileNet_V3_Large_Weights.DEFAULT)
            model = models.mobilenet_v3_large(weights=weights)
            self.n_outputs = 1000

        elif self.model_type == 'efficientnet_b0':
            weights = self.hparams.get('weights', models.EfficientNet_B0_Weights.DEFAULT)
            model = models.efficientnet_b0(weights=weights)
            self.n_outputs = 1280
        elif self.model_type == 'efficientnet_b1':
            weights = self.hparams.get('weights', models.EfficientNet_B1_Weights.DEFAULT)
            model = models.efficientnet_b1(weights=weights)
            self.n_outputs = 1280
        elif self.model_type == 'efficientnet_b3':
            weights = self.hparams.get('weights', models.EfficientNet_B3_Weights.DEFAULT)
            model = models.efficientnet_b3(weights=weights)
            self.n_outputs = 1536
        elif self.model_type == 'efficientnet_b7':
            weights = self.hparams.get('weights', models.EfficientNet_B7_Weights.DEFAULT)
            model = models.efficientnet_b7(weights=weights)
            self.n_outputs = 2560

        elif self.model_type == 'convnext_tiny':
            weights = self.hparams.get('weights', models.ConvNeXt_Tiny_Weights.DEFAULT)
            model = models.convnext_tiny(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'convnext_small':
            weights = self.hparams.get('weights', models.ConvNeXt_Small_Weights.DEFAULT)
            model = models.convnext_small(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'convnext_base':
            weights = self.hparams.get('weights', models.ConvNeXt_Base_Weights.DEFAULT)
            model = models.convnext_base(weights=weights)
            self.n_outputs = 1024
        elif self.model_type == 'convnext_large':
            weights = self.hparams.get('weights', models.ConvNeXt_Large_Weights.DEFAULT)
            model = models.convnext_large(weights=weights)
            self.n_outputs = 1536

        elif self.model_type == 'vit_b_16':
            weights = self.hparams.get('weights', models.ViT_B_16_Weights.DEFAULT)
            model = models.vit_b_16(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'vit_b_32':
            weights = self.hparams.get('weights', models.ViT_B_32_Weights.DEFAULT)
            model = models.vit_b_32(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'vit_l_16':
            weights = self.hparams.get('weights', models.ViT_L_16_Weights.DEFAULT)
            model = models.vit_l_16(weights=weights)
            self.n_outputs = 1024

        elif self.model_type == 'swin_t':
            weights = self.hparams.get('weights', models.Swin_T_Weights.DEFAULT)
            model = models.swin_t(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'swin_s':
            weights = self.hparams.get('weights', models.Swin_S_Weights.DEFAULT)
            model = models.swin_s(weights=weights)
            self.n_outputs = 768
        elif self.model_type == 'swin_b':
            weights = self.hparams.get('weights', models.Swin_B_Weights.DEFAULT)
            model = models.swin_b(weights=weights)
            self.n_outputs = 1024

        elif self.model_type == 'regnet_y_400mf':
            weights = self.hparams.get('weights', models.RegNet_Y_400MF_Weights.DEFAULT)
            model = models.regnet_y_400mf(weights=weights)
            self.n_outputs = 440
        elif self.model_type == 'regnet_y_800mf':
            weights = self.hparams.get('weights', models.RegNet_Y_800MF_Weights.DEFAULT)
            model = models.regnet_y_800mf(weights=weights)
            self.n_outputs = 784
        elif self.model_type == 'regnet_y_1_6gf':
            weights = self.hparams.get('weights', models.RegNet_Y_1_6GF_Weights.DEFAULT)
            model = models.regnet_y_1_6gf(weights=weights)
            self.n_outputs = 888
        elif self.model_type == 'regnet_y_3_2gf':
            weights = self.hparams.get('weights', models.RegNet_Y_3_2GF_Weights.DEFAULT)
            model = models.regnet_y_3_2gf(weights=weights)
            self.n_outputs = 1512
        elif self.model_type == 'regnet_y_8gf':
            weights = self.hparams.get('weights', models.RegNet_Y_8GF_Weights.DEFAULT)
            model = models.regnet_y_8gf(weights=weights)
            self.n_outputs = 2016

        # --- Additional models ---
        elif self.model_type == 'alexnet':
            weights = self.hparams.get('weights', models.AlexNet_Weights.DEFAULT)
            model = models.alexnet(weights=weights)
            self.n_outputs = 4096
        elif self.model_type == 'vgg11':
            weights = self.hparams.get('weights', models.VGG11_Weights.DEFAULT)
            model = models.vgg11(weights=weights)
            self.n_outputs = 4096
        elif self.model_type == 'vgg13':
            weights = self.hparams.get('weights', models.VGG13_Weights.DEFAULT)
            model = models.vgg13(weights=weights)
            self.n_outputs = 4096
        elif self.model_type == 'vgg16':
            weights = self.hparams.get('weights', models.VGG16_Weights.DEFAULT)
            model = models.vgg16(weights=weights)
            self.n_outputs = 4096
        elif self.model_type == 'vgg19':
            weights = self.hparams.get('weights', models.VGG19_Weights.DEFAULT)
            model = models.vgg19(weights=weights)
            self.n_outputs = 4096
        elif self.model_type == 'squeezenet1_0':
            weights = self.hparams.get('weights', models.SqueezeNet1_0_Weights.DEFAULT)
            model = models.squeezenet1_0(weights=weights)
            self.n_outputs = 512
        elif self.model_type == 'squeezenet1_1':
            weights = self.hparams.get('weights', models.SqueezeNet1_1_Weights.DEFAULT)
            model = models.squeezenet1_1(weights=weights)
            self.n_outputs = 512
        elif self.model_type == 'inception_v3':
            weights = self.hparams.get('weights', models.Inception_V3_Weights.DEFAULT)
            # Note: inception_v3 expects 299x299 inputs and has aux_logits by default.
            model = models.inception_v3(weights=weights, aux_logits=False)
            self.n_outputs = 2048
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        # Modify the first convolution layer if the number of input channels (nc) is not 3.
        if not isinstance(model, TextModelFeatureWrapper):
            model = self._modify_first_conv(model, self.input_shape[0])

        # Replace final classification layers with Identity to obtain feature vectors.
        if self.model_type.startswith('resnet') or self.model_type.startswith('regnet') or self.model_type.startswith('inception'):
            model.fc = nn.Identity()
        elif self.model_type.startswith('densenet'):
            model.classifier = nn.Identity()
        elif self.model_type.startswith('mobilenet') or self.model_type.startswith('efficientnet'):
            model.classifier[1] = nn.Identity()
        elif self.model_type.startswith('convnext'):
            model.classifier[2] = nn.Identity()
        elif self.model_type.startswith('vit'):
            model.heads.head = nn.Identity()
        elif self.model_type.startswith('swin'):
            model.head = nn.Identity()
        elif self.model_type.startswith('alexnet'):
            model.classifier[-1] = nn.Identity()
        elif self.model_type.startswith('vgg'):
            model.classifier[-1] = nn.Identity()
        elif self.model_type.startswith('squeezenet'):
            # Replace the conv layer in the classifier to get features.
            model.classifier[1] = nn.Identity()

        # Freeze all parameters for transfer learning if specified.
        if self.transfer:
            for param in model.parameters():
                param.requires_grad = False

        return model

    def _modify_first_conv(self, model, nc):
        """Modify the first convolution layer to accept nc channels if needed."""
        if nc == 3:
            return model

        # Case 1: Models with a conv1 attribute (e.g., ResNets, RegNets)
        if hasattr(model, 'conv1'):
            conv = model.conv1
            tmp = conv.weight.data.clone()
            new_conv = nn.Conv2d(
                nc,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=(conv.bias is not None)
            )
            with torch.no_grad():
                for i in range(nc):
                    new_conv.weight.data[:, i, :, :] = tmp[:, i % conv.in_channels, :, :]
            model.conv1 = new_conv
            return model

        # Case 2: DenseNets (first conv is at model.features.conv0)
        if self.model_type.startswith('densenet') and hasattr(model.features, 'conv0'):
            conv = model.features.conv0
            tmp = conv.weight.data.clone()
            new_conv = nn.Conv2d(
                nc,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=(conv.bias is not None)
            )
            with torch.no_grad():
                for i in range(nc):
                    new_conv.weight.data[:, i, :, :] = tmp[:, i % conv.in_channels, :, :]
            model.features.conv0 = new_conv
            return model

        # Case 3: For MobileNet/EfficientNet assuming first conv is at model.features[0][0]
        if (self.model_type.startswith('mobilenet') or self.model_type.startswith('efficientnet')) and isinstance(model.features, nn.Sequential):
            first_block = model.features[0]
            if isinstance(first_block, nn.Sequential):
                conv = first_block[0]
                tmp = conv.weight.data.clone()
                new_conv = nn.Conv2d(
                    nc,
                    conv.out_channels,
                    kernel_size=conv.kernel_size,
                    stride=conv.stride,
                    padding=conv.padding,
                    bias=(conv.bias is not None)
                )
                with torch.no_grad():
                    for i in range(nc):
                        new_conv.weight.data[:, i, :, :] = tmp[:, i % conv.in_channels, :, :]
                first_block[0] = new_conv
                return model

        # Case 4: For models like AlexNet or VGG, the first conv is typically in model.features[0]
        if hasattr(model, 'features') and isinstance(model.features, nn.Sequential) and isinstance(model.features[0], nn.Conv2d):
            conv = model.features[0]
            tmp = conv.weight.data.clone()
            new_conv = nn.Conv2d(
                nc,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                bias=(conv.bias is not None)
            )
            with torch.no_grad():
                for i in range(nc):
                    new_conv.weight.data[:, i, :, :] = tmp[:, i % conv.in_channels, :, :]
            model.features[0] = new_conv
            return model

        # If none of the above apply, return the model unmodified.
        return model

    def forward(self, x):
        """Encode x into a feature vector of size n_outputs."""
        return self.dropout(self.network(x))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters.
        """
        super().train(mode)
        self.freeze_bn()

    def freeze_bn(self):
        """Freeze all BatchNorm2d layers."""
        for m in self.network.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_model(self):
        return self.model

class MNIST_CNN(nn.Module):
    """
    Hand-tuned architecture for MNIST.
    Weirdness I've noticed so far with this architecture:
    - adding a linear layer after the mean-pool in features hurts
        RotatedMNIST-100 generalization severely.
    """
    n_outputs = 128

    def __init__(self, input_shape):
        super(MNIST_CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape[0], 64, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, padding=1)

        self.bn0 = nn.GroupNorm(8, 64)
        self.bn1 = nn.GroupNorm(8, 128)
        self.bn2 = nn.GroupNorm(8, 128)
        self.bn3 = nn.GroupNorm(8, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn0(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.bn2(x)

        x = self.conv4(x)
        x = F.relu(x)
        x = self.bn3(x)

        x = self.avgpool(x)
        x = x.view(len(x), -1)
        return x


class ContextNet(nn.Module):
    def __init__(self, input_shape):
        super(ContextNet, self).__init__()

        # Keep same dimensions
        padding = (5 - 1) // 2
        self.context_net = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 5, padding=padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 5, padding=padding),
        )

    def forward(self, x):
        return self.context_net(x)


def Featurizer(input_shape, hparams):
    """Auto-select an appropriate featurizer for the given input shape."""
    # if len(input_shape) == 1:
    #     return MLP(input_shape[0], hparams["mlp_width"], hparams)
    # elif input_shape[1:3] == (28, 28):
    #     return MNIST_CNN(input_shape)
    # elif input_shape[1:3] == (32, 32):
    #     return wide_resnet.Wide_ResNet(input_shape, 16, 2, 0.)
    # elif input_shape[1:3] == (224, 224):
    return ModelLoader(input_shape, hparams)
    # else:
    #     raise NotImplementedError


def Classifier(in_features, out_features, is_nonlinear=False):
    if is_nonlinear:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features, in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 2, in_features // 4),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features // 4, out_features))
    else:
        return torch.nn.Linear(in_features, out_features)
