
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor


class TeacherNetwork(nn.Module):
    """
    Extracteur de features ResNet-18 pretrained ImageNet.
    Totalement gelé : ses poids ne changent JAMAIS.

    Sorties multi-échelle :
        'layer2' → [B, 128, H/8,  W/8 ]
        'layer3' → [B, 256, H/16, W/16]
    """

    def __init__(self):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "feat_layer2",
                "layer3": "feat_layer3",
            },
        )

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.extractor(x)


class StudentNetwork(nn.Module):
    """
    Miroir du Teacher mais initialisé aléatoirement.
    Apprend à reproduire les features du Teacher sur des images normales.
    Sur une image anormale, l'écart Teacher/Student révèle le défaut.

    Sorties identiques au Teacher :
        'layer2' → [B, 128, H/8,  W/8 ]
        'layer3' → [B, 256, H/16, W/16]
    """

    def __init__(self):
        super().__init__()
        # pretrained=False → poids aléatoires intentionnels
        backbone = models.resnet18(weights=None)

        self.extractor = create_feature_extractor(
            backbone,
            return_nodes={
                "layer2": "feat_layer2",
                "layer3": "feat_layer3",
            },
        )

    def forward(self, x):
        return self.extractor(x)



def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_models(device: str = "cuda"):
    """
    Instancie Teacher et Student et les envoie sur le bon device.

    Returns:
        teacher (eval, gelé), student (train)
    """
    teacher = TeacherNetwork().to(device)
    student = StudentNetwork().to(device)

    teacher.eval()

    print(f"[Model] Teacher params (gelés)      : {sum(p.numel() for p in teacher.parameters()):,}")
    print(f"[Model] Student params (entraînable): {count_parameters(student):,}")

    return teacher, student