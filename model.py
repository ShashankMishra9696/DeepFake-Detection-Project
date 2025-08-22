# Deepfake_detection/model.py

import torch
from facenet_pytorch import InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    model = InceptionResnetV1(
        pretrained='vggface2',
        classify=True,
        num_classes=1,
        device=DEVICE
    )
    checkpoint = torch.load("resnetinceptionv1_epoch_32.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model
