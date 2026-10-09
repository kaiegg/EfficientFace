import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from models import EfficientFace

class EfficientFacePredictor:
    def __init__(self, checkpoint_path='./checkpoint/[09-30]-[05-21]-model_best.pth.tar'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.categories = ['0_Neutral', '1_Hapiness', '2_Sadness', '3_Surprise', '4_Fear', '5_Disgust', '6_Anger']

        # 1. โหลดโมเดลผ่านฟังก์ชัน efficient_face()
        self.model = EfficientFace.efficient_face()
        self.model.fc = nn.Linear(1024, len(self.categories))

        # 2. โหลด Checkpoint และจัดการ prefix 'module.'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        new_state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)

        self.model = self.model.to(self.device)
        self.model.eval()

        # 3. Transform ที่ใช้ค่า mean/std เฉพาะของคุณ
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.57535914, 0.44928582, 0.40079932],
                std=[0.20735591, 0.18981615, 0.18132027]
            )
        ])

    def predict(self, image_path):
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            predictions = F.softmax(outputs, dim=1)[0].cpu().numpy()

        predicted_class_idx = predictions.argmax()
        predicted_emotion = self.categories[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx] * 100)

        breakdown = {cat: float(score * 100) for cat, score in zip(self.categories, predictions)}

        return {
            "emotion": predicted_emotion,
            "confidence": confidence,
            "breakdown": breakdown
        }
