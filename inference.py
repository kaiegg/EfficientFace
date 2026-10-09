import torch
import torchvision.transforms as transforms
from PIL import Image
from models.EfficientFace import EfficientFace  # โครงสร้างโมเดลของคุณ

class EfficientFacePredictor:
    def __init__(self, checkpoint_path="checkpoint/[09-30]-[05-21]-model_best.pth.tar"):
        # 1. กำหนด Device (GPU/CPU)
        self.device = torch.device("cuda" if torch.cuda.device.is_available() else "cpu")

        # 2. โหลดสถาปัตยกรรมโมเดล
        self.model = EfficientFace()

        # 3. โหลด Weights จากไฟล์ในโฟลเดอร์ checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # 4. กำหนด Transformation ของรูปภาพ (ปรับขนาด / Normalize ให้ตรงกับตอนเทรน)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 5. ชื่อ Classes / Emotions
        self.labels = ['Surprise', 'Fear', 'Disgust', 'Happiness', 'Sadness', 'Anger', 'Neutral']

    def predict(self, image_path):
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)
            _, predicted = torch.max(outputs, 1)

        return self.labels[predicted.item()]
