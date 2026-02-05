# local_radiology_agent.py
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from transformers import T5ForConditionalGeneration, BertTokenizer
from transformers.modeling_outputs import BaseModelOutput

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# R2Gen Model Definition
# -------------------------------
class R2GenModel(nn.Module):
    def __init__(self, model_name="t5-small", device="cuda", dropout_prob=0.1):
        super().__init__()
        self.device = device

        # Tokenizer + Language model
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

        # Visual extractor (ResNet101)
        self.visual_extractor = models.resnet101(pretrained=True)
        self.visual_extractor.fc = nn.Linear(
            self.visual_extractor.fc.in_features,
            self.model.config.d_model
        )
        self.visual_extractor = self.visual_extractor.to(device)

        self.dropout = nn.Dropout(dropout_prob)

    def extract_visual_features(self, images):
        images = images.to(self.device)
        visual_features = self.visual_extractor(images)
        visual_features = self.dropout(visual_features)

        # Expand to sequence for T5 encoder
        visual_features = visual_features.unsqueeze(1)
        visual_features = visual_features.expand(
            visual_features.size(0), 512, visual_features.size(2)
        )
        return visual_features

# -------------------------------
# Image preprocessing
# -------------------------------
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    return image_transform(image).unsqueeze(0)

# -------------------------------
# Vision Report Agent (FINAL)
# -------------------------------
class VisionReportAgent:
    """
    Vision Agent:
    Generates a radiology report directly from an X-ray image
    using a trained R2Gen-style image-to-text model.
    """

    def __init__(self, model_ckpt, device=device):
        self.device = device
        self.model = R2GenModel(device=device).to(device)
        self.model.load_state_dict(
            torch.load(model_ckpt, map_location=device)
        )
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        print("✅ R2Gen Vision Agent loaded successfully")

    @torch.no_grad()
    def generate_report(self, image_path, max_length=150):
     image = preprocess_image(image_path).to(self.device)

     input_ids = self.tokenizer.encode(
        "generate report:",
        return_tensors="pt"
    ).to(self.device)

     visual_features = self.model.extract_visual_features(image)

     encoder_outputs = BaseModelOutput(
        last_hidden_state=visual_features
    )

     generated_ids = self.model.model.generate(
        input_ids=input_ids,
        encoder_outputs=encoder_outputs,
        max_length=max_length,
        num_beams=5,
        do_sample=False
    )

     report = self.tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

     return report


# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    image_path = (
        r"C:\Users\User\OneDrive\Desktop\gui\XMedAgent\data\iu_xray\images\CXR2867_IM-1274/0.png"
    )

    model_ckpt = (
        r"C:\Users\User\OneDrive\Desktop\gui\XMedAgent\model_weights\model_one.pth"
    )

    vision_agent = VisionReportAgent(model_ckpt=model_ckpt)

    report = vision_agent.generate_report(image_path)

    print("\n🩺 Generated Vision Radiology Report:\n")
    print(report)
