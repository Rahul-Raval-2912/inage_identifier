import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image
import requests
from torchvision.models import resnet18

model = resnet18(pretrained=True)
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

labels = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_labels = requests.get(labels).text.split("\n")

def predict(image):
    img_t = transform(image).unsqueeze(0)
    with torch.no_grad():
        pred = model(img_t).argmax().item()
    return imagenet_labels[pred]


iface = gr.Interface(fn=predict,inputs=gr.Image(type="pil"),outputs="text",title="Resnet18 image classifier")
iface.launch()