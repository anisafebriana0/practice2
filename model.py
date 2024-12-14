import torch
import supervision as sv
import transformers
import pytorch_lightning
import os
import torchvision

dataset = "C:/Dataset/paralysis face.v7i.coco"

ANNOTATION_FILE_NAME = "_annotations.coco.json"
TRAIN_DIRECTORY = os.path.join(dataset, "train")
VAL_DIRECTORY = os.path.join(dataset, "valid")
TEST_DIRECTORY = os.path.join(dataset, "test")

#class for load dataset
class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__( self, image_directory_path:str, image_processor, train: bool = True):
        annotation_file_path = os.path.join(image_directory_path, ANNOTATION_FILE_NAME)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self , idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations':annotations}
        encoding = self.image_processor(images = images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    

from transformers import DetrImageProcessor
image_processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

TRAIN_DATASET = CocoDetection(image_directory_path=TRAIN_DIRECTORY, image_processor=image_processor, train=True)
VAL_DATASET = CocoDetection(image_directory_path=VAL_DIRECTORY, image_processor=image_processor, train=False)
TEST_DATASET = CocoDetection(image_directory_path=TEST_DIRECTORY, image_processor=image_processor, train=False)

print("Number of train image : ", len(TRAIN_DATASET))
print("Number of validation image : ", len(VAL_DATASET))
print("Number of test image : ", len(TEST_DATASET))

# prepare data before training

def collate_fn(batch):
    pixel_vales = []

    for item in batch:
        pixel_vales.append(item[0])

    
    encoding = image_processor.pad(pixel_vales, return_tensors="pt")

    labels = []
    for item in batch:
        labels.append(item[1])

    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }
from torch.utils.data import DataLoader
torch.set_float32_matmul_precision('medium')

categories = TRAIN_DATASET.coco.cats
print("Categories:")
print(categories)


id2label = {}
for k, v, in categories.items():
    id2label[k] = v['name']


print("id2label", id2label)
print(len(id2label))

print("===========")
TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn,batch_size=4, shuffle=True)
VAL_DATALOADER = DataLoader(dataset=VAL_DATASET, collate_fn=collate_fn,batch_size=4)


import pytorch_lightning as pl
from transformers import DetrForObjectDetection
import torch

CHECKPOINT = "facebook/detr-resnet-50"

class Detr(pl.LightningModule):
    def __init__(self, lr, lr_backbone, weight_decay):
        super().__init__()
        self.model = DetrForObjectDetection.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT,
            num_labels=len(id2label),
            ignore_mismatched_sizes=True
        )

        self.lr = lr
        self.lr_backbone = lr_backbone
        self.weight_decay = weight_decay

    def forward(self, pixel_values, pixel_mask):
        return self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
    
    def common_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        pixel_mask = batch["pixel_mask"]
        labels = [{k:v.to(self.device) for k,v in t.items()} for t in batch["labels"]]

        outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=labels)

        loss = outputs.loss
        loss_dict = outputs.loss_dict

        return loss , loss_dict
    
    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("training_loss", loss)
        for k, v in loss_dict.items():
            self.log("Train_" + k, v.item())
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, loss_dict = self.common_step(batch, batch_idx)
        self.log("validation_loss", loss)
        for k, v in loss_dict.items():
            self.log("Validation_" + k, v.item())
        
        return loss
    
    def configure_optimizers(self):
        param_dicts = [
            {
                "params":[p for n,p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params":[p for n,p in self.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": self.lr_backbone,
            },

        ]
        return torch.optim.AdamW(param_dicts, lr= self.lr, weight_decay=self.weight_decay)

    def train_dataloader(self):
        return TRAIN_DATALOADER
    
    def val_dataloader(self):
        return VAL_DATALOADER
    

model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4)

#Train the model
from pytorch_lightning import Trainer
log_dir = './my_DETR_log'

MAX_EPOCH = 1

trainer = Trainer(devices=1,
                  accelerator="cpu",
                  max_epochs=MAX_EPOCH,
                  accumulate_grad_batches=8,
                  log_every_n_steps=1,
                  default_root_dir=log_dir)

trainer.fit(model)

#create a saved folder:
MODEL_PATH ="./DETR-My-Model-1"
model.model.save_pretrained(MODEL_PATH)
# Save the processor to the model path
image_processor.save_pretrained(MODEL_PATH)

    