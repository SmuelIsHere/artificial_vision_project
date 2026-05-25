import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os

IMG_WIDTH = 224
IMG_HEIGHT = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
os.makedirs('./par_models', exist_ok=True)


from extract_annotations import extract_annotation

training_folder = './dataset/trainset'
train_label = './dataset/trainset.txt'
validation_folder = './dataset/validation' 
validation_label = './dataset/validation.txt'

annotations_training_list = extract_annotation(train_label)
annotations_validation_list = extract_annotation(validation_label)

# normalizzazione sui 3 canali dell'immagine
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

data_transforms = {
    'train':
    transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.RandomAffine(degrees=30, shear=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        normalize
    ]),
    'validation':
    transforms.Compose([
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        normalize
    ]),
}

# path delle immagini singole sul quale iterare
training_image_paths = [os.path.join(training_folder, filename) for filename in os.listdir(training_folder)]
validation_image_paths = [os.path.join(validation_folder, filename) for filename in os.listdir(validation_folder)]

training_image_paths.sort()
validation_image_paths.sort()

print(len(training_image_paths))
print(len(validation_image_paths))
print(len(annotations_training_list))
print(len(annotations_validation_list))


# estrapolazione informazioni sul dataset 
from CustomImageDataset import CustomImageDataset
from CustomRandomSampler import CustomRandomSampler

training_dataset = CustomImageDataset(training_image_paths, annotations_training_list, device, transform=data_transforms['train'])
validation_dataset = CustomImageDataset(validation_image_paths, annotations_validation_list, device, transform=data_transforms['validation'])

batch_size = 64
num_batch = len(training_dataset) // batch_size
print('num batch ', num_batch)
sampler = CustomRandomSampler(dataset=training_dataset, batch_size=batch_size, replacement=False)

train_dataloader = DataLoader(training_dataset, batch_sampler = sampler)
val_dataloader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)


from AttributeRecognitionModel import AttributeRecognitionModel

num_attributes = 3 
num_layers = 3

model = AttributeRecognitionModel(num_attributes=num_attributes)
model.to(device)

# Caricare i pesi salvati
checkpoint_path = 'best_model.pth'
if os.path.exists(checkpoint_path):
    print("Caricamento dei pesi da:", checkpoint_path)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device,weights_only=True))
else:
    print("Checkpoint non trovato. Partenza da zero.")

model.freeze_backbone_parameters()
model.unfreeze_last_layer_backbone(num_layers)
model.unfreeze_parameters()

from BinaryAsymmetricLoss import BinaryAsymmetricLoss
gender_weights, bag_weights, hat_weights = CustomImageDataset.make_weights(training_dataset, device)
loss_list = [
    BinaryAsymmetricLoss(ignore_index=-1, gamma_neg=gender_weights[0], gamma_pos=gender_weights[1]),
    BinaryAsymmetricLoss(ignore_index=-1, gamma_neg=bag_weights[0], gamma_pos=bag_weights[1]),
    BinaryAsymmetricLoss(ignore_index=-1, gamma_neg=hat_weights[0], gamma_pos=hat_weights[1])
]

from Accuracy import Accuracy
accuracy_list = [
    Accuracy(threshold=0.5),
    Accuracy(threshold=0.5),
    Accuracy(threshold=0.5)
]
optimizer = optim.Adam(model.parameters(), lr=1e-1) 

from tqdm import tqdm
def one_epoch(model, criterion_list, optimizer, train_loader, val_loader, epoch_num, accuracy_list, num_batch):

  model.train()

  task_acc = dict((i, []) for i in range(num_attributes))
  train_loss = torch.tensor(0.0, dtype = torch.float32, device = device)
  
  
  for i, (images, labels) in tqdm(enumerate(train_loader), desc="epoch {} - train".format(epoch_num)):

    images = images.to(device)
    labels = labels.to(device).long()
    optimizer.zero_grad()

    o = model(images)

    batch_loss = []
  
    for attr_index in range(num_attributes):
  
      target = labels[:, attr_index]
      attribute_predictions = o[attr_index] 
      loss = criterion_list[attr_index](attribute_predictions, target)

      acc = accuracy_list[attr_index](attribute_predictions, target)
      batch_loss.append(loss)

      task_acc[attr_index].append(acc)
    
    aggregated_loss = sum(batch_loss).to(device)
    train_loss += aggregated_loss
    aggregated_loss.backward()
    optimizer.step()

  train_loss = (train_loss / num_batch).item()
  task_acc = [sum(task_acc[i]) / len(task_acc[i]) for i in range(num_attributes)]
  train_accuracy = sum(task_acc) / num_attributes

  print("Training loss and accuracy : {:.7f}\t{:.4f}".format(train_loss, train_accuracy))
  print("Train gender acc : ", task_acc[0])
  print("Train bag acc : ", task_acc[1])
  print("Train hat acc : ", task_acc[2])

  model.eval()
  with torch.no_grad():
    val_loss = []
    task_acc = dict((i, []) for i in range(num_attributes))

    for images, labels in tqdm(val_loader, desc="epoch {} - validation".format(epoch_num)):
      images = images.to(device)
      labels = labels.to(device).long()

      o = model(images)

      batch_loss = []

      for attr_index in range(num_attributes):
        target = labels[:, attr_index]
        attribute_predictions = o[attr_index]

        loss = criterion_list[attr_index](attribute_predictions, target)
        acc= accuracy_list[attr_index](attribute_predictions, target)

        batch_loss.append(loss)
        task_acc[attr_index].append(acc)

      aggregated_loss = sum(batch_loss)
      val_loss.append(aggregated_loss)
    
    task_acc = [sum(task_acc[i]) / len(task_acc[i]) for i in range(num_attributes)]
    
    val_accuracy = sum(task_acc) / num_attributes
    val_loss = sum(val_loss)/len(val_loss)
    print("Validation loss and accuracy : ")
    print(val_loss)
    print(val_accuracy)
    print("Validation gender accuracy : ", task_acc[0])
    print("Validation bag accuracy : ", task_acc[1])
    print("Validation hat accuracy : ", task_acc[2])
    
  return val_loss, val_accuracy

# loop di addestramento
device = torch.device("cuda")
EARLY_STOPPIMG_PATIENCE = 5
early_stopping_counter = EARLY_STOPPIMG_PATIENCE
epochs = 15
min_val_loss = 1e3

val_losses = torch.zeros(epochs)
val_accuracies = torch.zeros(epochs)

for e in range(epochs):
  print("EPOCH {}".format(e))

  val_loss, val_accuracy = one_epoch(model, loss_list, optimizer, train_dataloader, val_dataloader, e, accuracy_list, num_batch)

  val_losses[e] = torch.tensor(val_loss)
  val_accuracies[e] = torch.tensor(val_accuracy)
  
  if val_loss < min_val_loss:
    min_val_loss = val_loss
    early_stopping_counter = EARLY_STOPPIMG_PATIENCE
    torch.save(model.state_dict(), './best_model.pth')
    print("- saved best model: val_loss =", val_loss, "val_accuracy =", val_accuracy)
  
  torch.save(model.state_dict(), './epoch_'+str(e)+'.pth')

  if e > 0:
    if val_losses[e] > val_losses[e-1]:
        early_stopping_counter -= 1
    else:
        early_stopping_counter = EARLY_STOPPIMG_PATIENCE

  if early_stopping_counter == 0:
      break