import torch
import torch.nn as nn
from torch.utils import data
from mds189 import Mds189
import numpy as np
from skimage import io, transform
#import ipdb
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
from PIL import Image
import time
start = time.time()

# Helper functions for loading images.
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)

def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

# flag for whether you're training or not
is_train = True
is_key_frame = True # TODO: set this to false to train on the video frames, instead of the key frames
if is_key_frame:
  model_to_load = 'key_frame_model.ckpt' # This is the model to load during testing, if you want to eval a previously-trained model.
else:
  model_to_load = 'random_frame_model.ckpt'

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

# Parameters for data loader
params = {'batch_size': 32, 
          'shuffle': True,
          'num_workers': 2 
          }
params_test = {'batch_size': 32,  
          'shuffle': False,
          'num_workers': 2 
          }
# TODO: Hyper-parameters
num_epochs = 40
learning_rate = 0.001
# NOTE: depending on your optimizer, you may want to tune other hyperparameters as well

# Datasets
# TODO: put the path to your train, test, validation txt files
if is_key_frame:
    label_file_train =  './dataloader_files/keyframe_data_train.txt'
    label_file_val  =  './dataloader_files/keyframe_data_val.txt'
    # NOTE: the kaggle competition test data is only for the video frames, not the key frames
    # this is why we don't have an equivalent label_file_test with keyframes
else:
    label_file_train = './dataloader_files/videoframe_data_train.txt'
    label_file_val = './dataloader_files/videoframe_data_val.txt'
    label_file_test = './dataloader_files/videoframe_data_test.txt'

mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

# Generators
train_dataset = Mds189(label_file_train,loader=default_loader,transform=transforms.Compose([
                                               transforms.Pad(5), 
                                               transforms.Resize(100),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
train_loader = data.DataLoader(train_dataset, **params)

val_dataset = Mds189(label_file_val,loader=default_loader,transform=transforms.Compose([
                                               transforms.Pad(5),
                                               transforms.Resize(100),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean, std)
                                           ]))
val_loader = data.DataLoader(val_dataset, **params)

if not is_key_frame:
    test_dataset = Mds189(label_file_test,loader=default_loader,transform=transforms.Compose([
                                                   transforms.Pad(5),
                                                   transforms.Resize(100),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean, std)
                                               ]))
    test_loader = data.DataLoader(test_dataset, **params_test)

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.conv4 = nn.Conv2d(128, 256, 3)
        self.fc1 = nn.Linear(768, 450)
        self.fc2 = nn.Linear(450, 300)
        self.fc3 = nn.Linear(300, 120)
        self.fc4 = nn.Linear(120, 40)
        self.fc5 = nn.Linear(40, 8)
        self.dropout1 = nn.Dropout()
        self.dropout2 = nn.Dropout()
        self.dropout3 = nn.Dropout()

    def forward(self, x, test=False):
        x = F.max_pool2d(F.relu(self.conv1(x)), (4, 4))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv4(x)), (2, 2))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        if not test:
          x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        if not test:
          x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        if not test:
          x = self.dropout3(x)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

model = NeuralNet().to(device)

# if we're only testing, we don't want to train for any epochs, and we want to load a model
if not is_train:
    num_epochs = 0
    model.load_state_dict(torch.load('random_frame_model.ckpt'))

# Loss and optimizer
criterion = nn.CrossEntropyLoss(reduce=True, reduction='mean') 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.002)

# Train the model
# Loop over epochs
print('Beginning training..')
total_step = len(train_loader)
train_loss= []
val_loss = []
for epoch in range(num_epochs):
    # Training
    print('epoch {}'.format(epoch))
    for i, (local_batch,local_labels) in enumerate(train_loader):
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)
        
        # Forward pass
        outputs = model.forward(local_ims)
        loss = criterion(outputs, local_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 10 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))
    """
    with torch.no_grad():
        # Get training error
        t_loss = 0
        t_count = 0
        for (local_batch,local_labels) in train_loader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model.forward(local_ims, test=True)
            loss = criterion(outputs, local_labels)
            t_loss += loss.item()
            t_count += 1
        train_loss.append(t_loss/t_count)
        # Get validation error
        v_loss = 0
        v_count = 0
        for (local_batch,local_labels) in val_loader:
            # Transfer to GPU
            local_ims, local_labels = local_batch.to(device), local_labels.to(device)
            outputs = model.forward(local_ims, test=True)
            loss = criterion(outputs, local_labels)
            v_loss += loss.item()
            v_count += 1
        val_loss.append(v_loss/v_count)
    """     

end = time.time()
print('Time: {}'.format(end - start))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
# Training Error

print('Beginning Training Testing..')
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    groundtruth_list = []
    for (local_batch,local_labels) in train_loader:
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = model.forward(local_ims, test=True)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        predicted_list.extend(predicted)
        groundtruth_list.extend(local_labels)
        correct += (predicted == local_labels).sum().item()

    print('Training Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Validation Error
print('Beginning Validation Testing..')
with torch.no_grad():
    correct = 0
    total = 0
    predicted_list = []
    groundtruth_list = []
    for (local_batch,local_labels) in val_loader:
        # Transfer to GPU
        local_ims, local_labels = local_batch.to(device), local_labels.to(device)

        outputs = model.forward(local_ims, test=True)
        _, predicted = torch.max(outputs.data, 1)
        total += local_labels.size(0)
        predicted_list.extend(predicted)
        groundtruth_list.extend(local_labels)
        correct += (predicted == local_labels).sum().item()

    print('Validation Accuracy of the network on the {} test images: {} %'.format(total, 100 * correct / total))

# Look at some things about the model results..
# convert the predicted_list and groundtruth_list Tensors to lists
pl = [p.cpu().numpy().tolist() for p in predicted_list]
gt = [p.cpu().numpy().tolist() for p in groundtruth_list]

# view the per-movement accuracy
label_map = ['reach','squat','inline','lunge','hamstrings','stretch','deadbug','pushup']
for id in range(len(label_map)):
    print('{}: {}'.format(label_map[id],sum([p and g for (p,g) in zip(np.array(pl)==np.array(gt),np.array(gt)==id)])/(sum(np.array(gt)==id)+0.)))
    
# Save the model checkpoint
if is_key_frame:
  print("key_frame_model saved")
  torch.save(model.state_dict(), 'key_frame_model.ckpt')
else:
  print("random_frame_model saved")
  torch.save(model.state_dict(), 'random_frame_model.ckpt')

if not is_key_frame and not is_train:
    print("making predictions for kaggle...")
    with torch.no_grad():
      correct = 0
      total = 0
      predicted_list = []
      groundtruth_list = []
      for (local_batch,local_labels) in test_loader:
          # Transfer to GPU
          local_ims, local_labels = local_batch.to(device), local_labels.to(device)
          outputs = model.forward(local_ims, test=True)
          _, predicted = torch.max(outputs.data, 1)
          predicted_list.extend(predicted)
      print("predicted_list:", predicted_list)

      def results_to_csv(y_test):
          y_test = y_test.astype(int)
          df = pd.DataFrame({'Category': y_test})
          df.index += 1  
          df.to_csv('submission.csv', index_label='Id')
      results_to_csv(np.array(predicted_list))
