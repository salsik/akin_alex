import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import os

from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from scipy.stats import entropy


import PIL

from datetime import datetime

import argparse


class Alex_Inception(object):
    def __init__(self, config=None):

        print(config)

        if config:
           self.data_dir = config.data_dir  
        else:
            self.data_dir= "data/final500"

        #self.data_dir = "data/Akin_SAGAN_500/semantic"

        self.train_loader, self.test_loader = self.prepare_data()

        self.model = torchvision.models.inception_v3(pretrained=True)

        # check this if it should be eval or not
        self.model.eval()

        # Freeze the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False

        # Replace the final layer with a custom fully connected layer for 5-class classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 5)

        # 5 Define the loss function and optimizer:

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.fc.parameters(), lr=0.01, momentum=0.9)
        #self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001, betas=(0.0, 0.9), eps=1e-08, weight_decay=0)
        #self.optimizer = optim.Adam(self.model.fc.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    def trans(self):
        # Load the dataset and apply transformations
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

    def prepare_data(self):

        # Load the dataset and apply transformations

        transform = self.trans()
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])

        dataset = torchvision.datasets.ImageFolder(
            root=self.data_dir, transform=transform)

        # Split the dataset into train and test sets
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])

        # Load the datasets into data loaders for batch processing
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False)

        return train_loader, test_loader

    def train(self):

        # 3 Load the Inception v3 model and set it to evaluation mode:

        experiment_name = "__1000 epochs SGD post proc special lr 0.01 0.9"
        #experiment_name = "__100 epochs adam post proc 0.9 0.999"

        # Train the model for a specified number of epochs
        num_epochs = 1000
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                inputs, labels = data
                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            print('Epoch %d loss: %.3f' %
                  (epoch + 1, running_loss / len(self.train_loader)))

        print('Finished training')

        timestamp = "/" + datetime.now().strftime("%Y%m%d-%H%M%S") + experiment_name

        model_path = 'checkpoint/inception/{}.pt'.format(timestamp)

        torch.save(self.model.state_dict(), model_path)
        print("model saved in path !", model_path)

        return model_path, self.model

    def test_eval(self):

        # Evaluate the model on the validation set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.train_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the validation images: %d %%' %
              (100 * correct / total))

        # Test the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' %
              (100 * correct / total))

    def load_eval(self, model_dir):

        self.model.load_state_dict(torch.load(model_dir))
        self.model.eval()
        # self.test_eval()
        return


# get id from folder name

def get_label(folder_name):
    if folder_name == 'login':
        return 0
    elif folder_name == 'account_creation':
        return 1
    elif folder_name == 'product_listing':
        return 2
    elif folder_name == 'product_description':
        return 3
    elif folder_name == 'splash':
        return 4


images = []
labels = []

# I dont think there is aneed to do this step... pytorch already do it in image fodler


def labeling(data_dir):

    # I dont think there is aneed to do this step... pytorch already do it in image fodler

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)
        if os.path.isdir(folder_path):
            label = get_label(folder_name)
            for image_filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_filename)
                image = load_image(image_path)
                images.append(image)
                labels.append(label)

    # Convert the lists to tensors
    images = torch.stack(images)
    labels = torch.tensor(labels)

    # Create a TensorDataset from the images and labels
    dataset = TensorDataset(images, labels)

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


# train and eval
"""

In this code, we define a custom class Inception_v3 that inherits from nn.Module, 
and in the __init__ method, we replace the final fully connected layer in both 
the AuxLogits and the main branch with fully connected layers with num_classes outputs
, which is 5 in our case.
"""


class Inception_v3(nn.Module):
    def __init__(self, num_classes):
        super(Inception_v3, self).__init__()
        self.inception = torchvision.models.inception_v3(pretrained=True)
        #num_ftrs = self.inception.AuxLogits.fc.in_features
        #self.inception.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)

        for param in self.inception.parameters():
            param.requires_grad = False

        # Replace the final layer with a custom fully connected layer for 5-class classification
        self.inception.fc = nn.Linear(self.inception.fc.in_features, 5)

    def forward(self, x):
        x = self.inception(x)
        return x


def mainold():

    model_dir = "checkpoint/inception/20230130-222834__100 epochs adam 0.0 0.9.pt"
    model = Inception_v3(num_classes=5)
    model.load_state_dict(torch.load(model_dir))
    model.eval()

    im_path = "data/Akin_SAGAN_500/semantic/account_creation/1080.jpg"

    image = PIL.Image.open(im_path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)

    output = model(image_tensor)
    _, prediction = torch.max(output, 1)

    print(output, prediction)

    label = class_labels[int(prediction)]
    print(label)


def alextrain_inception():


    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-w",
                        help="image width", default="data/final500")
    args = parser.parse_args()



    print(args)

    model = Alex_Inception(args)

    path, m = model.train()
    model.test_eval()

    print("model is trained and saved in the path", path)


def alexeval_inception():

    model = Alex_Inception()

    # load and eval

    model_dir = "checkpoint/inception//20230130-212705__3 epochs adam 0.9 0.999.pt"
    model_dir = "checkpoint/inception//20230130-215106__3 epochs adam 0.9 0.999.pt"
    model_dir = "checkpoint/inception//20230130-231051__100 epochs sgd 0.001 0.5.pt"
    model_dir = "checkpoint/inception/20230130-222834__100 epochs adam 0.0 0.9.pt"

    model_dir = "checkpoint/inception/20230201-190341__100 epochs adam post proc 0.0 0.9.pt"
    model.load_eval(model_dir)

    # model.test_eval()

    # Use the trained model to classify new images
    # ....

    class_labels = {0: 'account_creation', 1: 'login',
                    2: 'product_description', 3: 'product_listing', 4: 'splash'}

    # be carfull the image should be in the same type of the one the model has tained on
    #im_path = "data/Akin_SAGAN_500/semantic/account_creation/1080.jpg"

    im_path = "data/final500/splash/28_False.jpg"

    transform = model.trans()

    img = PIL.Image.open(im_path)
    img.show()
    img = transform(img).unsqueeze(0)

    # no need for torch no grad because we already put the model on eval mode

    with torch.no_grad():
        # here we put model.model instead of model(img) because the class is not inheriting frmo nn.torch
        # it:s a normal class that has a model atttribute which is pytorch model
        output = model.model(img).softmax(-1)
        # this dim 1 to gt the class id
        _, prediction = torch.max(output.data, 1)

    print(output, prediction)

    label = class_labels[int(prediction)]
    print(label)


"""

 imgs argument should be a tensor with the shape (N, 3, H, W), 
 Where N is the number of generated images, or images that we need to calculate score  


"""

# calculate the inception score for p(y|x)


def calculate_inception_score(p_yx, eps=1E-16):
    # calculate p(y)
    p_y = np.expand_dims(p_yx.mean(axis=0), 0)
    # kl divergence for each image
    kl_d = p_yx * (np.log(p_yx + eps) - np.log(p_y + eps))
    # sum over classes
    sum_kl_d = kl_d.sum(axis=1)
    # average over images
    avg_kl_d = np.mean(sum_kl_d)
    # undo the logs
    is_score = np.exp(avg_kl_d)
    return is_score


def inception_score(imgs, inception_model, resize=True, splits=1):
    N = len(imgs)

    print("images shape", imgs.shape)

    # Compute the prediction probabilities for each image
    preds = inception_model(imgs)
    print("preds shape:", preds.shape)

   # important procedure to normalize the output to get a conditional probability betweeen 0 and 1
    preds = torch.softmax(preds, dim=1)
    # print(preds)

    # for p in preds:
    # to make sure that it:s normalized
    print(torch.sum(preds[0]))

    print("preds shape:", preds.shape)
    # print(preds)

    # Compute the KL divergence between the prediction probabilities and a uniform distribution
    kl_divergence = - \
        torch.mean(torch.sum(preds * torch.log(preds + 1e-10), dim=0))
    #kl_divergence = -torch.mean(torch.sum(preds * torch.log(preds + 1e-10), dim=1), dim=0)

    # Compute the exponential of the KL divergence to get the Inception Score
    inception_score = torch.exp(kl_divergence)

    # return inception_score
    return calculate_inception_score(preds.cpu().detach().numpy())


def inception_score_another_way(imgs, my_inception_model, cuda=False, batch_size=20, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    #inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model = my_inception_model
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        #print("x shape:", x.shape)
        x = inception_model(x)
        return torch.softmax(x, dim=1).data.cpu().numpy()
        # return F.softmax(x).data.cpu().numpy()

    #preds = inception_model(imgs).softmax(-1).mean(-1)

    # Get predictions
    preds = np.zeros((N, 5))

    print("pred shape:", preds.shape)

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def compute_folder_score(folder_path, model_dir):

    import glob
    image_list = []

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    for filename in glob.glob(folder_path + '/*.jpg'):  # assuming jpg
        #print(filename)
        im = PIL.Image.open(filename)
        im = transform(im)  # .unsqueeze(0)
        image_list.append(im)

    #print(image_list)

    for dir in os.listdir(folder_path):
        #print(dir)
        folder_name = os.path.join(folder_path, dir)
        if os.path.isdir(folder_name):
            for filename in os.listdir(folder_name):
                filename = os.path.join(folder_name, filename)
                #print(filename)
                im = PIL.Image.open(filename)
                im = transform(im)  # .unsqueeze(0)
                image_list.append(im)

    print(len(image_list))

    # another way
    image_list2 = []
    for image_filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_filename)
        if os.path.isfile(image_path):
            image = PIL.Image.open(image_path).convert('RGB')
            image = transform(image)
            image_list2.append(image)

    image_list = torch.stack(image_list)  # .cuda()
    print(image_list.shape)

    inception_model = Alex_Inception()

    inception_model.load_eval(model_dir)

    # first method to calculate is
    iscore = inception_score(
        image_list, inception_model.model, resize=True, splits=1)
    iscore = iscore.item()
    print("inception score =", iscore)

    # second method
    mean_iscore, std_iscore = inception_score_another_way(
        image_list.cuda(), inception_model.model)
    print(mean_iscore, std_iscore)


"""

this function apply inception score model on a specific folder
and compute the score in two methods ..
which give similar values

"""


def apply_inception_score(args=None):

    model_dir = "checkpoint/inception/20230130-222834__100 epochs adam 0.0 0.9.pt"

    model_dir = "checkpoint/inception/20230201-190341__100 epochs adam post proc 0.0 0.9.pt"

    #folder_path = "data/Akin_SAGAN_500/semantic/login"
    #folder_path = "data/Akin_SAGAN_500/all_inone"

    #folder_path ="/home/atsumilab/alex/rico/akin-generator/samples/20230128-072217 lr 0.0004 0.0016 0.0 0.9 10"

    # achieved score 3.56
    #folder_path = "data/final500"

    #folder_path = "data/final_pretrained"

    folder_path = args.data_path

    compute_folder_score(folder_path, model_dir)

    # apply it for folder

    folder = "/data1/data_alex/rico/akin experiments/samples_generator/"

    for sub_folder in os.listdir(folder):

        folder_path = folder + sub_folder
        #print (folder_path)
        # compute_folder_score(folder_path,model_dir)




def loop_inception_models(args=None):

    models_folder ="checkpoint/inception"
    for model_dir in os.listdir(models_folder):

        # to make sure it's the lastest models that we trained  on 1000 or 100 epochs
        if(model_dir.startswith("2023020")):
            
            model_dir = os.path.join(models_folder,model_dir)
            print("model name:",model_dir)

            #model_dir = "checkpoint/inception/20230201-190341__100 epochs adam post proc 0.0 0.9.pt"

            folder_path = args.data_dir
            print("testing on :",folder_path)
            
            compute_folder_score(folder_path, model_dir)








"""
here we compute inception using two methods according to a trained inception model


# trainng inception model is also part of this code we do it by calling the function 

alextrain_inception()

to train the inception model on a specific folder with a specifi parametrs .


is to eavluate and produce in

alexeval_inception()
    


"""

if __name__ == "__main__":

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", "-w",
                        help="data path", default="/data1/data_alex/rico/akin_experiments/samples_generator/postproc/final14")
    args = parser.parse_args()



    # doesn't work properly
    # mainold()

    # trainnig the model from the begingn
    #alextrain_inception()

    # evaluation and testing
    # alexeval_inception()

    # apply innception on args folder 
    ####apply_inception_score(args)

    # llop inception modes on args fodler
    loop_inception_models(args)

    # should reconsider computing inception score for those values
