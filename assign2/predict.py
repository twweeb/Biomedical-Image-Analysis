import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import argparse
from timeit import default_timer as timer
import csv


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def arg_parsing():
    parser = argparse.ArgumentParser(
        description='Model Training Setting')
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='architecture of model used')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='filename to save of the training model')
    parser.add_argument('--dir', type=str, required=True,
                        help='directory of images to predict')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of examples/minibatch')
    parser.add_argument('--output', type=str, required=True,
                        help='path to save the result (in .csv format)')

    args_pool = parser.parse_args()
    return args_pool


def redesign_model(arch='resnet18', classes_add=1):
    _model = None
    if arch == 'vgg16':  # Acc: 82.37%
        _model = models.vgg16(pretrained=True)

        for param in _model.parameters():
            param.requires_grad = False
        n_inputs = _model.classifier[6].in_features

        # Add on classifier
        _model.classifier[6] = nn.Sequential(
            nn.Linear(n_inputs, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, classes_add), nn.Sigmoid())

    elif arch == 'resnet18':  # Acc: 87.34%
        _model = models.resnet18(pretrained=True)
        for param in _model.parameters():
            param.requires_grad = False

        num_ftrs = _model.fc.in_features
        _model.fc = nn.Sequential(nn.Linear(num_ftrs, classes_add),
                                  nn.Sigmoid())

    elif arch == 'resnet50':  # Acc: 85.10%
        _model = models.resnet50(pretrained=True)
        for param in _model.parameters():
            param.requires_grad = False

        _model.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, classes_add),
                                  nn.Sigmoid())

    elif arch == 'resnet101':
        _model = models.resnet101(pretrained=True)
        for param in _model.parameters():
            param.requires_grad = False

        num_ftrs = _model.fc.in_features
        _model.fc = nn.Sequential(nn.Linear(num_ftrs, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, classes_add),
                                  nn.Sigmoid())
    return _model


def load_datasets(datadir, batch_size=1):
    data_transforms = {
        'predict':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
            ]),
    }

    image_datasets = {'predict': ImageFolderWithPaths(datadir, data_transforms['predict']),}
    dataloader = {'predict': torch.utils.data.DataLoader(image_datasets['predict'],
                                                         batch_size=batch_size,
                                                         shuffle=False,
                                                         num_workers=32),
                  }

    return dataloader


def predict(model, dataloader, filename):

    image_predicts = list()
    image_filenames = list()
    start = timer()
    print("0: NORMAL, 1: PNEUMONIA")
    for data_id, (inputs, labels, paths) in enumerate(dataloader):
        inputs = inputs.cuda()
        output = model(inputs).squeeze().item()
        index = 1 if output >= 0.5 else 0
        # index = output.data.cpu().numpy().argmax()
        image_predicts.append(index)
        image_filenames.append(paths[0].split('/')[-1])
        print(f'Predict | {100 * (data_id + 1) / len(dataloader):.2f}% complete.',
                    f'\t total: {len(image_predicts)} images.',
                    end='\r')
    print()
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["case", "predict"])
        for i in range(len(image_filenames)):
            writer.writerow([image_filenames[i], image_predicts[i]])

    print(f'Predict used {timer() - start:.2f} seconds.')


if __name__ == '__main__':
    args = arg_parsing()

    dataloader = load_datasets(datadir=args.dir)

    # show_a_batch(dataloaders['train'])

    # Please specify the network to used for training.
    model = redesign_model(arch=args.arch)
    if model is None:
        raise ValueError(f'There is no required architecture {args.arch}.')

    model = model.cuda()
    try:
        cp = torch.load(args.checkpoint)
        model.load_state_dict(cp)
    except:
        model = cp

    predict(model, dataloader['predict'], args.output+'/'+args.arch+'_result.csv')