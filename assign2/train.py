import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
import argparse
from datetime import datetime
from timeit import default_timer as timer


def arg_parsing():
    parser = argparse.ArgumentParser(
        description='Model Training Setting')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='directory of dataset')
    parser.add_argument('--arch', type=str, required=True,
                        help='architecture of model used')
    parser.add_argument('--model_save', type=str, default=None,
                        help='filename to save of the training model')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='number of examples/minibatch')
    parser.add_argument('--epoch', type=int, default=10,
                        help='number of epochs')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='pretrained-model path')

    args_pool = parser.parse_args()
    return args_pool


def show_a_batch(loader):
    def imshow(inp, title=None):
        """Imshow for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        # mean = np.array([0.485, 0.456, 0.406])
        # std = np.array([0.229, 0.224, 0.225])
        # inp = std * inp + mean
        # inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    # Get a batch of training data
    inputs, classes = next(iter(loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    imshow(out)


def show_info(datadir):
    img_preprocess = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5)
    ])
    img_dataset = datasets.ImageFolder(datadir + '/train')

    img, _ = img_dataset[0]
    img = transforms.ToTensor()(img)
    print('-' * 20 + ' Original Image ' + '-' * 20)
    print('Image Size: ', '{}x{}'.format(img.shape[1], img.shape[2]))
    print('Image min: ', '{:.4f}'.format(torch.min(img).item()))
    print('Image max: ', '{:.4f}'.format(torch.max(img).item()))
    print('Image mean: ', '{:.4f}'.format(torch.mean(img).item()))
    print()
    img = transforms.ToPILImage()(img)
    img = img_preprocess(img)
    print('-' * 20 + ' Pre-processed Image ' + '-' * 20)
    print('Image Size: ', '{}x{}'.format(img.shape[1], img.shape[2]))
    print('Image min: ', '{:.4f}'.format(torch.min(img).item()))
    print('Image max: ', '{:.4f}'.format(torch.max(img).item()))
    print('Image mean: ', '{:.4f}'.format(torch.mean(img).item()))
    print()


def load_datasets(datadir, batch_size=64, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    data_transforms = {
        'train':
            transforms.Compose([
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        'val':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
        'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]),
    }
    _image_datasets = {x: datasets.ImageFolder(datadir + '/' + x, transform=data_transforms[x]) for x in
                       ['train', 'val', 'test']}

    _data_loaders = {
        x: torch.utils.data.DataLoader(_image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=32) for x in
        ['train', 'val', 'test']}
    _dataset_sizes = {x: len(_image_datasets[x]) for x in ['train', 'val', 'test']}
    _class_names = _image_datasets['train'].classes

    return _data_loaders, _dataset_sizes, _class_names


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

        _model.fc = nn.Sequential(nn.Linear(2048, 512),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(512, classes_add),
                                  nn.Sigmoid())
    return _model


def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    def accurate(output, target):
        """Computes the accuracy for multiple binary predictions"""
        pred = output >= 0.5
        truth = target >= 0.5
        acc = pred.eq(truth).sum()
        return acc

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            start = timer()
            for batch_id, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.cuda()
                labels = labels.cuda().type(torch.float32)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze()
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += accurate(outputs, labels.data)
                print(
                    f'Epoch: {epoch}\t{100 * (batch_id + 1) / len(dataloaders[phase]):.2f}% complete.',
                    f'\t {timer() - start:.2f} seconds elapsed in epoch.',
                    end='\r')
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'\n{phase} | Epoch: {epoch} \tLoss: {epoch_loss:.4f}\t\t Accuracy: {100 * epoch_acc:.2f}%')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def evaluate_testset(_model, testset_loader, testset_sizes, phase='val'):
    def to_binary(out_prob):
        pred = out_prob >= 0.5
        out_bin = out_prob.clone()
        out_bin[pred] = 1
        out_bin[~pred] = 0
        return out_bin

    _model = _model.eval()
    pred_Positive = pred_Negative = label_Positive = label_Negative = pred_prob = torch.tensor([]).cuda()

    start = timer()
    for batch_id, (inputs, labels) in enumerate(testset_loader[phase]):
        inputs = inputs.cuda()
        labels = labels.cuda().type(torch.float32)

        outputs = _model(inputs).squeeze()
        probs = outputs
        preds = to_binary(outputs)

        # statistics
        pred_prob = torch.cat((pred_prob, probs))
        pred_Positive = torch.cat((pred_Positive, preds == torch.ones_like(preds).bool()))
        pred_Negative = torch.cat((pred_Negative, preds == torch.zeros_like(preds).bool()))
        label_Positive = torch.cat((label_Positive, labels.data == torch.ones_like(labels.data).bool()))
        label_Negative = torch.cat((label_Negative, labels.data == torch.zeros_like(labels.data).bool()))

        print(
            f'{phase} | {100 * (batch_id + 1) / len(testset_loader[phase]):.2f}% complete.',
            f'\t {timer() - start:.2f} seconds elapsed in testing.',
            end='\r')
    print()

    True_Positive = torch.sum(torch.logical_and(pred_Positive, label_Positive))
    True_Negative = torch.sum(torch.logical_and(pred_Negative, label_Negative))
    False_Positive = torch.sum(torch.logical_and(pred_Positive, label_Negative))
    False_Negative = torch.sum(torch.logical_and(pred_Negative, label_Positive))
    print()
    assert (True_Positive + True_Negative + False_Positive + False_Negative == testset_sizes[phase])
    accuracy = (True_Positive + True_Negative).double() / (
                True_Positive + True_Negative + False_Positive + False_Negative)
    sensitivity = True_Positive.double() / torch.sum(label_Positive)
    precision = True_Positive.double() / (True_Positive + False_Positive)
    print(phase + ' | Accuracy: {:.2f}%'.format(100 * accuracy))
    print(phase + ' | Sensitivity (Recall): {:.2f}%'.format(100 * sensitivity))
    print(phase + ' | Precision: {:.2f}%'.format(100 * precision))


    # Plot ROC curve
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    fpr, tpr, threshold = roc_curve(label_Positive.detach().cpu().numpy(), pred_prob.detach().cpu().numpy())
    # print(fpr, tpr, threshold)

    auc1 = auc(fpr, tpr)
    # Plot the result
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, color='orange', label='AUC = %0.2f' % auc1)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == '__main__':
    args = arg_parsing()
    show_info(args.dataset_dir)
    data_loaders, dataset_sizes, class_names = load_datasets(args.dataset_dir, batch_size=args.batch_size)

    # show_a_batch(data_loaders['train'])

    # Please specify the network to used for training.
    model = redesign_model(arch=args.arch)
    if model is None:
        raise ValueError(f'There is no required architecture {args.arch}.')

    model = model.cuda()
    criterion = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer = torch.optim.Adam(model.parameters())

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    torch.cuda.empty_cache()

    if args.checkpoint is None:
        model = train_model(model, data_loaders, criterion, optimizer, exp_lr_scheduler, num_epochs=args.epoch)
        torch.save(model, args.model_save if args.model_save is not None else
        args.arch + datetime.now().strftime("_%m_%d_%H_%M") + '.pt')
    else:
        try:
            cp = torch.load(args.checkpoint)
            model.load_state_dict(cp)
        except AttributeError:
            model = cp

    evaluate_testset(model, data_loaders, dataset_sizes, phase='train')
    evaluate_testset(model, data_loaders, dataset_sizes, phase='val')
    evaluate_testset(model, data_loaders, dataset_sizes, phase='test')
