import torch
import os
import glob
from tqdm import tqdm
from utils import create_validation_arg_parser, generate_dataset
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, recall_score, f1_score, classification_report
import pandas as pd
from scipy.special import softmax
import torch.nn as nn
import torchvision.models as models
from resnest.torch import resnest50


net_config = {
    'vgg16': models.vgg16_bn,
    'resnet50': models.resnet50,
    'resnext50': models.resnext50_32x4d,
    # 'resnest50': torch.hub.load('zhanghang1989/ResNeSt', 'resnest50', pretrained=True),
    'resnest50': resnest50(pretrained=True),

    'args': {}
}


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class CustomizedModel(nn.Module):
    def __init__(self, name, backbone, num_classes, pretrained=False, **kwargs):
        super(CustomizedModel, self).__init__()

        if 'resnest' in name:
            net = resnest50(pretrained=True)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        else:
            net = backbone(pretrained=pretrained, **kwargs)
        if 'resnet' in name or 'resnext' in name or 'shufflenet' in name:
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3,
                                  bias=False)
            net.fc = nn.Linear(net.fc.in_features, num_classes)
        elif 'densenet' in name:
            net.classifier = nn.Linear(net.classifier.in_features, num_classes)
        elif 'vgg' in name:
            net.features = make_layers(cfgs['D'], batch_norm=True)
            net.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes),
            )
        elif 'mobilenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(net.last_channel, num_classes),
            )
        elif 'squeezenet' in name:
            net.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif 'resnest' in name:
            pass
        else:
            raise NotImplementedError('Not implemented network.')
        self.net = net

    def forward(self, x):
        x = self.net(x)
        return x


def generate_model(network, out_features, net_config, device, pretrained=False, checkpoint=None):
    if pretrained:
        print('Loading weight from pretrained')
    if checkpoint:
        model = torch.load(checkpoint).to(device)
        print('Load weights form {}'.format(checkpoint))
    else:
        if network not in net_config.keys():
            raise NotImplementedError('Not implemented network.')

        model = CustomizedModel(
            network,
            net_config[network],
            out_features,
            pretrained,
            **net_config['args']
        ).to(device)

    if device == 'cuda' and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    return model


def evaluate(model, valLoader, device, save_path):

    model.eval()

    name = []
    _label = []
    pred = []
    prob = []

    for i, (img_file_name, inputs, targets1, targets2, targets3, targets4) in enumerate(tqdm(valLoader)):

        inputs = inputs.to(device)
        targets1, targets2 = targets1.to(device), targets2.to(device)
        targets3, targets4 = targets3.to(device), targets4.to(device)
        targets = [targets1, targets2, targets3, targets4]

        outputs = model(inputs)

        labels = torch.argmax(targets[3], dim=2).squeeze(1)
        preds = torch.argmax(outputs, dim=1)
        clf_preds = preds.detach()

        predictions = preds.detach().cpu().numpy()
        target = labels.detach().cpu().numpy()

        name.append(os.path.basename(img_file_name[0]))
        prob.append(softmax(clf_preds.detach().cpu().numpy().squeeze()))
        _label.append(target.item())
        pred.append(predictions.item())

    return name, prob, _label, pred


def main():
    args = create_validation_arg_parser().parse_args()

    names = []
    labels = []
    preds = []
    probs = []

    for i in range(1, 6):

        print(i)

        _train_path = args.train_path.replace('XXX', str(i))
        _val_path = args.val_path.replace('XXX', str(i))
        _model_file = args.model_file.replace('XXX', str(i))

        print(_train_path)
        print(_val_path)
        print(_model_file)

        print('Original:', _model_file)
        file_name = ''
        acc_max = 0
        for file in os.listdir(_model_file):
            if os.path.isfile(os.path.join(_model_file, file)):
                indexs = file.split('_')
                if float(indexs[2]) > acc_max:
                    acc_max = float(indexs[2])
                    file_name = file

        print('File name:', file_name)
        model_file = os.path.join(_model_file, file_name)
        print('New path:', model_file)
        save_path = args.save_path

        if args.pretrain == 'True':
            pretrain = True
        else:
            pretrain = False

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        args.batch_size = 16
        cuda_no = args.cuda_no
        CUDA_SELECT = "cuda:{}".format(cuda_no)
        device = torch.device(CUDA_SELECT if torch.cuda.is_available() else "cpu")
        train_file_names = glob.glob(os.path.join(_train_path, "*.png"))
        val_file_names = glob.glob(os.path.join(_val_path, "*.png"))

        _, valid_loader = generate_dataset(train_file_names, val_file_names, args.batch_size, 1, args.distance_type, False)

        model = generate_model(args.model_type, args.classnum, net_config, device, pretrained=pretrain, checkpoint=None).to(device)
        model.load_state_dict(torch.load(model_file))

        _name, _prob, _label, _pred = evaluate(model, valid_loader, device, save_path)

        names.extend(_name)
        probs.extend(_prob)
        labels.extend(_label)
        preds.extend(_pred)

    acc = accuracy_score(labels, preds)
    kappa = cohen_kappa_score(labels, preds)
    recall = recall_score(labels, preds, average='micro')
    f1 = f1_score(labels, preds, average='weighted')
    c_matrix = confusion_matrix(labels, preds)

    if args.classnum == 3:
        target_names = ['N', 'D', 'M']
        clas_report = classification_report(labels, preds, target_names=target_names)
    elif args.classnum == 2:
        target_names = ['N', 'D']
        clas_report = classification_report(labels, preds, target_names=target_names)

    dataframe = pd.DataFrame({'case': names, 'prob': probs, 'label': labels, 'pred': preds})
    dataframe.to_csv(os.path.join(save_path, "new_class.csv"), index=False, sep=',')
    print('Counting CSV generated!')
    with open(os.path.join(save_path, "new_cmatrix.txt"), "w") as f:
        f.write(str(c_matrix))
    with open(os.path.join(save_path, "new_clas_report.txt"), "w") as f:
        f.write(str(clas_report))
    with open(os.path.join(save_path, "new_clf_report.txt"), "w") as f:
        f.write('ACC : ' + str(acc) + ', Kappa: ' + str(kappa) + '\r\n')
        f.write('F1 : ' + str(f1) + ', Recall: ' + str(recall) + '\r\n')


if __name__ == "__main__":
    main()
