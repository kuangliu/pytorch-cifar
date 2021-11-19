import torch


def get_prune_params(net):
    parameters_to_prune = (
        (net.module.conv1, 'weight'),
        (net.module.bn1, 'weight'),

        (net.module.layer1[0].conv1, 'weight'),
        (net.module.layer1[0].bn1, 'weight'),
        (net.module.layer1[0].conv2, 'weight'),
        (net.module.layer1[0].bn2, 'weight'),
        (net.module.layer1[1].conv1, 'weight'),
        (net.module.layer1[1].bn1, 'weight'),
        (net.module.layer1[1].conv2, 'weight'),
        (net.module.layer1[1].bn2, 'weight'),

        (net.module.layer2[0].conv1, 'weight'),
        (net.module.layer2[0].bn1, 'weight'),
        (net.module.layer2[0].conv2, 'weight'),
        (net.module.layer2[0].bn2, 'weight'),
        (net.module.layer2[0].shortcut[0], 'weight'),
        (net.module.layer2[0].shortcut[1], 'weight'),
        (net.module.layer2[1].conv1, 'weight'),
        (net.module.layer2[1].bn1, 'weight'),
        (net.module.layer2[1].conv2, 'weight'),
        (net.module.layer2[1].bn2, 'weight'),

        (net.module.layer3[0].conv1, 'weight'),
        (net.module.layer3[0].bn1, 'weight'),
        (net.module.layer3[0].conv2, 'weight'),
        (net.module.layer3[0].bn2, 'weight'),
        (net.module.layer3[0].shortcut[0], 'weight'),
        (net.module.layer3[0].shortcut[1], 'weight'),
        (net.module.layer3[1].conv1, 'weight'),
        (net.module.layer3[1].bn1, 'weight'),
        (net.module.layer3[1].conv2, 'weight'),
        (net.module.layer3[1].bn2, 'weight'),

        (net.module.layer4[0].conv1, 'weight'),
        (net.module.layer4[0].bn1, 'weight'),
        (net.module.layer4[0].conv2, 'weight'),
        (net.module.layer4[0].bn2, 'weight'),
        (net.module.layer4[0].shortcut[0], 'weight'),
        (net.module.layer4[0].shortcut[1], 'weight'),
        (net.module.layer4[1].conv1, 'weight'),
        (net.module.layer4[1].bn1, 'weight'),
        (net.module.layer4[1].conv2, 'weight'),
        (net.module.layer4[1].bn2, 'weight'),

    )
    return parameters_to_prune


def print_sparsity(model):
    params = get_prune_params(model)
    zero_weights = 0
    total_weigts = 0
    for param in params:
        zero_weights += torch.sum(param[0].weight == 0)
        total_weigts += param[0].weight.nelement()
    print("Global sparsity: {:.2f}%".format(100. * zero_weights / total_weigts))
