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
    print(
        "Global sparsity: {:.2f}%".format(
            100. * float(
                torch.sum(model.conv1.weight == 0)
                + torch.sum(model.conv2.weight == 0)
                + torch.sum(model.fc1.weight == 0)
                + torch.sum(model.fc2.weight == 0)
                + torch.sum(model.fc3.weight == 0)
            )
            / float(
                model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement()
            )
        )
    )