import torch


def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # 初始化卷积层权重
    elif class_name.find('BatchNorm') != -1 and class_name.find(
            'WithFixedBatchNorm') == -1:  # batch norm层不能用kaiming_normal初始化
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif class_name.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif class_name.find("LSTM") != -1 or class_name.find("LSTMCell") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


def weights_init_without_kaiming(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        torch.nn.init.xavier_normal_(m.weight)
    elif class_name.find('BatchNorm') != -1 and class_name.find(
            'WithFixedBatchNorm') == -1:  # batch norm层不能用kaiming_normal初始化
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif class_name.find("Linear") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif class_name.find("LSTM") != -1 or class_name.find("LSTMCell") != -1:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)


