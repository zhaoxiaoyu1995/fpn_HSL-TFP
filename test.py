import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from model import fpn
from utils.mat2pic import TestDataset_basic, trans_separate

project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fpn_new')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    model = fpn().to(device)

    model_path = os.path.join(project_path, 'model', 'source', 'problem1', 'b', 'scale10', '0', 'best_model')

    target_dataset_test = TestDataset_basic(trans_separate, resize_shape=(200, 200))
    target_valid_loader = DataLoader(target_dataset_test, batch_size=8, shuffle=False, drop_last=False)
    print("model path:", model_path)

    if os.path.exists(model_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("model initiated with", model_path)

    mae, mse, num = 0, 0, 0
    model.eval()
    criterion1 = nn.L1Loss()
    criterion2 = nn.MSELoss()
    for it, images in enumerate(target_valid_loader):
        layout_image = images[0].to(device)
        heat_image = images[1].to(device)

        with torch.no_grad():
            m = model(layout_image)
            l1_loss = criterion1(m, heat_image) * layout_image.shape[0]
            l2_loss = criterion2(m, heat_image) * layout_image.shape[1]
        num += layout_image.shape[0]
        mae += l1_loss.item()
        mse += l2_loss.item()
    mae = mae / num
    mse = mse / num

    print("mae:", mae*100)
    print("mse:", mse*10000)