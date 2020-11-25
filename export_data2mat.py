import os
from torch.utils.data import DataLoader
import scipy.io as scio
import numpy as np

from model import fpn
from utils.mat2pic import ValDataset_diff, trans_separate
import torch
project_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'fpn3')


def test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = fpn().to(device)

    model_path = "/root/zhaoxiaoyu/fpn_new/model/target/problem2/b/train50/4/best_model"

    target_dataset_test = ValDataset_diff(trans_separate, resize_shape=(200, 200))
    target_test_loader = DataLoader(target_dataset_test, batch_size=1, shuffle=False, drop_last=False)
    print("model path:", model_path)

    if os.path.exists(model_path):
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_path, map_location='cuda:0'))
        else:
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("model initiated with", model_path)

    mae, mse, num = 0, 0, 0
    model.eval()
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    for it, images in enumerate(target_test_loader):
        layout_image = images[0].to(device)
        heat_image = images[1].to(device)

        with torch.no_grad():
            m = model(layout_image)
            scio.savemat("data/export_data/" + str(it) + ".mat", {'u_diff_pre' : np.squeeze(m[0, 0, :, :].cpu().numpy())})
            l1_loss = criterion1(m, heat_image) * layout_image.shape[0]
            l2_loss = criterion2(m, heat_image) * layout_image.shape[1]
        num += layout_image.shape[0]
        mae += l1_loss.item()
        mse += l2_loss.item()
    mae = mae / num
    mse = mse / num
    return mae, mse


if __name__ == "__main__":
    print("\n#################### Test ####################\n")
    mae_test, mse_test = 0.0, 0.0
    mae_test_idx, mse_test_idx = test()
    print("mae:", mae_test_idx, "mse:", mse_test_idx)