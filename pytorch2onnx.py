import torch

from unet import UNet

def main():
    net = UNet(n_channels=3, n_classes=1)
    net.load_state_dict(torch.load('MODEL.pth'))
    net.eval()

    input_var = torch.rand(1, 3, 640, 959)  # Use half of the original resolution.
    torch.onnx.export(net, input_var, 'Unet.onnx', verbose=True, export_params=True)


if __name__ == '__main__':
    main()
