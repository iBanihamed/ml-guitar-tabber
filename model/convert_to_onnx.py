import torch


from model import Net

onnx_path = "/mnt/disks/disk1/model_ta.onnx"
model_path = "/mnt/disks/disk1/model_ta.pt"
num_out_classes = 5


def main():
    # get available device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    pytorch_model = Net(num_out_classes)
    pytorch_model.load_state_dict(torch.load(model_path, map_location=device))
    if torch.cuda.is_available():
        pytorch_model.cuda()
    # put in eval mode
    pytorch_model.eval()
    # define the input size
    input_size = (3, 224, 224)
    # generate dummy data
    dummy_input = torch.rand(1, *input_size).type(torch.FloatTensor).to(device=device)
    # generate onnx file
    torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=True)


if __name__ == "__main__":
    main()
