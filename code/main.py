import argparse
import torch
from torchvision import transforms
from utils.dataLoader import load_data
from models.UnetModel import SegmentationModel
from configure import model_configs, training_configs


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", help="path to the data")

args = parser.parse_args()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training running on {device}")

    # transform = transforms.Compose([
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    model = SegmentationModel(model_configs).to(device)

    train_data, val_data, test_data = load_data(
        args.data_dir,
        None,
        batch_size=training_configs['batch_size'],
        train_ratio=training_configs['train_ratio'],
        val_ratio=training_configs['val_ratio']
    )

    model.train(train_data, val_data, training_configs, device)
    model.evaluate(test_data, training_configs, device)