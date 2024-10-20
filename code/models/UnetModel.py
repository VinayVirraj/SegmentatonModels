import os
import torch
import torch.nn as nn
from tqdm import tqdm
from networks.UnetNetwork import UNet
from utils.losses import DiceLoss

class UnetModel(nn.Module):

    def __init__(self, configs):
        super(UnetModel, self).__init__()
        self.model_configs = configs
        self.network = UNet(configs)
        self.criterion = DiceLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs['learning_rate'])

    def train(self, train_loader, val_loader, configs, device):
        
        self.network.train()

        patience = configs["patience"]  # Default patience is 5 epochs
        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False

        for epoch in range(configs["max_epochs"]):

            if early_stop:
                print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
                break

            train_running_loss = 0

            for images, masks in tqdm(train_loader):
                images = images.to(device)
                masks = masks.to(device)

                self.optimizer.zero_grad()
                output = self.network(images)

                loss = self.criterion(output, masks)

                loss.backward()
                self.optimizer.step()

                train_running_loss += loss.item()

            epoch_loss = train_running_loss / len(train_loader)

            self.network.eval()
            val_loss = 0
            with torch.no_grad():
                for val_images, val_masks in val_loader:
                    val_images = val_images.to(device)
                    val_masks = val_masks.to(device)

                    val_output = self.network(val_images)
                    val_loss += self.criterion(val_output, val_masks).item()

            val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{configs['max_epochs']}],  Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")


            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                checkpoint_path = os.path.join(configs["save_dir"], f'{self.model_configs["name"]}-best.ckpt')
                os.makedirs(configs["save_dir"], exist_ok=True)
                torch.save(self.network.state_dict(), checkpoint_path)
                print("Validation loss improved. Model has been saved.")
            else:
                epochs_no_improve += 1
                print(f"No improvement in validation loss for {epochs_no_improve} epochs.")

            if epochs_no_improve >= patience:
                early_stop = True
                print("Early stopping triggered.")

    def evaluate(self, test_loader, configs, device):

        self.network.eval()
        test_predictions = []
        test_ground_truths = []
        with torch.no_grad():
            checkpointfile = os.path.join(configs["save_dir"], f'{self.model_configs["name"]}-best.ckpt')
            ckpt = torch.load(checkpointfile, map_location="cpu")
            self.network.load_state_dict(ckpt, strict=True)
            print(f"Restored model parameters from {checkpointfile}")
            for test_images, test_masks in test_loader:
                test_images = test_images.to(device)

                test_pred = self.network(test_images)

                test_predictions.append(test_pred.cpu().numpy())
                test_ground_truths.append(test_masks.cpu().numpy())