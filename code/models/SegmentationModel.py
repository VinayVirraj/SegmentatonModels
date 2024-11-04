import os
import torch
import torch.nn as nn
from tqdm import tqdm
from networks.UnetNetwork import UNet
from networks.UnetPpNetwork import NestedUNet
from networks.DeeplabV3Network import DeeplabV3
from utils.losses import DiceLoss, jaccard_similarity
import matplotlib.pyplot as plt

class SegmentationModel(nn.Module):

    def __init__(self, configs):
        super(SegmentationModel, self).__init__()
        self.model_configs = configs
        if self.model_configs["name"] == "UnetPlus":
            self.network = NestedUNet(configs)
        elif self.model_configs["name"] == "Deeplabv3":
            self.network = DeeplabV3(configs)
        else:
            self.network = UNet(configs)
        self.criterion = DiceLoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=configs['learning_rate'], weight_decay=configs['regularization'])

    def train(self, train_loader, val_loader, configs, device):
        
        self.network.train()

        patience = configs["patience"]  # Default patience is 5 epochs
        best_val_loss = float("inf")
        epochs_no_improve = 0
        early_stop = False

        train_losses = []
        val_losses = []
        for epoch in range(configs["max_epochs"]):

            if early_stop:
                print(f"Early stopping triggered. Stopping training at epoch {epoch + 1}.")
                break

            train_running_loss = 0

            for images, masks in tqdm(train_loader):
                
                images = images.unsqueeze(0).permute(1,0,2,3).to(device)
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
                    val_images = val_images.unsqueeze(0).permute(1,0,2,3).to(device)
                    val_masks = val_masks.to(device)

                    val_output = self.network(val_images)
                    val_loss += self.criterion(val_output, val_masks).item()

            val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch + 1}/{configs['max_epochs']}],  Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}")
            train_losses.append(epoch_loss)
            val_losses.append(val_loss)

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

        if len(train_losses) > 1:
            plt.figure(figsize=(10, 5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Validation Loss")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Train and Validation Loss Over Epochs")
            plt.legend()
            plt.savefig("../plots/loss.png", format='png', dpi=300)  

    def evaluate(self, test_loader, configs, device):

        self.network.eval()
        test_jaccard_scores = []
        with torch.no_grad():
            checkpointfile = os.path.join(configs["save_dir"], f'{self.model_configs["name"]}-best.ckpt')
            ckpt = torch.load(checkpointfile, map_location="cpu")
            self.network.load_state_dict(ckpt, strict=True)
            print(f"Restored model parameters from {checkpointfile}")
            for test_images, test_masks in test_loader:
                test_images = test_images.unsqueeze(0).permute(1,0,2,3).to(device)

                test_pred = self.network(test_images)
                test_pred = (test_pred > 0.5).float()

                test_masks = test_masks.unsqueeze(1).to(device)

                for i in range(test_pred.size(0)):
                    jaccard = jaccard_similarity(test_pred[i], test_masks[i])
                    test_jaccard_scores.append(jaccard)

        mean_jaccard = sum(test_jaccard_scores) / len(test_jaccard_scores)
        print("Jaccard similarity score for test set: ", mean_jaccard.item())