model_configs = {
	"name": 'Unet', # can be Unet, UnetPlus, Deeplabv3
	"learning_rate": 0.001,
    "regularization": 0.01,
    "in_channels": 1,
    "out_channels": 1
}

training_configs = {
    "img_resize_shape": 128,
    "patience": 5,
	"batch_size": 32,
    "max_epochs": 2,
    "train_ratio": 0.7,
    "val_ratio": 0.2,
    "save_dir": './saved_models/'
}