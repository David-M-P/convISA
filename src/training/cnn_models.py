import torch.nn as nn
import torch


class UnFlatten(nn.Module):
    """
    Class that gets the tensor into desired shape, used at the end of
    the encoding step. Its output is fed in an unflattened shape to
    the decoder and in in a flat shape to the predictor.
    """

    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class AE_fixed(nn.Module):
    """
    Main component of the CNN with fixed parameters, tensor shapes
    after each step can be seen along each step of the net.
    """

    def __init__(self, output_size, image_channels=1):
        super(AE_fixed, self).__init__()
        self.encoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, 16, kernel_size=(5, 7), stride=(2, 3), padding=(3, 3)
            ),  # Output (B, 16, 11, 99)
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.Conv2d(32, 64, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.Conv2d(64, 1024, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024, 1, 1])
            UnFlatten(-1, 1024, 1, 1),
        )
        self.decoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1024, 1, 1])
            nn.ConvTranspose2d(1024, 64, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.ConvTranspose2d(64, 32, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.ConvTranspose2d(32, 16, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.ConvTranspose2d(
                16,
                image_channels,
                kernel_size=(5, 7),
                stride=(2, 3),
                padding=(3, 3),
                output_padding=(0, 0),
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1, 19, 295])
        )
        self.extractor = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, 64, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 24, 291])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 12, 97])
            nn.Conv2d(64, 128, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 17, 93])
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(0.32),
            # Tensor shape = ([Batch, 128, 8, 31])
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 14, 27])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 7, 9])
            nn.Conv2d(128, 128, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 7, 5])
            nn.BatchNorm2d(128),
        )
        self.predictor = nn.Sequential(
            # Input tensors shapes = ([Batch, 4480]) & ([Batch, 1024])
            nn.Linear(4480 + 1024, 1024),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024])
            nn.Dropout(0.22),
            nn.Linear(1024, 512),
            nn.ReLU(),
            # Tensor shape = ([Batch, 512])
            nn.Linear(512, output_size),
            # Tensor shape = ([Batch, 2])
        )
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        """
        Function used for prediction, takes as argument each image as
        a tensor and performs a forward pass to return the predicted value.
        """
        x = self.forward(x)
        return x

    def encode(self, x):
        """
        Function used for encoding image, takes as argument each image as
        a tensor and only passes it through the encoder to return the
        flattened encoded image, useful for having latent representation, used for plotting UMAP.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        """
        Full forward pass, takes as argument the batch of images as a
        tensor and performs a pass through the autoencoder, returns the
        predicted value, the reconstructed 3D image as well as the latent vector.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        x_dec_3d = self.decoder(x_enc)
        x = self.extractor(x_dec_3d)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, x_enc_flat], dim=1)
        x = self.predictor(x)
        return self.softmax(x), x_dec_3d, x_enc_flat


class AE_optuna(nn.Module):
    """
    Main component of the CNN for optuna optimization of dropout values,
    similar to fixed class but also takes as arguments trial and configuration
    dictionary of optimization. Tensor shapes after each step can be seen along
    each step of the net.
    """

    def __init__(self, trial, cfg, output_size, image_channels=1):
        super(AE_optuna, self).__init__()
        self.trial = trial
        self.cfg = cfg
        scale1_list=cfg["scale1"]
        scale2_list=cfg["scale2"]
        scale3_list=cfg["scale3"]
        scale4_list=cfg["scale4"]
        ext_scale_list = cfg["ext_scale"]
        scale1 = trial.suggest_categorical("scale1", scale1_list)# 4 8 16
        scale2 = trial.suggest_categorical("scale2", scale2_list)# 1 2 4
        scale3 = trial.suggest_categorical("scale3", scale3_list)# 1 2 4 
        scale4 = trial.suggest_categorical("scale4", scale4_list)# 2 4 8 16
        ext_scale = trial.suggest_categorical("ext_scale", ext_scale_list) # 8 16 32 64
        p_drop_ext = trial.suggest_float(
            "p_drop_ext", cfg["min_dropout_ext"], cfg["max_dropout_ext"]
        )

        p_drop_pred = trial.suggest_float(
            "p_drop_pred", cfg["min_dropout_pred"], cfg["max_dropout_pred"]
        )
        self.encoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, scale1, kernel_size=(3, 7), stride=(3, 7), padding=(3, 3)
            ),  # Output (B, 16, 11, 99)
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.Conv2d(scale1, scale1*scale2, kernel_size=(3, 5), stride=(3, 5), padding=(2, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.Conv2d(scale1*scale2, scale1*scale2*scale3, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.Conv2d(scale1*scale2*scale3, scale1*scale2*scale3*scale4, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024, 1, 1])
            UnFlatten(-1, scale1*scale2*scale3*scale4, 1, 1),
        )
        self.decoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1024, 1, 1])
            nn.ConvTranspose2d(scale1*scale2*scale3*scale4, scale1*scale2*scale3, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.ConvTranspose2d(scale1*scale2*scale3, scale1*scale2, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.ConvTranspose2d(scale1*scale2, scale1, kernel_size=(3, 5), stride=(3, 5), padding=(2, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.ConvTranspose2d(
                scale1,
                image_channels,
                kernel_size=(3, 7), stride=(3, 7), padding=(3, 3),
                output_padding=(1, 0),
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1, 19, 295])
        )
        self.extractor = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, ext_scale, kernel_size=(2, 5), stride=(2, 5), padding=(3, 3)
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 24, 291])
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 12, 97])
            nn.Conv2d(ext_scale, ext_scale*2, kernel_size=(2, 4), stride=(2, 4), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 17, 93])
            #nn.MaxPool2d((2, 3)),
            #nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 8, 31])
            #nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1), padding=(3, 0)),
            #nn.ReLU(),
            # Tensor shape = ([Batch, 128, 14, 27])
            #nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 7, 9])
            #nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1)),
            #nn.ReLU(),
            # Tensor shape = ([Batch, 128, 7, 5])
            nn.BatchNorm2d(ext_scale*2),
        )
        self.predictor = nn.Sequential(
            # Input tensors shapes = ([Batch, 4480]) & ([Batch, 1024])
            nn.Linear(ext_scale*60 + scale1*scale2*scale3*scale4, 512),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024])
            nn.Dropout(p_drop_pred),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Tensor shape = ([Batch, 512])
            nn.Linear(256, output_size),
            # Tensor shape = ([Batch, 2])
        )
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        """
        Function used for prediction, takes as argument each image as
        a tensor and performs a forward pass to return the predicted value.
        """
        x = self.forward(x)
        return x

    def encode(self, x):
        """
        Function used for encoding image, takes as argument each image as
        a tensor and only passes it through the encoder to return the
        flattened encoded image, useful for having latent representation, used for plotting UMAP.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        """
        Full forward pass, takes as argument the batch of images as a
        tensor and performs a pass through the autoencoder, returns the
        predicted value, the reconstructed 3D image as well as the latent vector.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        # print(f"tensor before encoding: {x.shape}")
        # print(f"tensor after encoding: {x_enc.shape}")
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        # print(f"tensor after encoding flat: {x_enc_flat.shape}")
        x_dec_3d = self.decoder(x_enc)
        # print(f"tensor after decoding: {x_dec_3d.shape}")
        x = self.extractor(x_dec_3d)
        # print(f"tensor after extractor: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"tensor after flattening: {x.shape}")
        x = torch.cat([x, x_enc_flat], dim=1)
        # print(f"tensor after concatenation: {x.shape}")
        x = self.predictor(x)
        # print(f"tensor after predictor: {x.shape}")
        return self.softmax(x), x_dec_3d, x_enc_flat


class AE_flexible_simple(nn.Module):
    """
    Main component of the CNN to be run after optimization of dropout rates by
    optuna framework, takes as arguments the optimized dropout values to be used
    during extractor and predictor. Tensor shapes after each step can be seen along
    each step of the net.
    """

    def __init__(self, p_drop_ext, p_drop_pred, scale1, scale2, scale3, scale4, ext_scale, output_size, image_channels=1):
        super(AE_flexible, self).__init__()
        self.encoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, scale1, kernel_size=(3, 7), stride=(3, 7), padding=(3, 3)
            ),  # Output (B, 16, 11, 99)
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.Conv2d(scale1, scale1*scale2, kernel_size=(3, 5), stride=(3, 5), padding=(2, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.Conv2d(scale1*scale2, scale1*scale2*scale3, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.Conv2d(scale1*scale2*scale3, scale1*scale2*scale3*scale4, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024, 1, 1])
            UnFlatten(-1, scale1*scale2*scale3*scale4, 1, 1),
        )
        self.decoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1024, 1, 1])
            nn.ConvTranspose2d(scale1*scale2*scale3*scale4, scale1*scale2*scale3, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.ConvTranspose2d(scale1*scale2*scale3, scale1*scale2, kernel_size=(2, 3), stride=(2, 3)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.ConvTranspose2d(scale1*scale2, scale1, kernel_size=(3, 5), stride=(3, 5), padding=(2, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.ConvTranspose2d(
                scale1,
                image_channels,
                kernel_size=(3, 7), stride=(3, 7), padding=(3, 3),
                output_padding=(1, 0),
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1, 19, 295])
        )
        self.extractor = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, ext_scale, kernel_size=(2, 5), stride=(2, 5), padding=(3, 3)
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 24, 291])
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 12, 97])
            nn.Conv2d(ext_scale, ext_scale*2, kernel_size=(2, 4), stride=(2, 4), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 17, 93])
            #nn.MaxPool2d((2, 3)),
            #nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 8, 31])
            #nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1), padding=(3, 0)),
            #nn.ReLU(),
            # Tensor shape = ([Batch, 128, 14, 27])
            #nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 7, 9])
            #nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1)),
            #nn.ReLU(),
            # Tensor shape = ([Batch, 128, 7, 5])
            nn.BatchNorm2d(ext_scale*2),
        )
        self.predictor = nn.Sequential(
            # Input tensors shapes = ([Batch, 4480]) & ([Batch, 1024])
            nn.Linear(ext_scale*60 + scale1*scale2*scale3*scale4, 512),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024])
            nn.Dropout(p_drop_pred),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Tensor shape = ([Batch, 512])
            nn.Linear(256, output_size),
            # Tensor shape = ([Batch, 2])
        )
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        """
        Function used for prediction, takes as argument each image as
        a tensor and performs a forward pass to return the predicted value.
        """
        x = self.forward(x)
        return x

    def encode(self, x):
        """
        Function used for encoding image, takes as argument each image as
        a tensor and only passes it through the encoder to return the
        flattened encoded image, useful for having latent representation, used for plotting UMAP.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        """
        Full forward pass, takes as argument the batch of images as a
        tensor and performs a pass through the autoencoder, returns the
        predicted value, the reconstructed 3D image as well as the latent vector.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        # print(f"tensor before encoding: {x.shape}")
        # print(f"tensor after encoding: {x_enc.shape}")
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        # print(f"tensor after encoding flat: {x_enc_flat.shape}")
        x_dec_3d = self.decoder(x_enc)
        # print(f"tensor after decoding: {x_dec_3d.shape}")
        x = self.extractor(x_dec_3d)
        # print(f"tensor after extractor: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"tensor after flattening: {x.shape}")
        x = torch.cat([x, x_enc_flat], dim=1)
        # print(f"tensor after concatenation: {x.shape}")
        x = self.predictor(x)
        # print(f"tensor after predictor: {x.shape}")
        return self.softmax(x), x_dec_3d, x_enc_flat
    



class AE_optuna_simple(nn.Module): #Old
    """
    Main component of the CNN for optuna optimization of dropout values,
    similar to fixed class but also takes as arguments trial and configuration
    dictionary of optimization. Tensor shapes after each step can be seen along
    each step of the net.
    """

    def __init__(self, trial, cfg, output_size, image_channels=1):
        super(AE_optuna, self).__init__()
        self.trial = trial
        self.cfg = cfg
        scale1_list=cfg["scale1"]
        scale2_list=cfg["scale2"]
        scale3_list=cfg["scale3"]
        scale4_list=cfg["scale4"]
        ext_scale_list = cfg["ext_scale"]
        scale1 = trial.suggest_categorical("scale1", scale1_list)# 4 8 16
        scale2 = trial.suggest_categorical("scale2", scale2_list)# 1 2 4
        scale3 = trial.suggest_categorical("scale3", scale3_list)# 1 2 4 
        scale4 = trial.suggest_categorical("scale4", scale4_list)# 2 4 8 16
        ext_scale = trial.suggest_categorical("ext_scale", ext_scale_list) # 8 16 32 64
        p_drop_ext = trial.suggest_float(
            "p_drop_ext", cfg["min_dropout_ext"], cfg["max_dropout_ext"]
        )

        p_drop_pred = trial.suggest_float(
            "p_drop_pred", cfg["min_dropout_pred"], cfg["max_dropout_pred"]
        )
        self.encoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, scale1, kernel_size=(5, 7), stride=(2, 3), padding=(3, 3)
            ),  # Output (B, 16, 11, 99)
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.Conv2d(scale1, scale1*scale2, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.Conv2d(scale1*scale2, scale1*scale2*scale3, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.Conv2d(scale1*scale2*scale3, scale1*scale2*scale3*scale4, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024, 1, 1])
            UnFlatten(-1, scale1*scale2*scale3*scale4, 1, 1),
        )
        self.decoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1024, 1, 1])
            nn.ConvTranspose2d(scale1*scale2*scale3*scale4, scale1*scale2*scale3, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.ConvTranspose2d(scale1*scale2*scale3, scale1*scale2, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.ConvTranspose2d(scale1*scale2, scale1, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.ConvTranspose2d(
                scale1,
                image_channels,
                kernel_size=(5, 7),
                stride=(2, 3),
                padding=(3, 3),
                output_padding=(0, 0),
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1, 19, 295])
        )
        self.extractor = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, ext_scale, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 24, 291])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 12, 97])
            nn.Conv2d(ext_scale, ext_scale*2, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 17, 93])
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 8, 31])
            nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 14, 27])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 7, 9])
            nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 7, 5])
            nn.BatchNorm2d(ext_scale*2),
        )
        self.predictor = nn.Sequential(
            # Input tensors shapes = ([Batch, 4480]) & ([Batch, 1024])
            nn.Linear(ext_scale*70 + scale1*scale2*scale3*scale4, 512),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024])
            nn.Dropout(p_drop_pred),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Tensor shape = ([Batch, 512])
            nn.Linear(256, output_size),
            # Tensor shape = ([Batch, 2])
        )
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        """
        Function used for prediction, takes as argument each image as
        a tensor and performs a forward pass to return the predicted value.
        """
        x = self.forward(x)
        return x

    def encode(self, x):
        """
        Function used for encoding image, takes as argument each image as
        a tensor and only passes it through the encoder to return the
        flattened encoded image, useful for having latent representation, used for plotting UMAP.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        """
        Full forward pass, takes as argument the batch of images as a
        tensor and performs a pass through the autoencoder, returns the
        predicted value, the reconstructed 3D image as well as the latent vector.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        # print(f"tensor before encoding: {x.shape}")
        # print(f"tensor after encoding: {x_enc.shape}")
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        # print(f"tensor after encoding flat: {x_enc_flat.shape}")
        x_dec_3d = self.decoder(x_enc)
        # print(f"tensor after decoding: {x_dec_3d.shape}")
        x = self.extractor(x_dec_3d)
        # print(f"tensor after extractor: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"tensor after flattening: {x.shape}")
        x = torch.cat([x, x_enc_flat], dim=1)
        # print(f"tensor after concatenation: {x.shape}")
        x = self.predictor(x)
        # print(f"tensor after predictor: {x.shape}")
        return self.softmax(x), x_dec_3d, x_enc_flat


class AE_flexible(nn.Module): #Old
    """
    Main component of the CNN to be run after optimization of dropout rates by
    optuna framework, takes as arguments the optimized dropout values to be used
    during extractor and predictor. Tensor shapes after each step can be seen along
    each step of the net.
    """

    def __init__(self, p_drop_ext, p_drop_pred, scale1, scale2, scale3, scale4, ext_scale, output_size, image_channels=1):
        super(AE_flexible, self).__init__()
        self.encoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, scale1, kernel_size=(5, 7), stride=(2, 3), padding=(3, 3)
            ),  # Output (B, 16, 11, 99)
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.Conv2d(scale1, scale1*scale2, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.Conv2d(scale1*scale2, scale1*scale2*scale3, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.Conv2d(scale1*scale2*scale3, scale1*scale2*scale3*scale4, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024, 1, 1])
            UnFlatten(-1, scale1*scale2*scale3*scale4, 1, 1),
        )
        self.decoder = nn.Sequential(
            # Input tensor shape = ([Batch, 1024, 1, 1])
            nn.ConvTranspose2d(scale1*scale2*scale3*scale4, scale1*scale2*scale3, kernel_size=(5, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 5, 5])
            nn.ConvTranspose2d(scale1*scale2*scale3, scale1*scale2, kernel_size=(1, 5), stride=(2, 5)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 32, 9, 25])
            nn.ConvTranspose2d(scale1*scale2, scale1, kernel_size=(3, 3), stride=(1, 4)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 16, 11, 99])
            nn.ConvTranspose2d(
                scale1,
                image_channels,
                kernel_size=(5, 7),
                stride=(2, 3),
                padding=(3, 3),
                output_padding=(0, 0),
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1, 19, 295])
        )
        self.extractor = nn.Sequential(
            # Input tensor shape = ([Batch, 1, 19, 295])
            nn.BatchNorm2d(image_channels),
            nn.Conv2d(
                image_channels, ext_scale, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)
            ),
            nn.ReLU(),
            # Tensor shape = ([Batch, 64, 24, 291])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 12, 97])
            nn.Conv2d(ext_scale, ext_scale*2, kernel_size=(2, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 17, 93])
            nn.MaxPool2d((2, 3)),
            nn.Dropout2d(p_drop_ext),
            # Tensor shape = ([Batch, 128, 8, 31])
            nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1), padding=(3, 0)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 14, 27])
            nn.MaxPool2d((2, 3)),
            # Tensor shape = ([Batch, 128, 7, 9])
            nn.Conv2d(ext_scale*2, ext_scale*2, kernel_size=(1, 5), stride=(1, 1)),
            nn.ReLU(),
            # Tensor shape = ([Batch, 128, 7, 5])
            nn.BatchNorm2d(ext_scale*2),
        )
        self.predictor = nn.Sequential(
            # Input tensors shapes = ([Batch, 4480]) & ([Batch, 1024])
            nn.Linear(ext_scale*70 + scale1*scale2*scale3*scale4, 512),
            nn.ReLU(),
            # Tensor shape = ([Batch, 1024])
            nn.Dropout(p_drop_pred),
            nn.Linear(512, 256),
            nn.ReLU(),
            # Tensor shape = ([Batch, 512])
            nn.Linear(256, output_size),
            # Tensor shape = ([Batch, 2])
        )
        self.softmax = nn.Softmax(dim=1)

    def predict(self, x):
        """
        Function used for prediction, takes as argument each image as
        a tensor and performs a forward pass to return the predicted value.
        """
        x = self.forward(x)
        return x

    def encode(self, x):
        """
        Function used for encoding image, takes as argument each image as
        a tensor and only passes it through the encoder to return the
        flattened encoded image, useful for having latent representation, used for plotting UMAP.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        return x_enc_flat

    def forward(self, x):
        """
        Full forward pass, takes as argument the batch of images as a
        tensor and performs a pass through the autoencoder, returns the
        predicted value, the reconstructed 3D image as well as the latent vector.
        """
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        x_enc = self.encoder(x)
        # print(f"tensor before encoding: {x.shape}")
        # print(f"tensor after encoding: {x_enc.shape}")
        x_enc_flat = x_enc.view(x_enc.size(0), -1)
        # print(f"tensor after encoding flat: {x_enc_flat.shape}")
        x_dec_3d = self.decoder(x_enc)
        # print(f"tensor after decoding: {x_dec_3d.shape}")
        x = self.extractor(x_dec_3d)
        # print(f"tensor after extractor: {x.shape}")
        x = x.view(x.size(0), -1)
        # print(f"tensor after flattening: {x.shape}")
        x = torch.cat([x, x_enc_flat], dim=1)
        # print(f"tensor after concatenation: {x.shape}")
        x = self.predictor(x)
        # print(f"tensor after predictor: {x.shape}")
        return self.softmax(x), x_dec_3d, x_enc_flat
    
    