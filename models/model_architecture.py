# =============================================================================
# model_architecture.py — Enhanced U-Net (MATCHES TRAINED WEIGHTS)
# =============================================================================

import torch
import torch.nn as nn


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + self.skip(x))
        return out


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, 1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class EnhancedUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # Encoder
        self.enc1 = ResidualConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = ResidualConvBlock(512, 1024)

        # Decoder  ⬅️ IMPORTANT: NAMES MATCH CHECKPOINT
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, 2)
        self.att4 = AttentionGate(512, 512, 256)
        self.dec4 = ResidualConvBlock(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.att3 = AttentionGate(256, 256, 128)
        self.dec3 = ResidualConvBlock(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.att2 = AttentionGate(128, 128, 64)
        self.dec2 = ResidualConvBlock(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.att1 = AttentionGate(64, 64, 32)
        self.dec1 = ResidualConvBlock(128, 64)

        self.out = nn.Sequential(
            nn.Conv2d(64, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))

        x = self.bridge(self.pool4(enc4))

        x = self.upconv4(x)
        x = self.dec4(torch.cat([x, self.att4(x, enc4)], dim=1))

        x = self.upconv3(x)
        x = self.dec3(torch.cat([x, self.att3(x, enc3)], dim=1))

        x = self.upconv2(x)
        x = self.dec2(torch.cat([x, self.att2(x, enc2)], dim=1))

        x = self.upconv1(x)
        x = self.dec1(torch.cat([x, self.att1(x, enc1)], dim=1))

        return self.out(x)
