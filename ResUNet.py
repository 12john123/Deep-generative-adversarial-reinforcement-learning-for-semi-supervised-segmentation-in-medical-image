import torch
import torch.nn as nn
import torch.nn.functional as F


class ResUNet_core(nn.Module):
    def __init__(self, training=True):
        super().__init__()

        self.training = training
        self.dorp_rate = 0.2

        self.encoder_stage1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.InstanceNorm2d(16),
            # nn.PReLU(16),
            nn.LeakyReLU(),

            nn.Conv2d(16, 16, 3, 1, padding=1),
            nn.InstanceNorm2d(16),
            # nn.PReLU(16),
            nn.LeakyReLU(),
        )

        self.encoder_stage2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),
        )

        self.encoder_stage3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, 1, padding=2, dilation=2),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, 1, padding=4, dilation=4),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),
        )

        self.encoder_stage4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, padding=3, dilation=3),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, padding=4, dilation=4),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, padding=5, dilation=5),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),
        )

        self.decoder_stage1 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm2d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm2d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.InstanceNorm2d(256),
            # nn.PReLU(256),
            nn.LeakyReLU(),
        )

        self.decoder_stage2 = nn.Sequential(
            nn.Conv2d(128 + 64, 128, 3, 1, padding=1),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128),
            nn.LeakyReLU(),
        )

        self.decoder_stage3 = nn.Sequential(
            nn.Conv2d(64 + 32, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64),
            nn.LeakyReLU(),
        )

        self.decoder_stage4 = nn.Sequential(
            nn.Conv2d(32 + 16, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32),
            nn.LeakyReLU(),
        )

        self.down_conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 2, 2),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32)
            nn.LeakyReLU(),
        )

        self.down_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 2, 2),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64)
            nn.LeakyReLU(),
        )

        self.down_conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 2, 2),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128)
            nn.LeakyReLU(),
        )

        self.down_conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.InstanceNorm2d(256),
            # nn.PReLU(256)
            nn.LeakyReLU(),
        )

        self.up_conv2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, 2),
            nn.InstanceNorm2d(128),
            # nn.PReLU(128)
            nn.LeakyReLU(),
        )

        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 2, 2),
            nn.InstanceNorm2d(64),
            # nn.PReLU(64)
            nn.LeakyReLU(),
        )

        self.up_conv4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, 2),
            nn.InstanceNorm2d(32),
            # nn.PReLU(32)
            nn.LeakyReLU(),
        )

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs) + inputs

        short_range1 = self.down_conv1(long_range1)

        long_range2 = self.encoder_stage2(short_range1) + short_range1
        long_range2 = F.dropout(long_range2, self.dorp_rate, self.training)

        short_range2 = self.down_conv2(long_range2)

        long_range3 = self.encoder_stage3(short_range2) + short_range2
        long_range3 = F.dropout(long_range3, self.dorp_rate, self.training)

        short_range3 = self.down_conv3(long_range3)

        long_range4 = self.encoder_stage4(short_range3) + short_range3
        long_range4 = F.dropout(long_range4, self.dorp_rate, self.training)

        short_range4 = self.down_conv4(long_range4)

        outputs = self.decoder_stage1(long_range4) + short_range4
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range6 = self.up_conv2(outputs)

        outputs = self.decoder_stage2(torch.cat([short_range6, long_range3], dim=1)) + short_range6
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range7 = self.up_conv3(outputs)

        outputs = self.decoder_stage3(torch.cat([short_range7, long_range2], dim=1)) + short_range7
        outputs = F.dropout(outputs, self.dorp_rate, self.training)

        short_range8 = self.up_conv4(outputs)

        outputs = self.decoder_stage4(torch.cat([short_range8, long_range1], dim=1)) + short_range8

        return outputs


class ResUNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2, training=True):
        super().__init__()

        self.encoder_stage0 = nn.Sequential(
            nn.Conv2d(in_channel, 8, 3, 1, padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),

            nn.Conv2d(8, 8, 3, 1, padding=1),
            nn.InstanceNorm2d(8),
            nn.LeakyReLU(),

            nn.Conv2d(8, 16, 3, 1, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(),
        )

        self.resunet = ResUNet_core(training=training)

        self.map = nn.Conv2d(32, out_channel, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.encoder_stage0(x)
        x = self.resunet(x)
        x = self.map(x)
        x = self.sigmoid(x)
        return x
