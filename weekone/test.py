class Conv5Net_depth(nn.Module):
    def __init__(self):
        super(Conv5Net_depth, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(32, 48, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(48, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True),

            nn.Conv2d(64, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(96, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

            #             nn.Conv2d(96,2, kernel_size=1, stride=1, padding=0),
            #             nn.BatchNorm2d(2),
            #             nn.ReLU(inplace=True),
            #             nn.AvgPool2d(kernel_size=6, stride=1, ceil_mode=True)
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 3755)
        )

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), 512 * 3 * 3)
        x = self.classifier(x)
        return x
