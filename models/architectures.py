"""
Model architectures for deepfake detection.
Contains 3D-CNN and SyncNet implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Simple3DCNN(nn.Module):
    """
    Simple 3D-CNN for temporal deepfake detection.
    Analyzes sequences of frames for temporal artifacts.
    """
    
    def __init__(self, num_classes=2, input_channels=3):
        """
        Initialize 3D-CNN model.
        
        Args:
            num_classes: Number of output classes (2 for binary: real/fake)
            input_channels: Number of input channels (3 for RGB)
        """
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        # Input shape: (batch, channels, frames, height, width)
        self.conv1 = nn.Conv3d(input_channels, 64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, frames, height, width)
            
        Returns:
            Output logits of shape (batch, num_classes)
        """
        # x shape: (batch, 3, T, 224, 224)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)  # (batch, 64, T, 112, 112)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)  # (batch, 128, T/2, 56, 56)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)  # (batch, 256, T/4, 28, 28)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, 256, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (batch, 256)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class SimpleSyncNet(nn.Module):
    """
    Simple SyncNet-style model for lip-sync detection.
    Compares audio and visual features to detect synchronization.
    """
    
    def __init__(self, audio_feature_dim=13, visual_feature_dim=512):
        """
        Initialize SyncNet model.
        
        Args:
            audio_feature_dim: Dimension of audio features (e.g., MFCC features)
            visual_feature_dim: Dimension of visual features from mouth crops
        """
        super(SimpleSyncNet, self).__init__()
        
        # Visual encoder (CNN for mouth crops)
        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, visual_feature_dim)
        )
        
        # Audio encoder (1D CNN for audio features)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_feature_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, visual_feature_dim)  # Match visual feature dim
        )
        
        # Sync prediction head
        # Computes similarity between audio and visual features
        self.sync_head = nn.Sequential(
            nn.Linear(visual_feature_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output sync score 0-1
        )
        
    def forward(self, visual_input, audio_input):
        """
        Forward pass.
        
        Args:
            visual_input: Mouth crop images, shape (batch, 3, H, W)
            audio_input: Audio features, shape (batch, audio_feature_dim, time)
            
        Returns:
            Sync score between 0 and 1 (higher = better sync)
        """
        # Encode visual features
        visual_features = self.visual_encoder(visual_input)  # (batch, visual_feature_dim)
        
        # Encode audio features
        audio_features = self.audio_encoder(audio_input)  # (batch, visual_feature_dim)
        
        # Concatenate features
        combined = torch.cat([visual_features, audio_features], dim=1)  # (batch, visual_feature_dim * 2)
        
        # Predict sync score
        sync_score = self.sync_head(combined)  # (batch, 1)
        
        return sync_score.squeeze(1)  # (batch,)

