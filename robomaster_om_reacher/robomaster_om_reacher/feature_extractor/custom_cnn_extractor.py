import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor that applies a CNN to image data and concatenates
    with additional vector data, followed by a series of fully connected layers.
    """

    def __init__(self, observation_space, cnn_output_dim=256, vector_input_dim=10):
        super(CustomCNNExtractor, self).__init__(observation_space, features_dim=32)

        # CNN part for image data
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),  # Changed in_channels to 1
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )

        # Compute CNN output size dynamically
        with th.no_grad():
            sample_input = th.zeros(1, *observation_space["scan_matrix"].shape)  # Example input for shape calculation
            print(f"[custom_cnn_extractor] shape of sample_input: {sample_input.shape}")
            cnn_output_size = self.cnn(sample_input).shape[1]

        # Fully connected layer for vector input
        self.fc_vector = nn.Linear(vector_input_dim, vector_input_dim)

        # Fully connected layers after concatenation
        self.fc_layers = nn.Sequential(
            nn.Linear(cnn_output_size + vector_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Set the final features_dim to the output size of the last FC layer
        self._features_dim = 32

    def forward(self, observations):
        print(f"shape of scan_matrix: {observations['scan_matrix'].shape}") 
        # Process image with CNN
        cnn_output = self.cnn(observations["scan_matrix"])

        # Process vector input
        vector_output = self.fc_vector(observations["vector"])

        # Concatenate CNN and vector outputs
        combined_features = th.cat([cnn_output, vector_output], dim=1)

        # Pass through the fully connected layers
        return self.fc_layers(combined_features)
