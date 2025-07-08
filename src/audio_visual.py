import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioVisual(nn.Module):
    def __init__(self, observation_dict, num_actions, num_classes=1):
        super().__init__()
        #self.observation_shape = observation_dict["image"].shape
        #self.observation_shape_sound = observation_dict["sound"].shape
        self.num_actions = num_actions
        self.observation_shape = observation_dict["image"].shape[-3:]
        self.observation_shape_sound = observation_dict["sound"].shape[-3:]

        # Visuel
        self.visual_net = nn.Sequential(
            nn.Conv2d(self.observation_shape[0], 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # Audio
        self.audio_net = nn.Sequential(
            nn.Conv2d(self.observation_shape_sound[0], 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4)),
            nn.Flatten()
        )

        # Features combinés
        self.fc = nn.Sequential(
            nn.Linear(self._get_flattened_size(), 512),
            nn.ReLU()
        )

        # LSTM
        self.lstm = nn.LSTM(512, 256, batch_first=True)

        # Tête de prédiction
        self.output_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=-1)
        )

    def _get_flattened_size(self):
        with torch.no_grad():
            dummy_visual = torch.zeros(1, *self.observation_shape)
            dummy_audio = torch.zeros(1, *self.observation_shape_sound)
            v = self.visual_net(dummy_visual)
            a = self.audio_net(dummy_audio)
            return v.shape[1] + a.shape[1]

    def forward(self, visual_input, audio_input, hidden_state=None):
        batch_size, seq_len = visual_input.shape[:2]

        outputs = []
        for t in range(seq_len):
            vis_feat = self.visual_net(visual_input[:, t])
            aud_feat = self.audio_net(audio_input[:, t])
            combined = torch.cat((vis_feat, aud_feat), dim=1)
            combined_proj = self.fc(combined)
            outputs.append(combined_proj.unsqueeze(1))

        lstm_input = torch.cat(outputs, dim=1)  # (B, T, 512)
        lstm_out, new_hidden = self.lstm(lstm_input, hidden_state)
        final_output = self.output_layer(lstm_out[:, -1])

        return final_output, new_hidden
