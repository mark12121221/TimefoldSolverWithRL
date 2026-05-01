import torch
import torch.nn as nn
import torch.nn.functional as F

from rl_environment import STATE_SIZE, ACTION_SIZE


class PolicyNetwork(nn.Module):

    def __init__(
        self,
        state_size: int = STATE_SIZE,
        action_size: int = ACTION_SIZE,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Three hidden layers, ReLU activations
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        state : torch.Tensor of shape (..., state_size)

        Returns
        -------
        logits : torch.Tensor of shape (..., action_size)
            Raw (un-normalized) scores. Pass through softmax / log_softmax
            for probabilities / log-probabilities.
        """
        return self.net(state)

    def get_action(
        self,
        state: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        greedy: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (or greedily select) an action given the current state.

        Parameters
        ----------
        state       : torch.Tensor of shape (state_size,) or (1, state_size)
        action_mask : optional bool tensor of shape (action_size,) — True where
                      an action is valid. Invalid actions get logit = -1e9.
        greedy      : if True, return argmax instead of sampling.

        Returns
        -------
        action   : torch.Tensor scalar — selected action index
        log_prob : torch.Tensor scalar — log π_θ(action | state)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)  # (1, state_size)

        logits = self.forward(state).squeeze(0)  # (action_size,)

        if action_mask is not None:
            # Mask out invalid actions by setting their logits to -inf
            logits = logits.masked_fill(~action_mask, -1e9)

        log_probs = F.log_softmax(logits, dim=-1)

        if greedy:
            action = torch.argmax(log_probs)
        else:
            probs = torch.exp(log_probs)
            action = torch.multinomial(probs, num_samples=1).squeeze()

        return action, log_probs[action]


def save_model(model: PolicyNetwork, path: str):
    """Save model weights and hyperparameters to a checkpoint file."""
    torch.save({
        'state_dict': model.state_dict(),
        'state_size': model.state_size,
        'action_size': model.action_size,
    }, path)


def load_model(path: str, device: str = 'cpu') -> PolicyNetwork:
    """Load a PolicyNetwork from a checkpoint file."""
    checkpoint = torch.load(path, map_location=device)
    model = PolicyNetwork(
        state_size=checkpoint['state_size'],
        action_size=checkpoint['action_size'],
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model


# Quick smoke-test

if __name__ == '__main__':
    model = PolicyNetwork()
    print(f"PolicyNetwork architecture:\n{model}\n")

    dummy_state = torch.zeros(STATE_SIZE)
    action, log_prob = model.get_action(dummy_state)
    print(f"Dummy action: {action.item()}, log_prob: {log_prob.item():.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")
