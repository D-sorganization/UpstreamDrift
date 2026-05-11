import logging
from pathlib import Path
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

from .inverse_surrogate import MyoSuiteInverseSurrogate, InverseSurrogateConfig

logger = logging.getLogger(__name__)


class SurrogateDataset(Dataset):
    """Dataset for training the MyoSuite Inverse Surrogate.

    In a real implementation, this would load trajectories from HDF5 or NPZ
    files containing recorded (joint_q, joint_v) kinematics and the
    corresponding muscle activations computed via a slow offline optimizer.
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int = 300,
        n_joints: int = 22,
        n_muscles: int = 290,
    ) -> None:
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.n_joints = n_joints
        self.n_muscles = n_muscles

        # Mock dataset of 100 samples
        self.num_samples = 100

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return random mock data for phase 3 infrastructure skeleton
        joint_q = torch.randn(self.seq_len, self.n_joints)
        joint_v = torch.randn(self.seq_len, self.n_joints)
        muscle_act = torch.rand(self.seq_len, self.n_muscles)  # [0, 1] range
        return joint_q, joint_v, muscle_act


def train_surrogate(
    data_path: str | Path,
    epochs: int = 10,
    batch_size: int = 8,
    lr: float = 1e-3,
    save_dir: str | Path = "weights",
) -> None:
    """Train the MyoSuite inverse surrogate model.

    Args:
        data_path: Path to the training dataset.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate for Adam optimizer.
        save_dir: Directory to save model checkpoints.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    n_joints = 22
    n_muscles = 290
    seq_len = 300

    cfg = InverseSurrogateConfig(
        n_joints=n_joints, n_muscles=n_muscles, seq_len=seq_len
    )
    model = MyoSuiteInverseSurrogate(cfg)
    optimizer = Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = SurrogateDataset(data_path, seq_len, n_joints, n_muscles)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Starting training loop for {epochs} epochs...")

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for joint_q, joint_v, target_muscles in dataloader:
            optimizer.zero_grad()

            # Forward pass
            pred_muscles = model(joint_q, joint_v)

            # Compute loss
            loss = criterion(pred_muscles, target_muscles)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch [{epoch + 1}/{epochs}] Loss: {avg_loss:.6f}")

    weights_path = save_dir / "surrogate_weights.pt"
    torch.save(model.state_dict(), weights_path)
    logger.info(f"Training complete. Weights saved to {weights_path}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_surrogate("mock_data_dir")
