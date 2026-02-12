"""Train stage - trains VAE model using scropipe.synth."""

from pathlib import Path

from .base import Stage, StageResult


class TrainStage(Stage):
    """Stage that trains a VAE model on spectrograms."""

    name = "03-model"
    description = "Train VAE model on spectrograms"

    def run(
        self,
        data_dir: Path,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        z_dim: int = 64,
        kl_weight: float = 0.001,
    ) -> StageResult:
        """Run the train stage.

        Args:
            data_dir: Directory containing preprocessed spectrograms.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            learning_rate: Learning rate.
            z_dim: Latent space dimension.
            kl_weight: Weight for KL divergence in loss.

        Returns:
            StageResult with success status.
        """
        data_dir = Path(data_dir)
        if not data_dir.exists():
            return StageResult(
                success=False,
                message=f"Data directory not found: {data_dir}",
            )

        # Check for spectrograms
        npy_files = list(data_dir.glob("*.npy"))
        if not npy_files:
            return StageResult(
                success=False,
                message=f"No spectrogram files found in {data_dir}",
            )

        # Import synth modules (requires ML dependencies)
        try:
            import torch
            from ..synth.data_loader import create_dataloader
            from ..synth.model import VAE
            from ..synth.train_utils import Trainer
        except ImportError as e:
            return StageResult(
                success=False,
                message=f"ML dependencies not installed. Run: pip install scropipe[ml]\nError: {e}",
            )

        output_dir = self.ensure_output_dir()
        model_path = output_dir / "model.pth"

        try:
            # Create dataloader
            dataloader, dataset = create_dataloader(
                data_dir,
                batch_size=batch_size,
                shuffle=True,
                normalize=True,
            )

            # Get input shape from dataset
            input_shape = dataset.spec_shape
            self.log(f"[dim]Spectrogram shape: {input_shape}[/dim]")
            self.log(f"[dim]Latent dimension: {z_dim}[/dim]")
            self.log(f"[dim]Dataset size: {len(dataset)} samples[/dim]")

            # Create model
            model = VAE(input_shape=input_shape, z_dim=z_dim)

            # Count parameters
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            self.log(f"[dim]Model parameters: {num_params:,}[/dim]")

            # Create trainer and train
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            trainer = Trainer(
                model=model,
                dataloader=dataloader,
                dataset=dataset,
                lr=learning_rate,
                kl_weight=kl_weight,
                device=device,
            )

            trainer.train(epochs=epochs, save_path=model_path)

            if not model_path.exists():
                return StageResult(
                    success=False,
                    output_dir=output_dir,
                    message="Model file not created",
                )

            self.log_success(f"Model saved to {model_path}")

            return StageResult(
                success=True,
                output_dir=output_dir,
                message=f"Model trained and saved to {model_path}",
                details={
                    "model_path": str(model_path),
                    "epochs": epochs,
                    "z_dim": z_dim,
                },
            )

        except Exception as e:
            return StageResult(
                success=False,
                message=f"Unexpected error: {e}",
            )
