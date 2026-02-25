"""Entry point for the training worker.

Polls the Redis queue for TrainJob messages pushed by the label studio,
then runs a full training cycle for each job.
"""

import logging
import time

from train_pipeline.configs import settings
from train_pipeline.factory import (
    build_dataset,
    build_loader,
    build_loss,
    build_model,
    build_tracker,
)
from train_pipeline.queue.consumer import TrainQueueConsumer
from train_pipeline.tracking.mlflow_tracker import MLflowTracker
from train_pipeline.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


def _run_training_job(job, s=settings) -> None:
    """Wire everything together and run one training job."""
    dataset = build_dataset(s)
    loader = build_loader(s, dataset)
    model = build_model(s)
    criterion = build_loss(s)
    tracker = build_tracker(s)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        loader=loader,
        tracker=tracker,
        config=s.training,
        registered_model_name=s.registered_model_name,
    )

    log.info("Dataset: %d patches  |  %d batches", len(dataset), len(loader))

    run_params = {
        "job_id": job.job_id,
        "num_pairs": len(job.pairs),
        "model_name": s.model.model_name,
        "num_epochs": s.training.num_epochs,
        "learning_rate": s.training.optimizer.learning_rate,
        "weight_decay": s.training.optimizer.weight_decay,
        "lr_decay_factor": s.training.scheduler.lr_decay_factor,
        "lr_decay_patience": s.training.scheduler.lr_decay_patience,
        "loss_name": s.training.loss.loss_name,
        "dice_weight": s.training.loss.dice_weight,
        "bce_weight": s.training.loss.bce_weight,
        "base_channels": s.model.base_channels,
        "batch_size": s.data.batch_size,
        "window_size": s.data.window_size,
        "stride": s.data.stride,
        "device": str(next(model.parameters()).device),
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }

    tracker.start_run(tags={"job_id": job.job_id})
    try:
        trainer.run(run_params)

        if isinstance(tracker, MLflowTracker):
            run_id = tracker.get_run_id()
            tracker.promote_model(s.registered_model_name, run_id)

        log.info("Training complete for job %s", job.job_id)
    finally:
        tracker.end_run()


def main() -> None:
    consumer = TrainQueueConsumer(settings.redis_url, settings.train_queue_name)
    log.info(
        "Train worker started — polling queue '%s' every %ds",
        settings.train_queue_name,
        settings.poll_interval_sec,
    )

    while True:
        job = consumer.poll()
        if job is None:
            time.sleep(settings.poll_interval_sec)
            continue

        log.info("Received job %s with %d pairs", job.job_id, len(job.pairs))

        try:
            _run_training_job(job)
            consumer.ack(job.job_id)
            log.info("Job %s completed successfully", job.job_id)
        except Exception:
            log.exception("Job %s failed — returning to queue", job.job_id)
            consumer.nack(job.job_id)


if __name__ == "__main__":
    main()
