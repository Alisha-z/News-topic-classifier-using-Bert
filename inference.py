from tensorboard.backend.event_processing import event_accumulator

log_path = "logs/events.out.tfevents.1756391481.DESKTOP-EA7Q8KT.896.0"

ea = event_accumulator.EventAccumulator(log_path, size_guidance={"scalars": 0})
ea.Reload()

# List available metrics
print("Available tags:", ea.Tags()["scalars"])

# Example: get training loss
loss_events = ea.Scalars("train/loss")  # adjust the tag name
for e in loss_events[:10]:
    print(f"Step: {e.step}, Value: {e.value}")
