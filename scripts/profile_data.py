"""Time task.sample_batch alone on the current device (after warmup)."""
import sys, time
import torch, yaml
from witnesses.tasks import get_task
from witnesses.meta import get_device
dev = get_device("auto")
for path in sys.argv[1:]:
    cfg = yaml.safe_load(open(path))
    t = get_task(cfg["task"], **cfg.get("task_args", {}))
    bs = cfg.get("optim", {}).get("batch_size", 256)
    for _ in range(5):
        t.sample_batch(bs, dev)
    if dev.type == "cuda": torch.cuda.synchronize()
    t0 = time.time(); n = 30
    for _ in range(n):
        t.sample_batch(bs, dev)
    if dev.type == "cuda": torch.cuda.synchronize()
    print(f"{path:30s} sample_batch {1000*(time.time()-t0)/n:7.1f} ms")
