import asyncio
import io
import time
from typing import List, Tuple
import numpy as np
from PIL import Image
import onnxruntime as ort

CIFAR10_CLASSES=[
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

NORM_MEAN=np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(3,1,1)
NORM_STD=np.array([0.2470, 0.2435, 0.2616], dtype=np.float32).reshape(3,1,1)

def preprocess_image_bytes(image_bytes: bytes)-> np.ndarray:
    image=Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image=image.resize((32,32), Image.Resampling.BILINEAR)
    arr=np.array(image, dtype=np.float32)/255.0
    arr=np.transpose(arr, (2,0,1))
    arr=(arr-NORM_MEAN)/NORM_STD
    return arr

def preprocess_tensor(tensor_list: list) -> np.ndarray:
    arr=np.array(tensor_list, dtype=np.float32)
    if arr.shape!=(3,32,32):
        raise ValueError(f"Expected tensor shape (3, 32, 32), got {arr.shape}")
    return arr

def softmax(x:np.ndarray)->np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

class AsyncBatcher:
    def __init__(
        self,
        session: ort.InferenceSession,
        max_batch_size: int=32,
        max_wait_ms: float=5.0
    ):
        self.session=session
        self.input_name=session.get_inputs()[0].name
        self.output_name=session.get_outputs()[0].name
        self.max_batch_size=max_batch_size
        self.max_wait_ms=max_wait_ms
        self.queue: asyncio.Queue=asyncio.Queue()
        self.worker_task: asyncio.Task = None
        
    def start(self):
        self.worker_task=asyncio.create_task(self._batch_worker())
        
    async def stop(self):
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            
    async def predict(self, preprocessed_arr: np.ndarray)->dict:
        loop=asyncio.get_running_loop()
        future: asyncio.Future=loop.create_future()
        await self.queue.put((preprocessed_arr, future))
        return await future
    
    async def _batch_worker(self):
        while True:
            items: List[Tuple[np.ndarray, asyncio.Future]]=[]
            item=await self.queue.get()
            items.append(item)
            
            start_wait=time.perf_counter()
            while len(items)< self.max_batch_size:
                timeout=(self.max_wait_ms/1000.0) - (time.perf_counter() - start_wait)
                if timeout<=0:
                    break
                try:
                    next_item=await asyncio.wait_for(self.queue.get(), timeout=timeout)
                    items.append(next_item)
                except asyncio.TimeoutError:
                    break
                
            try:
                batch_tensors=np.stack([it[0] for it in items], axis=0)
                actual_batch_size=batch_tensors.shape[0]
                
                t0=time.perf_counter()
                outputs=self.session.run(
                    [self.output_name],
                    {self.input_name: batch_tensors}
                )[0]
                
                total_latency=(time.perf_counter() - t0) * 1000.0
                
                probs=softmax(outputs)
                per_item_latency=total_latency/actual_batch_size
                
                for idx,(_,future) in enumerate(items):
                    if not future.cancelled():
                        item_probs=probs[idx]
                        top_id=int(np.argmax(item_probs))
                        top_k_indices=np.argsort(item_probs)[::-1][:3]
                        
                        top_3=[
                            {
                                "class_id": int(i),
                                "class_name": CIFAR10_CLASSES[i],
                                "confidence": float(item_probs[i])
                            }
                            for i in top_k_indices
                        ]
                        
                        future.set_result({
                            "class_id":top_id,
                            "class_name": CIFAR10_CLASSES[top_id],
                            "confidence": float(item_probs[top_id]),
                            "top_3": top_3,
                            "latency_ms": per_item_latency,
                            "batched_size": actual_batch_size
                        })
                        
            except Exception as e:
                for _, future in items:
                    if not future.cancelled():
                        future.set_exception(e)
            finally:
                for _ in range(len(items)):
                    self.queue.task_done()                        