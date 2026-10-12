import torch
print("bắt đầu nạp...", flush=True)
sd = torch.load(r"E:\vjepa2-ac-vitg.pt", map_location="cuda", weights_only=True)
print("nạp xong, số khoá:", len(sd), flush=True)