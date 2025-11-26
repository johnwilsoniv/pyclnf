import torch
print(torch.cuda.is_available())        # Should be True
print(torch.version.cuda)               # Should show 13.0
print(torch.cuda.get_device_name(0))    # Your GPU name