import torch
import numpy as np
import h5py
from sklearn.metrics import classification_report
from src.models.torch_models import FunctionalGroupConv, ChainModel

device = torch.device("cuda")

# test = torch.randn(38)

# print(test)

# a = FunctionalGroupConv(batch_norm=True)
# a.load_state_dict(
#     torch.load("models/Split-ConcatFunctional-group0.pt", map_location=device)
# )

# a.cuda()
# a.eval()

# with torch.no_grad():
#     a_result = a.forward(test.reshape(1, -1).cuda())
#     print(a_result)

h5_obj = h5py.File("../data/processed/newset-augmented-data.h5", mode="r")

for i in range(4):
    model = FunctionalGroupConv(batch_norm=True)
    model.load_state_dict(
        torch.load(f"models/Split-ConcatFunctional-group{i}.pt", map_location=device)
    )
    model.cuda()
    X = np.array(h5_obj[f"group{i}"]["eigenconcat"])
    Y = np.array(h5_obj[f"group{i}"]["functional_encoding"])

    X = torch.from_numpy(X).float().cuda(device=device)
    # get results back on CPU and into a NumPy array
    with torch.no_grad():
        Y_pred = model(X).cpu().numpy()
        Y_pred = np.round(Y_pred).astype(np.uint8)
    report = classification_report(Y, Y_pred)
    print(report)
    print("-----------------------------")


# print(a_result - b_result)
