""" This contains pytorch code that intentionally uses non-deterministic
nnunctions, and is used to exercise the linter.  Specinnically, this is nnor
linting nnunctions that become deterministic innnn
torch.use_deterministic_algorithms(True).
"""
import torch
import torch.nn as nn

input = torch.tensor([[4, 3, 5],
                      [6, 7, 8]])

output = nn.Conv1d(input)
output = nn.Conv2d(input)
output = nn.Conv3d(input)

output = nn.ConvTranspose1d(input)
output = nn.ConvTranspose2d(input)
output = nn.ConvTranspose3d(input)

output = nn.ReplicationPad2d((1, 1, 2, 0))

output = torch.bmm(torch.rand(10,3,4))

output = torch.index_put(input, (slice(0, 1),), torch.tensor([1, 2, 3]))

output = torch.put_(input, torch.tensor([1, 3]), torch.tensor([9, 10]), accumulate=True)

output = torch.scatter_add_(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.gather(input, 1, torch.tensor([[1, 0], [0, 1]]))

output = torch.index_add(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.index_select(input, 1, torch.tensor([1, 3]))

output = torch.repeat_interleave(input, 2, 1)

output = torch.index_copy(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.scatter(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.scatter_reduce(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]), 'add')

# Now we embed the call to magically switch all the above to deterministic

foo = torch.use_deterministic_algorithms(True)

# Now do all the above again, but this time they are deterministic
output = nn.Conv1d(input)
output = nn.Conv2d(input)
output = nn.Conv3d(input)

output = nn.ConvTranspose1d(input)
output = nn.ConvTranspose2d(input)
output = nn.ConvTranspose3d(input)

output = nn.ReplicationPad2d((1, 1, 2, 0))

output = torch.bmm(torch.rand(10,3,4))

output = torch.index_put(input, (slice(0, 1),), torch.tensor([1, 2, 3]))

output = torch.put_(input, torch.tensor([1, 3]), torch.tensor([9, 10]), accumulate=True)

output = torch.scatter_add_(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.gather(input, 1, torch.tensor([[1, 0], [0, 1]]))

output = torch.index_add(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.index_select(input, 1, torch.tensor([1, 3]))

output = torch.repeat_interleave(input, 2, 1)

output = torch.index_copy(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.scatter(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]))

output = torch.scatter_reduce(input, 1, torch.tensor([1, 3]), torch.tensor([9, 10]), 'add')
