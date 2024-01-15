import torch
from torch import nn, optim

# Random tensor
t = torch.randn(size=(1, 3, 5))
print(t)
print(t.mean())
print(t.std())
print(t.get_device())

# Linear layer
linear = nn.Linear(10, 2)
example_input = torch.randn(3, 10)
example_output = linear(example_input)
print(example_output)

# ReLU activation function
relu = nn.ReLU()
relu_out = relu(example_output)
print(example_output)
print(relu_out)

# Batch Normalization
batch_norm = nn.BatchNorm1d(2)
batch_norm_out = batch_norm(example_output)

print(batch_norm_out.size())
print(batch_norm_out)
print(batch_norm_out[:, 0].mean())
print(relu_out.mean())
print(relu_out.std())

# Sequential Layer

mlp_layer = nn.Sequential(
    nn.Linear(5, 2),
    nn.BatchNorm1d(2),
    nn.ReLU()
)
test_t = torch.randn(5, 5) + 1
print("Input: ", test_t)
print("Output: ", mlp_layer(test_t))

'''Training Loop
A (basic) training step in PyTorch consists of four basic parts:

1. Set all of the gradients to zero using opt.zero_grad()
2. Calculate the loss, loss
3. Calculate the gradients with respect to the loss using loss.backward()
4. Update the parameters being optimized using opt.step()
That might look like the following code (and you'll notice that if you run it several times, the loss goes down):
'''
# optimizers
adam_opt = optim.Adam(mlp_layer.parameters(), lr=1e-1)
train_example = torch.randn(100, 5) + 1
adam_opt.zero_grad()

curr_loss = torch.abs(1 - mlp_layer(train_example)).mean()

curr_loss.backward()
adam_opt.step()

print(curr_loss)

'''
requires_grad_()
You can also tell PyTorch that it needs to calculate the gradient with respect to a tensor that you created by saying 
example_tensor.requires_grad_(), which will change it in-place. This means that even if PyTorch wouldn't normally store
 a grad for that particular tensor, it will for that specified tensor.

with torch.no_grad():
PyTorch will usually calculate the gradients as it proceeds through a set of operations on tensors. This can often take 
up unnecessary computations and memory, especially if you're performing an evaluation. However, you can wrap a piece of
 code with with torch.no_grad() to prevent the gradients from being calculated in a piece of code.

detach():
Sometimes, you want to calculate and use a tensor's value without calculating its gradients. For example, if you have 
two models, A and B, and you want to directly optimize the parameters of A with respect to the output of B, without
calculating the gradients through B, then you could feed the detached output of B to A. There are many reasons you might
 want to do this, including efficiency or cyclical dependencies (i.e. A depends on B depends on A).

New nn Classes
You can also create new classes which extend the nn module. For these classes, all class attributes, as in self.layer or
self.param will automatically treated as parameters if they are themselves nn objects or if they are tensors wrapped in
nn.Parameter which are initialized with the class.

The __init__ function defines what will happen when the object is created. The first line of the init function of 
a class, for example, WellNamedClass, needs to be super(WellNamedClass, self).__init__().

The forward function defines what runs if you create that object model and pass it a tensor x, as in model(x). If you 
choose the function signature, (self, x), then each call of the forward function, gets two pieces of information: self, 
which is a reference to the object with which you can access all of its parameters, and x, which is the current tensor 
for which you'd like to return y.

'''


class ExampleModule(nn.Module):
    def __init__(self, input_dims, output_dims):
        super(ExampleModule, self).__init__()
        self.linear = nn.Linear(input_dims
                                , output_dims)
        self.exponent = nn.Parameter(torch.tensor(1.))

    def forward(self, x):
        x = self.linear(x)
        x = x ** self.exponent
        return x


print("### Sample Model ###")
example_model = ExampleModule(10, 2)
print(list(example_model.parameters()))
print(list(example_model.named_parameters()))

input_t = torch.randn(2, 10)
print(example_model(input_t))

print("----- Matrix multiplication -----")
a = torch.randn(2, 3)
b = torch.randn(3, 2)

print(a)
print(b)
c = a @ b
print(c)

