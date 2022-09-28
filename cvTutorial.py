import torch

from torch.autograd import Variable

sample_x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
sample_y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))


class LinearRegression(torch.nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        predict_y = self.linear(x)
        return predict_y


def train_model(linear_model):
    define_criterion = torch.nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adagrad(linear_model.parameters(), lr=10)
    # SGD_optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)

    for epoch in range(500):
        predict_y = linear_model(sample_x_data)
        loss = define_criterion(predict_y, sample_y_data)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        print(f'epoch {epoch}, loss function {loss.item()}')


def test_model(linear_model):
    test_variable = Variable(torch.Tensor([[20.0]]))

    predict_y = linear_model(test_variable)

    print("The result of predictions after training", 20, linear_model(test_variable).item())


def predict_model():
    linear_model = LinearRegression()
    train_model(linear_model)
    test_model(linear_model)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    exit(1)

    predict_model()
