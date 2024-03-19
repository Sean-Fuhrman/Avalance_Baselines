#%%
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks.classic import SplitCIFAR10
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, forward_transfer_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import InteractiveLogger, TextLogger, TensorboardLogger, WandBLogger, CSVLogger
from avalanche.training.plugins import EvaluationPlugin
import torchvision.transforms as transforms

from avalanche.training import Naive,GDumb, GEM, AGEM, Cumulative,  SynapticIntelligence, Replay
import os
import glob

benchmark = SplitCIFAR10(
    5, 
    return_task_id=False,
    train_transform=transforms.ToTensor(),
    eval_transform=transforms.ToTensor()
)

#Shared init eval function
def init_eval(name):
    loggers = []
    #remove any files in the current logs folder
    for file in glob.glob("./logs/SplitCIFAR10/" + name + "/*"):
        os.remove(file)

    loggers.append(CSVLogger("./logs/SplitCIFAR10/" + name))

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # forward_transfer_metrics(experience=True, stream=True),
        loggers=loggers
    )

    return eval_plugin
def test_strategy(name):

    eval_plugin = init_eval(name)

    # MODEL CREATION
    model = SimpleMLP(input_size=3072, num_classes=benchmark.n_classes)

    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = None
    if name == "Naive":
        cl_strategy = Naive(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    elif name == "GDumb":
        cl_strategy = GDumb(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    elif name == "Replay":
        cl_strategy = Replay(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    elif name == "GEM":
        cl_strategy = GEM(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), patterns_per_exp=500, train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    elif name == "SI": ##DOUBLE CHECK IMPLEMENTATION
        cl_strategy = SynapticIntelligence(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), si_lambda=0.5, train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    elif name == "Cumulative":
        cl_strategy = Cumulative(
            model, SGD(model.parameters(), lr=0.001, momentum=0.9),
            CrossEntropyLoss(), train_mb_size=500, train_epochs=1, eval_mb_size=100,
            evaluator=eval_plugin)
    else:
        raise Exception("Invalid strategy name")
        

    # TRAINING LOOP
    for experience in benchmark.train_stream:
        # train returns a dictionary which contains all the metric values
        cl_strategy.train(experience)
        # test also returns a dictionary which contains all the metric values
        cl_strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    test_strategy("Naive")
    test_strategy("GDumb")
    test_strategy("Replay")
    test_strategy("GEM")
    test_strategy("SI")
    test_strategy("Cumulative")

# %%
