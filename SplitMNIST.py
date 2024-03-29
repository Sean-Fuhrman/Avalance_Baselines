#%%
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from torch import nn
from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation.metrics import forgetting_metrics, \
accuracy_metrics, loss_metrics, forward_transfer_metrics
from avalanche.models import SimpleMLP
from avalanche.logging import CSVLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training import ICaRL, GEM, LwF, EWC, SynapticIntelligence, MAS, Cumulative, Naive
import os
import glob
from torchvision import transforms


benchmark = SplitMNIST(n_experiences=5, return_task_id=False)

#Shared init eval function
def init_eval(name):
    loggers = []
    #remove any files in the current logs folder
    for file in glob.glob("./logs/SplitMNIST/" + name + "/*"):
        os.remove(file)

    loggers.append(CSVLogger("./logs/SplitMNIST/" + name))

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
    # model = SimpleMLP(num_classes=benchmark.n_classes)
   
    # CREATE THE STRATEGY INSTANCE (NAIVE)
    cl_strategy = None 
    model = SimpleMLP(num_classes=benchmark.n_classes)

    if name == "ICaRL": #Replay methods
        feature_size = 64 
        feature_extractor = SimpleMLP(num_classes=feature_size)
        classifier = nn.Linear(feature_size, benchmark.n_classes)

        buffer_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor()
        ])
        cl_strategy = ICaRL(
            feature_extractor=feature_extractor, classifier=classifier,
            optimizer=Adam(list(feature_extractor.parameters()) + list(classifier.parameters()), lr=0.001),
            buffer_transform=buffer_transform,
            fixed_memory=True, memory_size=1000,
            evaluator=eval_plugin)
    elif name == "GEM":
        optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
        criterion = CrossEntropyLoss()
        cl_strategy = GEM(model, optimizer, criterion, patterns_per_exp=2, memory_strength=0.5, evaluator=eval_plugin)
    elif name == "LwF": #Regularization methods
        pass
    elif name == "EBLL":
        pass
    elif name == "EWC": 
        pass
    elif name == "SI":
        pass
    elif name == "MAS":
        pass
    elif name == "mean-IMM":
        pass
    elif name == "mode-IMM":
        pass
    elif name == "PackNet": #Parameter isolation methods
        pass
    elif name == "HAT":
        pass
    elif name == "Cumulative": #Baseline methods
        pass
    elif name == "Naive":
        pass
    else:
        raise Exception("Invalid strategy name")
        

    # TRAINING LOOP
    for experience in benchmark.train_stream:
        # train returns a dictionary which contains all the metric values
        cl_strategy.train(experience)
        # test also returns a dictionary which contains all the metric values
        cl_strategy.eval(benchmark.test_stream)


if __name__ == "__main__":
    # strategies = ["ICaRL", "GEM", "LwF", "EBLL", "EWC", "SI", "MAS", "mean-IMM", "mode-IMM", "PackNet", "HAT", "Cumulative", "Naive"]
    # for strategy in strategies:
    #     test_strategy(strategy)
    test_strategy("GEM")

# %%
