import numpy as np

from typing_extensions import override

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from torch.nn import Sequential, ReLU, Linear
from torch import utils,Tensor
import lightning as L
from .deep_detection import AutoEncoder

from sklearn.metrics import roc_auc_score

import optuna

from sklearn.model_selection import train_test_split


model_map = { "ForestBased": IsolationForest,
             "SVMBased": OneClassSVM,
             "NearestNeigborBased": LocalOutlierFactor,
             "AutoEncoderBased": AutoEncoder}

class AnomalyModel():
    """
    This M
    """
    def __init__(self, type: str = 'ForestBased') -> None:
        self.type = type
        self.X = None
        self.y = None

    def fit(self):
        pass

    def predict(self):
        pass 

    def _optimize():
        pass

class NearestNeighborBased(AnomalyModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass 

    def predict(self):
        pass 

class AutoEncoderBased(AnomalyModel):
    def __init__(self) -> None:
        super().__init__(type="AutoEncoderBased")
        self.trainer = L.Trainer(limit_train_batches=100, 
                                 max_epochs=1)
        self.loader = None
    
    @override
    def fit(self, X):
        #self._optimize()

        encoder = Sequential(Linear(X.shape[1], 64), 
                                    ReLU(), 
                                        Linear(64, 3))
        
        decoder = Sequential(Linear(3, 64), 
                                ReLU(), 
                                    Linear(64, X.shape[1]))

        if self.y is None:
            self.y = np.zeros(shape=(X.shape[0], 1) ,
                              dtype=np.float32) # Making it work with Autoencoder
        
        model = model_map[self.type](encoder=encoder, 
                                            decoder= decoder)

        
        trainset = utils.data.TensorDataset(Tensor(X), Tensor(self.y))

        self.trainer.fit(model=model, 
                        train_dataloaders = utils.data.DataLoader(trainset))

        return model

    def predict(self):
        
        pass 

    def _optimize():
        """
        Optimization will be tricky for this loop. 
        Read up some of the optimization strategies based on how the model performs.
        """

        pass

class SVMBased(AnomalyModel):
    def __init__(self) -> None:
        super().__init__()

    def fit(self):
        pass 

    def predict(self):
        pass 

    def _optimize():
        return None 

class ForestBased(AnomalyModel):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y = None):
        self.X = X 
        self.y = y

        if self.y is None:
            self.model = model_map[self.type](*self._optimize())
        else:
            self.model = model_map[self.type]()
        
        self.model.fit(X)


    def predict(self, X):
        return self.model.predict(X)

    def _optimize(self):
        if self.y.is_empty():
            study = optuna.create_study(direction='maximize')
            study.optimize(self.objective, 
                        n_trials=50) # Default 50 trials.
            trial = study.best_trial

        return trial.params 
    
    def objective(self, trial):        
        # Define hyperparameters to tune
        train_x, val_x ,train_y, val_y = train_test_split(self.X, self.y)
        
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_samples = trial.suggest_float('max_samples', 0.1, 1.0)
        contamination = trial.suggest_float('contamination', 0.01, 0.5)
        max_features = trial.suggest_float('max_features', 0.1, 1.0)
        
        # Create and train the model
        model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            random_state=42
        )
        model.fit(train_x) # y is not supposed to be there. 
        
        # Predict and evaluate
        y_pred = model.predict(val_x)
        y_pred = np.where(y_pred == -1, 0, 1) # Convert -1 to 1.
        score = roc_auc_score(val_y, y_pred)
    
        return score

class AnomalySuite:
    def __init__(self) -> None:
        self.model_suite = [AnomalyModel(model) for model in model_map.keys()]
        self.y = None
        self.x = None

    def fit_predict(self, X, y=None):
        for model in self.model_suite:
            self.model.fit(X)
        return None
    
    def predict_scores(self, X):
        return self.predict_scores(X)
    
    def return_scores(self,X):
        pass
    


