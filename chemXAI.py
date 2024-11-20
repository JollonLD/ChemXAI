import shap
import sklearn.datasets
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import os
from lime import lime_tabular
from abc import ABC, abstractmethod
import torch


class DataSet:
    
    def __init__(self):
        self.directory_path = r"C:\IC\toolboxXAI\DataSets\QM9"
        self.properties = [
            'Rotational constant A: GHz',
            'Rotational constant B: GHz',
            'Rotational constant C: GHz',
            'Dipole moment (μ): Debye (D)',
            'Isotropic polarizability (α): atomic units (a.u.)',
            'Energy of HOMO (ϵHOMO): Hartree (Ha)',
            'Energy of LUMO (ϵLUMO): Hartree (Ha)',
            'Gap (ϵgap): Hartree (Ha)',
            'Electronic spatial extent: atomic units (a.u.)',
            'Zero point vibrational energy (zpve): Hartree (Ha)',
            'Internal energy at 0 K (U0): Hartree (Ha)',
            'Internal energy at 298.15 K (U): Hartree (Ha)',
            'Enthalpy at 298.15 K (H): Hartree (Ha)',
            'Free energy at 298.15 K (G): Hartree (Ha)',
            'Heat capacity at 298.15 K (Cv): cal/mol·K'
        ]
        
    
    def load_qm9_xyz(self, file_path):
        """Load a single QM9.xyz file."""
        
        with open(file_path, 'r') as f:
            # Number of atoms
            natoms = int(f.readline())
            # Properties are in the second line
            properties = list(map(float, f.readline().split()[2:]))
            # Read atomic coordinates and types
            atoms = []
            coordinates = []
            for num_line, line in enumerate(f):
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))

        return {
            "natoms": natoms,
            "atoms": atoms,
            "coordinates": np.array(coordinates),
            "properties": properties
        }

    def load_qm9_dataset(self, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        
        coords = []
        prop = []
        natoms = []
        i = 0
        for file_name in os.listdir(self.directory_path):
            if i == 10:
                break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if molecule_data['natoms'] in list_mols or len(list_mols) == 0:
                    coords.append([molecule_data['atoms'], molecule_data['coordinates']])
                    prop.append(molecule_data['properties'])
                    natoms.append(molecule_data['natoms'])
            i += 1
        return coords, prop, natoms
    
    def dataset_to_numpy(dataset):  
        """Convert a dataset to NumPy arrays."""
        
        all_data = []
        all_targets = []

        for data, target in dataset:
            all_data.append(data.numpy())
            all_targets.append(target.numpy())
            
        data_numpy = np.array(all_data)
        targets_numpy = np.array(all_targets)
        
        return data_numpy, targets_numpy


    
    def get_smiles(self, file_path):
        """Get the SMILES representation of a molecule."""
        
        smiles = []
        
        with open(file_path, 'r') as f:
            # Read number of atoms
            natoms = int(f.readline())
            # Skip the second line
            f.readline()
            for i in range(natoms+1):  # Skip to the SMILES line
                f.readline()
            smiles_tuple = tuple(f.readline().strip().split('\t'))
            smiles.append(smiles_tuple)

        return smiles
        
    def load_smiles(self):
        """Load all SMILES representations in the QM9 dataset."""    
    
        list_smiles = []
        i = 0
        for file_name in os.listdir(self.directory_path):
            if i == 100:
                break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                list_smiles.append(self.get_smiles(file_path))
            i += 1
        
        return list_smiles    
    
    def df_props(self):
        """Create a DataFrame with the properties of the QM9 dataset."""
        
        coords, props, natoms = self.load_qm9_dataset()
        
        # Create the DataFrame with properties
        df = pd.DataFrame(props)
        
        # Reset the DataFrame indices
        df.reset_index(drop=True, inplace=True)
        
        # Rename the DataFrame columns
        df.columns = self.properties
        
        return df  
  

class Shap:
    def __init__(self, model, train_loader, test_loader, device):
        """
        Initializes the Shap class with the model, training, and test data.
        
        Parameters:
        - model: model to be explained.
        - train_loader: training DataLoader to obtain the background data.
        - test_loader: test DataLoader for explanations.
        - device: device (CPU or GPU) for operations.
        """
        self.model = model
        self.device = device

        # Get a batch from the training DataLoader as background data
        background = next(iter(train_loader))[0].cpu().numpy()  # Convert to numpy for KernelExplainer
        print("Background shape:", background.shape)  # Check background shape
        
        # Define prediction function for KernelExplainer compatibility with PyTorch model
        def predict_fn(data):
            self.model.eval()
            with torch.no_grad():
                data_tensor = torch.from_numpy(data).float().to(self.device)
                return self.model(data_tensor).cpu().numpy()
        
        # Initialize KernelExplainer with model and background data
        self.explainer = shap.KernelExplainer(predict_fn, background)
        
        # Load test data for explanations
        self.test_data, _ = next(iter(test_loader))
        self.test_data = self.test_data.cpu().numpy()  # Convert to numpy for KernelExplainer
        print("Test data shape:", self.test_data.shape)  # Check test data shape
        
        # Compute shap_values for test data
        self.shap_values = self.explainer.shap_values(self.test_data)

    def local_explanation(self, index):
        """
        Generates a local explanation for a specific instance and displays a DataFrame
        with feature indices and SHAP values for the chosen instance.
        
        Parameters:
        - index: index of the instance in the test set to be explained.
        """
        # Select SHAP values for the instance and flatten the extra dimension
        local_shap_values = self.shap_values[index].flatten()
        
        # Create a DataFrame with feature indices and SHAP values
        feature_importance = pd.DataFrame({
            'Feature Index': range(len(local_shap_values)),
            'SHAP Value': local_shap_values
        }).sort_values(by='SHAP Value', ascending=False).reset_index(drop=True)
        
        return feature_importance

    def global_explanation(self):
        """
        Generates global explanations by calculating the average feature importance for each instance
        in the test data and the overall mean importance across all instances.
        
        Returns:
        - all_local_importances: DataFrame where each row represents an instance and each column represents a feature.
        - global_feature_importance: DataFrame with average feature importance across all instances.
        """
        # Squeeze shap_values to remove any extra dimension if present
        shap_values_2d = np.squeeze(self.shap_values)  # Converts to Matrix
        
        # Compute local explanations for each instance and collect them in a DataFrame
        all_local_importances = pd.DataFrame(shap_values_2d)
        all_local_importances.columns = [f'Feature {i}' for i in range(shap_values_2d.shape[1])]
        
        # Compute global importance as the mean of absolute SHAP values across all instances
        mean_absolute_shap_values = np.mean(np.abs(shap_values_2d), axis=0)
        
        # Additional check on the shape of the mean SHAP values
        print("Global SHAP values shape:", mean_absolute_shap_values.shape)
        
        # DataFrame with mean absolute feature importance across all instances
        global_feature_importance = pd.DataFrame({
            'Feature': [f'{i}' for i in range(len(mean_absolute_shap_values))],
            'Importance': mean_absolute_shap_values
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return all_local_importances, global_feature_importance

        
class LIME:
    def __init__(self, model, train_loader, test_loader, device, mode='regression'):
        """
        Initializes the LIME class with the model and DataLoaders.
        
        Parameters:
        - model: model to be explained.
        - train_loader: DataLoader for the training set.
        - test_loader: DataLoader for the test set.
        - device: device (CPU or GPU) for operations.
        - mode: select whether it is a regression or classification model.
        """
        self.model = model
        self.device = device
        self.mode = mode
        
        # Get a batch from the training DataLoader and convert it to numpy
        self.x_train = next(iter(train_loader))[0].cpu().numpy()
        self.x_test = next(iter(test_loader))[0].cpu().numpy()  # Test data for explanation
        
        # Detect the number of features from x_train
        self.num_features = self.x_train.shape[1]
        
        # Configure LimeTabularExplainer with training data
        self.explainer_lime = lime_tabular.LimeTabularExplainer(
            training_data=self.x_train,
            mode=self.mode,  # Use "classification" if the model is a classifier
            feature_names=[f"Feature {i}" for i in range(self.num_features)],
            discretize_continuous=True,
            verbose=True
        )
        
    def predict_fn(self, data):
        """Prediction function to adapt the PyTorch model for LIME."""
        self.model.eval()
        with torch.no_grad():
            data_tensor = torch.from_numpy(data).float().to(self.device)
            return self.model(data_tensor).cpu().numpy().flatten()

    def local_explanation(self, index, num_features=None):
        """
        Generates a local explanation for a specific instance and displays a DataFrame
        with feature indices and LIME values for the chosen instance.
        
        Parameters:
        - index: index of the instance in the test set to be explained.
        - num_features: number of features to display in the explanation. If None, use all features.
        """
        # Define the number of features for explanation if not specified
        if num_features is None:
            num_features = self.num_features  # Use all features if `num_features` is not provided

        # Select the instance from the test set for explanation
        instance_to_explain = self.x_test[index]

        # Generate explanation with LIME
        exp = self.explainer_lime.explain_instance(
            data_row=instance_to_explain,
            predict_fn=self.predict_fn,
            num_features=num_features
        )
        
        # Extract the explanation as a list of tuples and convert to DataFrame
        explanation_list = exp.as_list()
        lime_df = pd.DataFrame(explanation_list, columns=["Feature", "LIME Value"]).sort_values(by="LIME Value", ascending=False)
        
        return lime_df

def main():
    pass

if __name__ == '__main__':
    main()
