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
            # print(properties)
            for num_line, line in enumerate(f):
                # print(num_line, line)
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
                if molecule_data['natoms'] in list_mols or len(list_mols)==0:
                    coords.append([molecule_data['atoms'], molecule_data['coordinates']])
                    prop.append(molecule_data['properties'])
                    natoms.append(molecule_data['natoms'])
            i += 1
        return coords, prop, natoms
    
    def dataset_to_numpy(dataset):  
        all_data = []
        all_targets = []

        for data, target in dataset:
            all_data.append(data.numpy())
            all_targets.append(target.numpy())
            
        data_numpy = np.array(all_data)
        targets_numpy = np.array(all_targets)
        
        return data_numpy, targets_numpy


    
    def get_smiles(self, file_path):
        """Get the one SMILES representation of a molecule."""
        
        smiles = []
        
        with open(file_path, 'r') as f:
            # readn number of atoms
            natoms = int(f.readline())
            # skip the second line
            f.readline()
            for i in range(natoms+1): # skips to the smiles
                f.readline()
            smiles_tuple = tuple(f.readline().strip().split('\t'))
            smiles.append(smiles_tuple)

        return smiles
        
    def load_smiles(self):
        """Load all SMILES representation of the QM9 dataset."""    
    
        list_smiles = []
        i = 0
        for file_name in os.listdir(self.directory_path):
            if i == 100:
                break
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.directory_path, file_name)
                list_smiles.append(self.get_smiles(file_path))
            i+=1
        
        return list_smiles    
    
    def df_props(self):
        """Create a DataFrame with the properties of the QM9 dataset."""
        
        coords, props, natoms = self.load_qm9_dataset()
        
        # Criar o DataFrame com as propriedades
        df = pd.DataFrame(props)
        
        # Redefinir os índices do DataFrame
        df.reset_index(drop=True, inplace=True)
        
        # Renomear as colunas do DataFrame
        df.columns = self.properties
        
        return df  
  
class Shap:
    def __init__(self, model, train_loader, test_loader, device):
        """
        Inicializa a classe Shap com o modelo, dados de treinamento e teste.
        
        Parâmetros:
        - model: modelo a ser explicado.
        - train_loader: DataLoader de treinamento para obter o background.
        - test_loader: DataLoader de teste para explicações.
        - device: dispositivo (CPU ou GPU) para rodar as operações.
        """
        self.model = model
        self.device = device

        # Seleciona um batch do DataLoader de treinamento como referência (background)
        background = next(iter(train_loader))[0].cpu().numpy()  # Convertendo para numpy para o KernelExplainer
        print("Background shape:", background.shape)  # Verificar a forma do background
        
        # Converte o modelo PyTorch em uma função de previsão compatível com KernelExplainer
        def predict_fn(data):
            self.model.eval()
            with torch.no_grad():
                data_tensor = torch.from_numpy(data).float().to(self.device)
                return self.model(data_tensor).cpu().numpy()
        
        # Inicializa o KernelExplainer com o modelo e o background
        self.explainer = shap.KernelExplainer(predict_fn, background)
        
        # Carrega os dados de teste para explicação
        self.test_data, _ = next(iter(test_loader))
        self.test_data = self.test_data.cpu().numpy()  # Convertendo para numpy para o KernelExplainer
        print("Test data shape:", self.test_data.shape)  # Verificar a forma do test_data
        
        # Calcula os shap_values para os dados de teste
        self.shap_values = self.explainer.shap_values(self.test_data)

    def local_explanation(self, index):
        """
        Gera uma explicação local para uma instância específica e exibe um DataFrame
        com o índice das features e seus valores SHAP para a instância escolhida.
        
        Parâmetros:
        - index: índice da instância no conjunto de teste para a qual se deseja a explicação.
        """
        # Seleciona os valores SHAP da instância e remove a dimensão extra
        local_shap_values = self.shap_values[index].flatten()  # Achatar para remover a segunda dimensão
        
        # Verificação adicional da forma dos valores SHAP
        print("Local SHAP values shape:", local_shap_values.shape)
        
        # Cria um DataFrame com o índice das features e seus valores SHAP
        feature_importance = pd.DataFrame({
            'Feature Index': range(len(local_shap_values)),
            'SHAP Value': local_shap_values
        }).sort_values(by='SHAP Value', ascending=False).reset_index(drop=True)
        
        return feature_importance

    def global_explanation(self):
        """
        Gera uma explicação global exibindo um DataFrame com a importância média das features,
        calculada como a média dos valores absolutos dos SHAP values para todas as instâncias.
        """
        # Calcula a importância global como a média dos valores absolutos das contribuições
        global_importance = np.mean(np.abs(self.shap_values), axis=0).flatten()  # Achatar para remover a segunda dimensão
        
        # Verificação adicional da forma dos valores globais
        print("Global SHAP values shape:", global_importance.shape)
        
        # Cria um DataFrame para armazenar as importâncias globais com o índice das features
        feature_importance = pd.DataFrame({
            'Feature Index': range(len(global_importance)),
            'Importance': global_importance
        }).sort_values(by='Importance', ascending=False).reset_index(drop=True)
        
        return feature_importance
        
class LIME:
    def __init__(self, model, data, x_train, x_test, y_test):    
        self.explainer_lime = lime_tabular.LimeTabularExplainer(
            x_train, feature_names=data.feature_names, 
            class_names=data.target_names, discretize_continuous=True)
        
    def local_explanation(self, index):
        exp = self.explainer_lime.explain_instance(self.x[index], self.model.predict)
        exp.show_in_notebook(show_table=True, show_all=False)
    
    def global_explanation(self):
        pass

def main():
    pass

if __name__ == '__main__':
    main()