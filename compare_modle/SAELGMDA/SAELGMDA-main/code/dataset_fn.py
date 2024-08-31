
import pandas as pd
import os
import scipy.io as scio



def get_load_fn(dataset_name):
    if dataset_name == "HMDAD":
        load_fn = load_HMDAD()
    elif dataset_name == "Disbiome":
        load_fn = load_Disbiome()
    return load_fn


def load_HMDAD(path="../dataset/HMDAD"):
    '''
    interaction = scio.loadmat(os.path.join(path, "interaction.mat"))
    interaction = interaction['interaction']
    disease_info = pd.read_excel(os.path.join(path, "diseases.xlsx"), names=["idx", "name"], header=None)
    disease_feature = pd.read_csv(os.path.join(path, "disease_features.txt"), sep='\t', header=None)
    microbe_info = pd.read_excel(os.path.join(path, "microbes.xlsx"), names=["idx", "name"], header=None)
    microbe_feature = pd.read_csv(os.path.join(path, "microbe_features.txt"), sep='\t', header=None)
    return interaction, disease_feature.values, microbe_feature.values, disease_info, microbe_info
    '''
    # Load the interaction matrix from a .mat file
    interaction = scio.loadmat(os.path.join(path, "interaction.mat"))
    interaction = interaction['interaction']

    # Load disease information from an Excel file
    disease_info = pd.read_excel(os.path.join(path, "diseases.xlsx"), names=["idx", "name"], header=None,
                                 engine='openpyxl')

    # Load disease features from a text file
    disease_feature = pd.read_csv(os.path.join(path, "disease_features.txt"), sep='\t', header=None)

    # Load microbe information from an Excel file
    microbe_info = pd.read_excel(os.path.join(path, "microbes.xlsx"), names=["idx", "name"], header=None,
                                 engine='openpyxl')

    # Load microbe features from a text file
    microbe_feature = pd.read_csv(os.path.join(path, "microbe_features.txt"), sep='\t', header=None)

    # Return all loaded data
    return interaction, disease_feature.values, microbe_feature.values, disease_info, microbe_info

def load_Disbiome(path="../dataset/Disbiome"):
    interaction = scio.loadmat(os.path.join(path, "interaction.mat"))
    interaction = interaction['interaction1']
    disease_info = pd.read_excel(os.path.join(path, "diseases.xlsx"), names=["idx", "name"], header=None)
    disease_feature = pd.read_csv(os.path.join(path, "disease_features.txt"), sep='\t', header=None)
    microbe_info = pd.read_excel(os.path.join(path, "microbes.xlsx"), names=["idx", "name"], header=None)
    microbe_feature = pd.read_csv(os.path.join(path, "microbe_features.txt"), sep='\t', header=None)
    return interaction, disease_feature.values, microbe_feature.values, disease_info, microbe_info