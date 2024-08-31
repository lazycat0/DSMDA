import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x


class MicrobeDiseaseModel(nn.Module):
    def __init__(self, disease_dim, microbe_dim, hidden_dim, output_dim):
        super(MicrobeDiseaseModel, self).__init__()
        self.disease_encoder = Encoder(disease_dim, hidden_dim, output_dim)
        self.microbe_encoder = Encoder(microbe_dim, hidden_dim, output_dim)

    def forward(self, disease_features, microbe_features):
        # 对每一行的特征进行编码
        encoded_disease_features = self.disease_encoder(disease_features)
        encoded_microbe_features = self.microbe_encoder(microbe_features)

        # 计算编码后的特征矩阵行之间的点积
        # 对于每一对疾病和微生物，计算它们的点积作为预测概率
        pred_probs = torch.sum(encoded_disease_features * encoded_microbe_features, dim=1)

        # 对点积结果进行归一化，这里使用 Sigmoid 函数
        pred_probs_normalized = torch.sigmoid(pred_probs)

        return pred_probs_normalized



class MicroDiseaseModel(nn.Module):
    def __init__(self, mic_input_dim, dis_input_dim, latent_dim, tau=1.0):
        super(MicroDiseaseModel, self).__init__()
        self.tau = tau
        # 微生物特征的变分编码器，使用三层网络
        self.mic_encoder = nn.Sequential(
            nn.Linear(mic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )
        # 疾病特征的变分编码器，使用三层网络
        self.dis_encoder = nn.Sequential(
            nn.Linear(dis_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, encoder):
        x = encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        return self.reparameterize(mu, log_var)

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def forward(self, mic_feature_tenor, dis_feature_tensor, train_samples_tensor):
        loss = 0.0

        mic_latent_tensor=self.encode(mic_feature_tenor, self.mic_encoder)
        dis_latent_tensor=self.encode(dis_feature_tensor, self.dis_encoder)
        loss_weight = self.sim(mic_latent_tensor,dis_latent_tensor)
        for sample in train_samples_tensor:
            '''
            i, j, i_hat, j_hat = map(int, sample)
            mic_i = mic_latent_tensor[i].unsqueeze(0)
            dis_j = dis_latent_tensor[j].unsqueeze(0)
            mic_i_hat = mic_latent_tensor[i_hat].unsqueeze(0)
            dis_j_hat = dis_latent_tensor[j_hat].unsqueeze(0)

            # 计算正相关对和负相关对的相似度
            pos_sim = torch.exp(self.sim(mic_i, dis_j) / self.tau)
            neg_sim_i_j_hat = torch.exp(self.sim(mic_i, dis_j_hat) / self.tau)
            neg_sim_i_hat_j = torch.exp(self.sim(mic_i_hat, dis_j) / self.tau)

            # 计算当前样本的损失
            sample_loss = (pos_sim / (pos_sim + neg_sim_i_j_hat + neg_sim_i_hat_j))
            loss += sample_loss
            '''
            i, j, i_hat, j_hat = map(int, sample)
            # 计算正相关对和负相关对的相似度
            pos_sim = torch.exp(loss_weight[i , j] / self.tau)
            neg_sim_i_j_hat = torch.exp(loss_weight[i , j_hat] / self.tau)
            neg_sim_i_hat_j = torch.exp(loss_weight[i , j_hat] / self.tau)
            neg_sim = neg_sim_i_hat_j + neg_sim_i_j_hat
            # 计算当前样本的损失
            sample_loss2 = torch.abs((loss_weight[i , j]+1)/2-1)+(loss_weight[i , j_hat]+1)/2+(loss_weight[i , j_hat]+1)/2
            sample_loss1 = -torch.log((2*pos_sim) / (2*pos_sim + neg_sim))
            loss += sample_loss1+sample_loss2



        return loss / len(train_samples_tensor)  ,mic_latent_tensor,  dis_latent_tensor # 返回平均损失


class MicroDiseaseModel_v2(nn.Module):
    def __init__(self, mic_input_dim, dis_input_dim, latent_dim, tau=1.0):
        super(MicroDiseaseModel_v2, self).__init__()
        self.tau = tau
        # 微生物特征的变分编码器，使用三层网络
        self.mic_encoder = nn.Sequential(
            nn.Linear(mic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )
        # 疾病特征的变分编码器，使用三层网络
        self.dis_encoder = nn.Sequential(
            nn.Linear(dis_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, encoder):
        x = encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        return self.reparameterize(mu, log_var)

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def forward(self, mic_feature_tenor, dis_feature_tensor):

        mic_latent_tensor=self.encode(mic_feature_tenor, self.mic_encoder)
        dis_latent_tensor=self.encode(dis_feature_tensor, self.dis_encoder)
        similarity_matrix = torch.mm(mic_latent_tensor, dis_latent_tensor.T)
        normalized_similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        return normalized_similarity_matrix



class MicroDiseaseModel_v3(nn.Module):
    def __init__(self, mic_input_dim, dis_input_dim, latent_dim, tau=1):
        super(MicroDiseaseModel_v3, self).__init__()
        self.tau = tau
        # 微生物特征的变分编码器，使用三层网络
        self.mic_encoder = nn.Sequential(
            nn.Linear(mic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim )  # 输出均值和方差
        )
        # 疾病特征的变分编码器，使用三层网络
        self.dis_encoder = nn.Sequential(
            nn.Linear(dis_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim )  # 输出均值和方差
        )
        self.smooth1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.smooth2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.smooth3 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        # self._init_weights(self.dis_encoder)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, encoder):
        x = encoder(x)
        # mu, log_var = torch.chunk(x, 2, dim=-1)
        # # add_z =self.reparameterize(mu, log_var)
        # # z = mu + add_z
        # z=self.reparameterize(mu, log_var)
        z1=x+self.smooth1(x)
        z2=z1+self.smooth2(z1)
        z3=z2+self.smooth3(z2)
        # z_out = (z3 - z3.min()) / (z3.max() - z3.min())
        return z3

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def mm_one(self, z1, z2):
        return torch.mm(z1, z2.t())

    def forward(self, mic_feature_tenor, dis_feature_tensor, mic_i_feature, mic_i_hat_feature, disease_j_feature,
                disease_j_hat_feature):
        mic_latent_tensor = self.encode(mic_feature_tenor, self.mic_encoder)
        dis_latent_tensor = self.encode(dis_feature_tensor, self.dis_encoder)

        mic_i_latent_tensor = self.encode(mic_i_feature, self.mic_encoder)
        mic_i_hat_latent_tensor = self.encode(mic_i_hat_feature, self.mic_encoder)
        dis_j_latent_tensor = self.encode(disease_j_feature, self.dis_encoder)
        dis_j_hat_latent_tensor = self.encode(disease_j_hat_feature, self.dis_encoder)

        f = lambda x: torch.exp(x / self.tau)
        i_j_sim = f(self.sim(mic_i_latent_tensor, dis_j_latent_tensor))
        i_j_hat_sim = f(self.sim(mic_i_latent_tensor, dis_j_hat_latent_tensor))
        i_hat_j_sim = f(self.sim(mic_i_hat_latent_tensor, dis_j_latent_tensor))

        similarity_matrix = torch.mm(mic_latent_tensor, dis_latent_tensor.T)
        # similarity_matrix =self.sim(mic_latent_tensor, dis_latent_tensor)
        normalized_similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        return normalized_similarity_matrix, -(torch.log(i_j_sim.diag() / (i_j_sim.diag() + i_j_hat_sim.diag() + i_hat_j_sim.diag()))).mean()



class MicroDiseaseModel_v3_copy(nn.Module):
    def __init__(self, mic_input_dim, dis_input_dim, latent_dim, tau=1):
        super(MicroDiseaseModel_v3, self).__init__()
        self.tau = tau
        # 微生物特征的变分编码器，使用三层网络
        self.mic_encoder = nn.Sequential(
            nn.Linear(mic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )
        # 疾病特征的变分编码器，使用三层网络
        self.dis_encoder = nn.Sequential(
            nn.Linear(dis_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim * 2)  # 输出均值和方差
        )
        #self._init_weights(self.dis_encoder)


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x, encoder):
        x = encoder(x)
        mu, log_var = torch.chunk(x, 2, dim=-1)
        #z=self.reparameterize(mu, log_var)
        #z = (z - z.min()) / (z.max() - z.min())
        return self.reparameterize(mu, log_var)

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def mm_one(self, z1, z2):
        return torch.mm(z1, z2.t())

    def forward(self, mic_feature_tenor, dis_feature_tensor,mic_i_feature, mic_i_hat_feature, disease_j_feature , disease_j_hat_feature):

        mic_latent_tensor=self.encode(mic_feature_tenor, self.mic_encoder)
        dis_latent_tensor=self.encode(dis_feature_tensor, self.dis_encoder)

        mic_i_latent_tensor = self.encode(mic_i_feature, self.mic_encoder)
        mic_i_hat_latent_tensor = self.encode(mic_i_hat_feature, self.mic_encoder)
        dis_j_latent_tensor = self.encode(disease_j_feature, self.dis_encoder)
        dis_j_hat_latent_tensor = self.encode(disease_j_hat_feature, self.dis_encoder)

        f = lambda x: torch.exp(x / self.tau)
        i_j_sim = f(self.sim(mic_i_latent_tensor, dis_j_latent_tensor))
        i_j_hat_sim = f(self.sim(mic_i_latent_tensor, dis_j_hat_latent_tensor))
        i_hat_j_sim = f(self.sim(mic_i_hat_latent_tensor, dis_j_latent_tensor))

        similarity_matrix = torch.mm(mic_latent_tensor, dis_latent_tensor.T)
        #similarity_matrix =f(self.sim(mic_latent_tensor, dis_latent_tensor))
        normalized_similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (similarity_matrix.max() - similarity_matrix.min())
        return normalized_similarity_matrix, -(torch.log(i_j_sim.diag()/ (i_j_sim.diag() + i_j_hat_sim.diag() + i_hat_j_sim.diag()))).mean()



class MicroDiseaseModel_v3_No_VAE(nn.Module):
    def __init__(self, mic_input_dim, dis_input_dim, latent_dim, tau=1):
        super(MicroDiseaseModel_v3_No_VAE, self).__init__()
        self.tau = tau
        # 微生物特征的变分编码器，使用三层网络
        self.mic_encoder = nn.Sequential(
            nn.Linear(mic_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim )  # 输出均值和方差
        )
        # 疾病特征的变分编码器，使用三层网络
        self.dis_encoder = nn.Sequential(
            nn.Linear(dis_input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim )  # 输出均值和方差
        )

        self.smooth1 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.smooth2 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        self.smooth3 = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )



    def encode(self, x, encoder):
        x = encoder(x)
        #mu, log_var = torch.chunk(x, 2, dim=-1)
        z1 = self.smooth1(x)
        z2 = self.smooth2(z1)
        z3 = self.smooth3(z2)

        return z3

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def forward(self, mic_feature_tenor, dis_feature_tensor,mic_i_feature, mic_i_hat_feature, disease_j_feature , disease_j_hat_feature):
        mic_latent_tensor = self.encode(mic_feature_tenor, self.mic_encoder)
        dis_latent_tensor = self.encode(dis_feature_tensor, self.dis_encoder)

        mic_i_latent_tensor = self.encode(mic_i_feature, self.mic_encoder)
        mic_i_hat_latent_tensor = self.encode(mic_i_hat_feature, self.mic_encoder)
        dis_j_latent_tensor = self.encode(disease_j_feature, self.dis_encoder)
        dis_j_hat_latent_tensor = self.encode(disease_j_hat_feature, self.dis_encoder)
        f = lambda x: torch.exp(x / self.tau)
        i_j_sim = f(self.sim(mic_i_latent_tensor, dis_j_latent_tensor))
        i_j_hat_sim = f(self.sim(mic_i_latent_tensor, dis_j_hat_latent_tensor))
        i_hat_j_sim = f(self.sim(mic_i_hat_latent_tensor, dis_j_latent_tensor))
        similarity_matrix = torch.mm(mic_latent_tensor, dis_latent_tensor.T)
        normalized_similarity_matrix = (similarity_matrix - similarity_matrix.min()) / (
                    similarity_matrix.max() - similarity_matrix.min())
        return normalized_similarity_matrix, (-torch.log(i_j_sim.diag() / (i_j_sim.diag() + i_j_hat_sim.diag() + i_hat_j_sim.diag()))).mean()
