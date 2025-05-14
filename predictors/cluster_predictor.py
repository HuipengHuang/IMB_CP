import numpy as np
from sklearn.cluster import KMeans
from .utils import get_quantile_threshold, get_clustering_parameters
from scores.utils import get_score
import torch
import math
from dataset.utils import split_dataloader
from torch.utils.data import DataLoader

class ClusterPredictor:
    def __init__(self, args, net):
        self.score_function = get_score(args)
        self.net = net
        self.alpha = args.alpha
        self.device = next(net.parameters()).device
        self.args = args
        self.k = int(args.k) if args.k is not None else None
        self.num_classes = args.num_classes
        self.class_threshold = None
        self.class2cluster = None
        self.cluster2class = None
        self.split = args.split if args.split else "proportional"
        self.cluster_frac = 0.3

    def calibrate(self, cal_loader, alpha=None):
        """Input calibration dataloader.
        Compute scores for all the calibration data and take the (1 - alpha) quantile.
        If args.null_qhat == 'standard', we compute the qhat for standard conformal and use that as the default value
        """

        self.net.eval()
        with torch.no_grad():
            if alpha is None:
                alpha = self.alpha
            cluster_assignment, cal_scores, cal_targets = self.cluster_class(cal_loader, alpha)

            self.class_threshold = self.__compute_cluster_specific_qhats(cluster_assignment,cal_scores, cal_targets, alpha)


    def __get_rare_classes(self, labels, alpha, num_classes):
        """
        Choose classes whose number is less than or equal to the threshold.

        Args:
            labels (torch.Tensor): The ground truth labels.
            alpha (torch.Tensor): The significance level.
            num_classes (int): The number of classes.

        Returns:
            torch.Tensor: The rare classes.
        """
        thresh = self.__get_quantile_minimum(alpha)
        classes, cts = torch.unique(labels, return_counts=True)
        rare_classes = classes[cts < thresh].to(self.device)

        # Also included any classes that are so rare that we have 0 labels for it

        all_classes = torch.arange(num_classes, device=self.device)
        zero_ct_classes = all_classes[(all_classes.view(1, -1) != classes.view(-1, 1)).all(dim=0)]
        rare_classes = torch.concatenate((rare_classes, zero_ct_classes))

        return rare_classes

    def __get_quantile_minimum(self, alpha):
        """
        Compute smallest n such that ceil((n+1)*(1-alpha)/n) <= 1

        Args:
            alpha (torch.Tensor): The significance level.

        Returns:
            torch.Tensor: The smallest n.
        """
        n = torch.tensor(0, device=self.device)
        while torch.ceil((n + 1) * (1 - alpha) / n) > 1:
            n += 1
        return n

    def cluster_class(self, cal_loader, alpha):
        n_threshold = get_quantile_threshold(alpha if alpha < 0.1 else 0.1)

        with torch.no_grad():
            all_targets = []
            all_scores = []

            for data, target in cal_loader:
                data, target = data.to(self.device, non_blocking=True), target.to(self.device, non_blocking=True)
                logits = self.net(data)
                prob = torch.softmax(logits, dim=1)
                target_score = self.score_function.compute_target_score(prob, target)
                all_targets.append(target)
                all_scores.append(target_score)

            all_targets = torch.cat(all_targets)
            all_scores = torch.cat(all_scores)

            num_per_class = torch.tensor([(all_targets == i).sum() for i in range(self.num_classes)])
            n_min = min(num_per_class)
            n_min = max(n_min, n_threshold)
            num_remaining_classes = torch.sum(num_per_class >= n_min)

            if self.k is None:
                n_clustering, self.k = get_clustering_parameters(num_remaining_classes, n_min)
                cluster_frac = n_clustering / n_min
                self.cluster_frac = cluster_frac
                cluster_scores, cluster_targets, cal_scores, cal_targets = self.split_data(all_scores, all_targets, num_per_class)
            else:
                cluster_scores, cluster_targets, cal_scores, cal_targets = self.split_data(all_scores, all_targets, num_per_class)

            rare_classes = self.__get_rare_classes(cluster_targets, alpha, self.num_classes)

            if (self.num_classes - len(rare_classes) > self.k) and (self.k > 1):
                # Filter out rare classes and re-index
                remaining_idx, filtered_labels, class_remapping = self.__remap_classes(cluster_targets, rare_classes)
                filtered_scores = cluster_scores[remaining_idx]

                # Compute embedding for each class and get class counts
                embeddings, class_cts = self.__embed_all_classes(filtered_scores, filtered_labels)
                kmeans = KMeans(n_clusters=int(self.k), n_init=10, random_state=2023).fit(
                    X=embeddings.detach().cpu().numpy(),
                    sample_weight=np.sqrt(
                        class_cts.detach().cpu().numpy()),
                )
                nonrare_class_cluster_assignments = torch.tensor(kmeans.labels_, device=self.device)

                cluster_assignments = - torch.ones((self.num_classes,), dtype=torch.int32, device=self.device)

                for cls, remapped_cls in class_remapping.items():
                    cluster_assignments[cls] = nonrare_class_cluster_assignments[remapped_cls]
            else:
                cluster_assignments = - torch.ones((self.num_classes,), dtype=torch.int32, device=self.device)
            self.cluster_assignments = cluster_assignments

            return cluster_assignments, cal_scores, cal_targets

    def __embed_all_classes(self, scores_all, labels, q=[0.5, 0.6, 0.7, 0.8, 0.9]):
        """
        Embed all classes based on the quantiles of their scores.

        Args:
            scores_all (torch.Tensor): A num_instances-length array where scores_all[i] is the score of the true class for instance i.
            labels (torch.Tensor): A num_instances-length array of true class labels.
            q (list, optional): Quantiles to include in embedding. Default is [0.5, 0.6, 0.7, 0.8, 0.9].

        Returns:
            tuple:
                - embeddings (torch.Tensor): A num_classes x len(q) array where the ith row is the embeddings of class i.
                - cts (torch.Tensor): A num_classes-length array where cts[i] is the number of times class i appears in labels.
        """
        num_classes = len(torch.unique(labels))
        embeddings = torch.zeros((num_classes, len(q)), device=self.device)
        cts = torch.zeros((num_classes,), device=self.device)

        for i in range(num_classes):
            if len(scores_all.shape) > 1:
                raise ValueError

            class_i_scores = scores_all[labels == i]

            cts[i] = class_i_scores.shape[0]
            for k in range(len(q)):
                # Computes the q-quantiles of samples and returns the vector of quantiles
                embeddings[i, k] = torch.kthvalue(class_i_scores, int(math.ceil(cts[i] * q[k])), dim=0).values.to(
                    self.device)

        return embeddings, cts


    def __remap_classes(self, labels, rare_classes):
        """
        Exclude classes in rare_classes and remap remaining classes to be 0-indexed.

        Args:
            labels (torch.Tensor): The ground truth labels.
            rare_classes (torch.Tensor): The rare classes.

        Returns:
            tuple: A tuple containing remaining_idx, remapped_labels, and remapping.
        """
        labels = labels.detach().cpu().numpy()
        rare_classes = rare_classes.detach().cpu().numpy()
        remaining_idx = ~np.isin(labels, rare_classes)

        remaining_labels = labels[remaining_idx]
        remapped_labels = np.zeros(remaining_labels.shape, dtype=int)
        new_idx = 0
        remapping = {}
        for i in range(len(remaining_labels)):
            if remaining_labels[i] in remapping:
                remapped_labels[i] = remapping[remaining_labels[i]]
            else:
                remapped_labels[i] = new_idx
                remapping[remaining_labels[i]] = new_idx
                new_idx += 1

        return torch.from_numpy(remaining_idx).to(self.device), torch.tensor(remapped_labels,
                                                                              device=self.device), remapping


    def split_data(self, all_scores, all_targets, num_per_class):
        if self.split == 'proportional':
            # Split dataset along with fraction "frac_clustering"
            num_classes = num_per_class.shape[0]
            n_k = torch.tensor([int(self.cluster_frac * num_per_class[k]) for k in range(num_classes)],
                               device=self.device, dtype=torch.int32)
            idx1 = torch.zeros(all_targets.shape, dtype=torch.bool, device=self.device)
            for k in range(num_classes):
                # Randomly select n instances of class k
                idx = torch.argwhere(all_targets == k).flatten()
                random_indices = torch.randint(0, num_per_class[k], (n_k[k],), device=self.device)
                selected_idx = idx[random_indices]
                idx1[selected_idx] = 1
            clustering_scores = all_scores[idx1]
            clustering_labels = all_targets[idx1]
            cal_scores = all_scores[~idx1]
            cal_labels = all_targets[~idx1]

        elif self.split == 'doubledip':
            clustering_scores, clustering_labels = all_scores, all_targets
            cal_scores, cal_labels = all_scores, all_targets
            idx1 = torch.ones((all_scores.shape[0])).bool()

        elif self.split == 'random':
            # Each point is assigned to clustering set w.p. frac_clustering
            idx1 = torch.rand(size=(len(all_targets),), device=self.device) < self.cluster_frac

            clustering_scores = all_scores[idx1]
            clustering_labels = all_targets[idx1]
            cal_scores = all_scores[~idx1]
            cal_labels = all_targets[~idx1]

        return clustering_scores, clustering_labels, cal_scores, cal_labels

    def calibrate_batch_logit(self, logits, target, alpha):
        """Design for conformal training, which needs to compute threshold in every batch"""
        prob = torch.softmax(logits, dim=-1)
        batch_score = self.score_function.compute_target_score(prob, target)
        N = target.shape[0]
        return torch.quantile(batch_score, math.ceil((1 - alpha) * (N + 1)) / N, dim=0)

    def evaluate(self, test_loader):
        """Must be called after calibration.
        Output a dictionary containing Top1 Accuracy, Coverage and Average Prediction Set Size."""
        if self.class_threshold is None:
            raise ValueError("Threshold score is None. Please do calibration first.")
        self.net.eval()
        with torch.no_grad():
            total_accuracy = 0
            total_coverage = 0
            total_prediction_set_size = 0
            class_coverage = [0 for _ in range(self.num_classes)]
            class_size = [0 for _ in range(self.num_classes)]
            total_samples = 0

            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                batch_size = target.shape[0]
                total_samples += batch_size

                logits = self.net(data)
                prob = torch.softmax(logits, dim=-1)
                prediction = torch.argmax(prob, dim=-1)
                total_accuracy += (prediction == target).sum().item()

                batch_score = self.score_function(prob)
                prediction_set = (batch_score <= self.class_threshold).to(torch.int)

                target_prediction_set = prediction_set[torch.arange(batch_size), target]
                total_coverage += target_prediction_set.sum().item()
                total_prediction_set_size += prediction_set.sum().item()

                for i in range(prediction_set.shape[0]):
                    class_coverage[target[i]] += 1
                    class_size[target[i]] += 1


            accuracy = total_accuracy / total_samples
            coverage = total_coverage / total_samples
            avg_set_size = total_prediction_set_size / total_samples
            class_coverage_gap = np.array(class_coverage) / np.array(class_size)
            class_coverage_gap = np.sum(np.abs(class_coverage_gap - (1 - self.alpha))) / 100
            result_dict = {
                f"{self.args.score}_Top1Accuracy": accuracy,
                f"{self.args.score}_AverageSetSize": avg_set_size,
                f"{self.args.score}_Coverage": coverage,
                f"{self.args.score}_class_coverage_gap": class_coverage_gap,
            }
            return result_dict

    def __compute_cluster_specific_qhats(self, cluster_assignments, cal_class_scores, cal_true_labels, alpha):
        """
        Compute cluster-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha).

        Args:
            cluster_assignments (torch.Tensor): A num_classes-length array where entry i is the index of the cluster that class i belongs to. Rare classes can be assigned to cluster -1 and they will automatically be given as default_qhat.
            cal_class_scores (torch.Tensor): The scores for each instance in the calibration set.
            cal_true_labels (torch.Tensor): The true class labels for instances in the calibration set.
            alpha (float): Desired coverage level.

        Returns:
            torch.Tensor: A num_classes-length array where entry i is the quantile corresponding to the cluster that class i belongs to.
        """

        # Map true class labels to clusters
        cal_true_clusters = torch.tensor([cluster_assignments[label] for label in cal_true_labels], device=self.device)
        num_clusters = torch.max(cluster_assignments) + 1

        cluster_qhats = self.__compute_class_specific_qhats(cal_class_scores, cal_true_clusters, num_clusters, alpha)
        # Map cluster qhats back to classes
        num_classes = len(cluster_assignments)
        qhats_class = torch.tensor([cluster_qhats[cluster_assignments[k]] for k in range(num_classes)],
                                   device=self.device)

        return qhats_class

    def __compute_class_specific_qhats(self, cal_class_scores, cal_true_clusters, num_clusters, alpha):
        """
        Compute class-specific quantiles (one for each class) that will result in marginal coverage of (1-alpha).

        Args:
            cal_class_scores (torch.Tensor): A num_instances-length array where cal_class_scores[i] is the score for instance i.
            cal_true_clusters (torch.Tensor): A num_instances-length array of true class labels. If class -1 appears, it will be assigned the null_qhat value. It is appended as an extra entry of the returned q_hats so that q_hats[-1] = null_qhat.
            num_clusters (int): The number of clusters.
            alpha (float): Desired coverage level.

        Returns:
            torch.Tensor: The threshold of each class.
        """

        # Compute quantile q_hat that will result in marginal coverage of (1-alpha)
        # null_qhat = self._calculate_conformal_value(cal_class_scores, alpha)
        null_qhat = torch.inf

        q_hats = torch.zeros((num_clusters,), device=self.device)  # q_hats[i] = quantile for class i
        for k in range(num_clusters):
            # Only select data for which k is true class
            idx = (cal_true_clusters == k)
            scores = cal_class_scores[idx]
            N = scores.shape[0]
            quantile_value = math.ceil((N + 1) * (1 - alpha)) / N
            q_hats[k] = torch.kthvalue(scores, math.ceil(N*quantile_value), dim=0).values.to(scores.device)
        # print(torch.argwhere(cal_true_clusters==-1))
        if -1 in cal_true_clusters:
            q_hats = torch.concatenate((q_hats, torch.tensor([null_qhat], device=self.device)))

        return q_hats

