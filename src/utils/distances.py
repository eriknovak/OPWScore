import math
import torch
import torch.nn as nn
import torch.nn.functional as f

# ===================================================================
# Sinkhorn algorithm and Wasserstein Distance Helper Functions
# ===================================================================


def get_cost_matrix(source: torch.Tensor, target: torch.Tensor):
    """Calculates the cost matrix between the source and target tensors
    Args:
        source: The source tensor.
        target: The target tensor.
    Returns:
        torch.Tensor: The cost matrix between the source and target tensors.
    """
    # normalize the embeddings
    source = f.normalize(source, p=2, dim=2)
    target = f.normalize(target, p=2, dim=2)
    # calculate and return the cost matrix
    cost_matrix = source.matmul(target.transpose(1, 2))
    cost_matrix = torch.ones_like(cost_matrix) - cost_matrix
    return torch.exp(cost_matrix) - 1


def get_weight_dist(attn: torch.FloatTensor):
    """Generates the weight distribution tensor
    Args:
        attn: The attention tensor.
    Returns:
        torch.Tensor: The weight distribution tensor.
    """
    dist = torch.ones_like(attn) * attn
    dist = dist / dist.sum(dim=1).view(-1, 1).repeat(1, attn.shape[1])
    dist.requires_grad = False
    return dist


def prior_distribution(N: int, M: int, std: float = 1):
    """The prior distribution used to fit the Wasserstein distance to a particular pattern
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
        std (float): The standard deviation used to define the prior distribution.
    Returns:
        torch.Tensor: The prior distribution.

    """
    Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
    Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
    Lij = torch.abs(Ni - Mj) / math.sqrt(1 / N**2 + 1 / M**2)
    exp = torch.exp(-torch.pow(Lij, 2) / (2 * std**2))
    return 1 / (std * math.sqrt(2 * math.pi)) * exp


def inverse_difference_moment_matrix(N: int, M: int):
    """The inverse different moment matrix for measuring local homogeneity of the transport matrix
    Args:
        N (int): The number of tokens in the source tensor.
        M (int): The number of tokens in the target tensor.
    Returns:
        torch.Tensor: The inverse different moment matrix.
    """
    Ni = torch.Tensor([[i / N for i in range(1, N + 1)] for _ in range(M)]).T
    Mj = torch.Tensor([[j / M for j in range(1, M + 1)] for _ in range(N)])
    return 1 / ((Ni - Mj).pow(2) + 1)


# ===================================================================
# Sinkhorn Algorithm and Wasserstein Distance
# ===================================================================


def sinkhorn(
    source_dist: torch.FloatTensor,
    target_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg: float,
    nit: int = 20,
):
    """The sinkhorn algorithm adapted for PyTorch from the
        PythonOT library <https://pythonot.github.io/>.
    Args:
        source_dist: The source distribution tensor.
        target_dist: The target distribution tensor.
        cost_matrix: The cost matrix tensor.
        reg: The regularization factor.
        nit: Number of maximum iterations.
    Returns:
        torch.Tensor: The transportation matrix.
    """
    # assert the tensor dimensions
    assert (
        source_dist.shape[0] == cost_matrix.shape[0]
        and source_dist.shape[1] == cost_matrix.shape[1]
    )
    assert (
        target_dist.shape[0] == cost_matrix.shape[0]
        and target_dist.shape[1] == cost_matrix.shape[2]
    )

    # prepare the initial variables
    K = torch.exp(-cost_matrix / reg)
    Kp = (1 / source_dist).reshape(source_dist.shape[0], -1, 1) * K
    # initialize the u and v tensors
    u = torch.ones_like(source_dist)
    v = torch.ones_like(target_dist)

    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
        # calculate the v_{i} tensor
        v = target_dist / KTransposeU
        # calculate the u_{i} tensor
        u = 1.0 / Kp.bmm(v.unsqueeze(2)).squeeze(2)
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)


def get_wasserstein_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
    reg: float,
    nit: int,
):
    """Gets the Wasserstein distances between the source and target embeddings.
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks.
        target_attns: The target attention masks.
        reg: The regularization factor.
        nit: The number of iterations.
    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: The tuple containing
            - the wasserstein distances between the embeddings,
            - the cost matrix, and
            - the transportation matrix.
    """
    # get the cost matrix
    C = get_cost_matrix(source_embed, target_embed)
    # get query and texts distributions
    q_dist = get_weight_dist(source_attns.float())
    d_dist = get_weight_dist(target_attns.float())
    # solve the optimal transport problem
    T = sinkhorn(q_dist, d_dist, C, reg, nit)
    # calculate the distances
    distances = (C * T).view(C.shape[0], -1).sum(dim=1)
    # return the loss, transport and cost matrices
    return distances, C, T


def seq_sinkhorn(
    source_dist: torch.FloatTensor,
    target_dist: torch.FloatTensor,
    cost_matrix: torch.Tensor,
    reg1: float,
    reg2: float,
    nit: int = 20,
):
    """The sequence sinkhorn algorithm adapted for PyTorch from the
        PythonOT library <https://pythonot.github.io/>.
    Args:
        source_dist: The source distribution tensor.
        target_dist: The target distribution tensor.
        cost_matrix: The cost matrix tensor.
        reg: The regularization factor.
        nit: Number of maximum iterations.
    Returns:
        torch.Tensor: The transportation matrix.
    """
    # assert the tensor dimensions
    assert (
        source_dist.shape[0] == cost_matrix.shape[0]
        and source_dist.shape[1] == cost_matrix.shape[1]
    )
    assert (
        target_dist.shape[0] == cost_matrix.shape[0]
        and target_dist.shape[1] == cost_matrix.shape[2]
    )
    # construct the sinkhorn matrix
    _, N, M = cost_matrix.shape
    P_ij = prior_distribution(N, M)
    S_ij = inverse_difference_moment_matrix(N, M)
    K = P_ij * torch.exp(1 / reg2 * (reg1 * S_ij - cost_matrix))
    Kp = (1 / source_dist).reshape(source_dist.shape[0], -1, 1) * K

    # initialize the u and v tensors
    u = torch.ones_like(source_dist)
    v = torch.ones_like(target_dist)

    istep = 0
    while istep < nit:
        # calculate K.T * u for each example in batch
        KTransposeU = K.transpose(1, 2).bmm(u.unsqueeze(2)).squeeze(2)
        # calculate the v_{i} tensor
        v = target_dist / KTransposeU
        # calculate the u_{i} tensor
        u = 1.0 / Kp.bmm(v.unsqueeze(2)).squeeze(2)
        # go to next step
        istep = istep + 1
    # calculate the transport matrix
    U = torch.diag_embed(u)
    V = torch.diag_embed(v)
    return U.bmm(K).bmm(V)


def get_seq_wasserstein_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
    reg1: float,
    reg2: float,
    nit: int,
):
    """Gets the sequence Wasserstein distances between the source and target embeddings.
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks.
        target_attns: The target attention masks.
        reg1: The first regularization factor associated with the inverse difference
            moment.
        reg2: The second regularization factor associated with the Kullback-Leibler
            Divergence between the transport matrix and the prior (diagonal)
            distribution.
        nit: The number of iterations.
    Returns:
        torch.Tensor, torch.Tensor, torch.Tensor: The tuple containing
            - the wasserstein distances between the embeddings,
            - the cost matrix, and
            - the transportation matrix.
    """
    # get the cost matrix
    C = get_cost_matrix(source_embed, target_embed)
    # get query and texts distributions
    q_dist = get_weight_dist(source_attns.float())
    d_dist = get_weight_dist(target_attns.float())
    # solve the optimal transport problem
    T = seq_sinkhorn(q_dist, d_dist, C, reg1, reg2, nit)
    # calculate the distances
    distances = (C * T).view(C.shape[0], -1).sum(dim=1)
    # return the loss, transport and cost matrices
    return distances, C, T


# ===================================================================
# Support for Cosine Distance-Based Similarity Measures
# ===================================================================


def prep_relevant_embeds(embed: torch.Tensor, attns: torch.Tensor):
    """Prepares the token embeddings for mean and max types
    Args:
        embed: The token embeddings.
        attns: The attention mask.
    Returns:
        torch.Tensor: The tensor with padding embeddings set to zero.
    """
    return embed * attns.unsqueeze(2).repeat(1, 1, embed.shape[2])


def __get_cosine_dist(source_embed: torch.Tensor, target_embed: torch.Tensor):
    """Calculate the cosine distance between the source and target embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # normalize the vectors before calculating
    source_embed = f.normalize(source_embed, p=2, dim=1)
    target_embed = f.normalize(target_embed, p=2, dim=1)

    # calculate the mean distances
    distances = source_embed.matmul(target_embed.T).diagonal()
    distances = torch.ones_like(distances) - distances
    return distances, None, None


def get_cls_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Gets the cosine distance using the first [CLS] token
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the first [CLS] token
    source_embed = source_embed[:, 0, :]
    target_embed = target_embed[:, 0, :]
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)


def get_max_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Get the cosine distance using the maximum values of the embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the maximum values of the embeddings
    source_embed, _ = prep_relevant_embeds(source_embed, source_attns).max(dim=1)
    target_embed, _ = prep_relevant_embeds(target_embed, target_attns).max(dim=1)
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)


def get_mean_dist(
    source_embed: torch.Tensor,
    target_embed: torch.Tensor,
    source_attns: torch.Tensor,
    target_attns: torch.Tensor,
):
    """Get the cosine distance using the mean values of the embeddings
    Args:
        source_embed: The source embeddings.
        target_embed: The target embeddings.
        source_attns: The source attention masks. Not used by this method; it is
            here just for having the consistent input schema.
        target_attns: The target attention masks. Not used by this method; it is
            here just for having the consistent input schema.
    Returns:
        torch.Tensor, None, None: The tuple containing the distances between
            the embeddings and two None values.
    """

    # get the maximum values of the embeddings
    source_embed = prep_relevant_embeds(source_embed, source_attns).mean(dim=1)
    target_embed = prep_relevant_embeds(target_embed, target_attns).mean(dim=1)
    # return the cosine distance
    return __get_cosine_dist(source_embed, target_embed)
