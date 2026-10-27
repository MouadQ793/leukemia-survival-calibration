"""
Ranking from pairwise comparisons with a Minimum Weight Feedback Arc Set (MWFAS).

The pairwise model gives, for every pair of patients (i, j), a probability that
patient i dies before patient j. These probabilities define a weighted directed
graph; removing a minimum-weight set of edges that breaks every cycle yields a
DAG whose topological order is the final risk ranking.

Also contains helpers for upset losses, synthetic ERO graphs and Kendall tau.
"""
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import torch
from scipy.stats import rankdata


def initialize_graph(edges):
    """
    Converts edge list to an adjacency list and weights dictionary.
    Handles parallel edges by including edge_id as part of the structure.

    Args:
        edges (list): List of edges as (start, end, weight, edge_id).

    Returns:
        graph (dict): Adjacency list with (neighbor, edge_id) pairs.
        weights (dict): Dictionary mapping (start, end, edge_id) to weights.
    """
    graph = defaultdict(list)
    weights = {}
    for u, v, w, edge_id in edges:
        graph[u].append((v, edge_id))
        weights[(u, v, edge_id)] = w
    return graph, weights


def find_cycles_and_reduce(graph, weights, n):
    """
    Phase 1: Find cycles and reduce weights using a copy.
    Handles graphs with parallel edges by considering edge identifiers.

    Args:
        graph (dict): Adjacency list with (neighbor, edge_id) pairs.
        weights (dict): Dictionary mapping (u, v, edge_id) to weights.
        n (int): Number of vertices in the graph.

    Returns:
        removed_edges (set): Set of removed edges as (u, v, edge_id).
        removed_weights (dict): Dictionary of removed edges with their original weights.
    """
    weights_copy = weights.copy()  # Work with a copy of weights
    removed_edges = set()
    removed_weights = {}

    while True:
        cycle = find_cycle(graph, n)  # Modified `find_cycle` returns edges with edge_id
        if not cycle:  # No cycle found
            break

        # Ensure all edges in the cycle exist in the weights dictionary
        cycle = [(u, v, edge_id) for u, v, edge_id in cycle if (u, v, edge_id) in weights_copy]

        if not cycle:  # If no valid cycle exists, continue
            continue

        # Find the minimum weight in the cycle
        min_weight = min(weights_copy[(u, v, edge_id)] for u, v, edge_id in cycle)

        for u, v, edge_id in cycle:
            weights_copy[(u, v, edge_id)] -= min_weight
            if weights_copy[(u, v, edge_id)] <= 0:
                # Ensure the edge is in the graph before removing
                if (v, edge_id) in graph[u]:
                    graph[u].remove((v, edge_id))
                    removed_edges.add((u, v, edge_id))
                    removed_weights[(u, v, edge_id)] = weights[(u, v, edge_id)]

    return removed_edges, removed_weights


def find_cycle(graph, n):
    """
    Detect a cycle in the graph using DFS and return the cycle as a list of edges.
    Handles parallel edges and cycles of length 2 caused by reverse edges.

    Args:
        graph (dict): Adjacency list with (neighbor, edge_id) pairs.
        n (int): Number of vertices in the graph.

    Returns:
        cycle (list): List of edges forming the cycle, or None if no cycle is found.
    """
    visited = [False] * n
    stack = [False] * n
    parent = [-1] * n
    edge_to_parent = {}  # Map to track edge_id for cycle reconstruction

    def dfs(v):
        visited[v] = True
        stack[v] = True
        for neighbor, edge_id in graph[v]:
            if not visited[neighbor]:
                parent[neighbor] = v
                edge_to_parent[neighbor] = edge_id
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
            elif stack[neighbor]:
                # Found a cycle, reconstruct it
                cycle = []
                current = v
                while current != neighbor:
                    if current not in edge_to_parent:
                        break  # Avoid KeyError if edge metadata is missing
                    cycle.append((parent[current], current, edge_to_parent[current]))
                    current = parent[current]

                # Handle the root of the cycle
                if neighbor in edge_to_parent and parent[neighbor] != -1:
                    cycle.append((parent[neighbor], neighbor, edge_to_parent[neighbor]))
                return cycle

        stack[v] = False
        return None

    # Detect length-2 cycles caused by reverse edges
    for u in list(graph):  # Use list(graph) to iterate over a static copy of keys
        for neighbor, edge_id1 in graph[u]:
            for neighbor_of_neighbor, edge_id2 in graph[neighbor]:
                if neighbor_of_neighbor == u and edge_id1 != edge_id2:
                    # Found a length-2 cycle
                    return [(u, neighbor, edge_id1), (neighbor, u, edge_id2)]

    # Run DFS for longer cycles
    for i in range(n):
        if not visited[i]:
            cycle = dfs(i)
            if cycle:
                return cycle

    return None


def update_edge_weights(graph, weights):
    """
    Updates the weights of the edges in the graph based on the specified rule.
    If both (i, j) and (j, i) edges exist, the smaller weight is removed, and the larger weight is updated to:
    w_{i,j} / (w_{i,j} + w_{j,i}).
    If only one edge exists, its weight is updated to 1.

    Args:
        graph (dict): Adjacency list with (neighbor, edge_id) pairs.
        weights (dict): Dictionary mapping (start, end, edge_id) to weights.

    Returns:
        updated_graph (dict): Updated adjacency list.
        updated_weights (dict): Updated weights dictionary.
    """
    processed_edges = set()  # Keep track of processed edges
    updated_graph = defaultdict(list)
    updated_weights = {}

    # Iterate over a copy of the graph's keys to avoid modification during iteration
    for u in list(graph.keys()):
        for v, edge_id in graph[u]:
            if (u, v, edge_id) not in processed_edges:
                reverse_edge = next(
                    ((w, rev_edge_id) for w, rev_edge_id in graph[v] if w == u),
                    None
                )
                if reverse_edge:
                    # Get reverse edge weight
                    rev_edge_id = reverse_edge[1]
                    w_uv = weights[(u, v, edge_id)]
                    w_vu = weights[(v, u, rev_edge_id)]

                    if w_uv >= w_vu:
                        # Update weight of (u, v)
                        updated_weight = w_uv / (w_uv + w_vu)
                        updated_graph[u].append((v, edge_id))
                        updated_weights[(u, v, edge_id)] = updated_weight
                        # Mark (v, u) as processed
                        processed_edges.add((v, u, rev_edge_id))
                    else:
                        # Update weight of (v, u)
                        updated_weight = w_vu / (w_uv + w_vu)
                        updated_graph[v].append((u, rev_edge_id))
                        updated_weights[(v, u, rev_edge_id)] = updated_weight
                        # Mark (u, v) as processed
                        processed_edges.add((u, v, edge_id))
                else:
                    # No reverse edge, update the weight to 1
                    updated_graph[u].append((v, edge_id))
                    updated_weights[(u, v, edge_id)] = 1.0

                # Mark this edge as processed
                processed_edges.add((u, v, edge_id))

    return updated_graph, updated_weights

def check_and_readd_edges(graph, removed_edges, n):
    """
    Phase 2: Check and re-add edges if they do not create a cycle.
    Handles graphs with parallel edges using edge identifiers.

    Args:
        graph (dict): Adjacency list with (neighbor, edge_id) pairs.
        removed_edges (set): Set of removed edges as (u, v, edge_id).
        n (int): Number of vertices in the graph.

    Returns:
        readded_edges (set): Set of edges that were successfully re-added.
        remaining_removed_edges (set): Set of edges that could not be re-added.
    """

    def has_path(start, end, graph):
        """
        Helper function to check if there is a path from start to end using DFS.
        Avoids cycles when re-adding edges.
        """
        visited = [False] * n
        stack = [start]
        while stack:
            node = stack.pop()
            if node == end:
                return True
            if not visited[node]:
                visited[node] = True
                stack.extend(neighbor for neighbor, _ in graph[node])  # Add only neighbors
        return False

    readded_edges = set()
    removed_edges_list = sorted(list(removed_edges), reverse=True)

    for u, v, edge_id in removed_edges_list:
        if not has_path(v, u, graph):  # Only re-add if it doesn't create a cycle
            graph[u].append((v, edge_id))
            readded_edges.add((u, v, edge_id))

    remaining_removed_edges = removed_edges - readded_edges
    return readded_edges, remaining_removed_edges
def mwfas_synthetic(edges):
    """
    Main function to find Minimum Weighted Feedback Arc Set (MWFAS) in a graph with parallel edges.

    :param file_path: Path to the file containing the graph.
    :return: A dictionary with metrics, updated graph, removed edges, and their weights.
    """
    # Read the graph and initialize its structure

    n = max(max(u, v) for u, v, _, _ in edges) + 1
    graph, weights = initialize_graph(edges)

    # Original graph statistics
    total_edges = len(edges)
    total_weight = sum(w for _, _, w, _ in edges)

    # Phase 1: Reduce cycles
    removed_edges, removed_weights = find_cycles_and_reduce(graph, weights, n)

    # Phase 2: Re-add edges (if applicable)
    readded_edges, remaining_removed_edges = check_and_readd_edges(graph, removed_edges, n)

    # Compute final metrics
    num_removed_edges = len(remaining_removed_edges)
    total_removed_weight = sum(removed_weights.get(edge, 0) for edge in remaining_removed_edges)

    # Return results
    return {
        "total_edges": total_edges,
        "total_weight": total_weight,
        "num_removed_edges": num_removed_edges,
        "removed_weight": total_removed_weight,
        "final_graph": graph,
        "removed_edges": remaining_removed_edges,
        "removed_weights": {edge: removed_weights.get(edge, 0) for edge in remaining_removed_edges},
    }


class DirectedGraph:
    def __init__(self, vertices):
        self.graph = defaultdict(list)
        self.vertices = vertices
        self.weights = {}  # Maintain weights dictionary with unique edge keys

    def add_edge(self, src, dest, weight=1, edge_id=None):
        """
        Add a directed edge with weight and an optional unique edge_id.
        """
        if edge_id is None:
            edge_id = (src, dest, len(self.graph[src]))  # Generate unique edge_id
        self.graph[src].append((dest, edge_id))
        self.weights[(src, dest, edge_id)] = weight

    def get_edges(self):
        edges = []
        for src in self.graph:
            for dest, edge_id in self.graph[src]:
                edges.append((src, dest, self.weights[(src, dest, edge_id)]))
        return edges

    def remove_edge(self, src, dest):
        self.graph[src] = [(d, eid) for d, eid in self.graph[src] if d != dest]
        self.weights = {key: weight for key, weight in self.weights.items() if key[0] != src or key[1] != dest}

    def get_indegree(self):
        indegree = {v: 0 for v in range(self.vertices)}
        for src in self.graph:
            for dest, _ in self.graph[src]:
                indegree[dest] += 1
        return indegree

    def get_outdegree(self):
        outdegree = {v: 0 for v in range(self.vertices)}
        for src in self.graph:
            for dest, _ in self.graph[src]:
                outdegree[src] += 1
        return outdegree

    def eliminate_parallel_arcs(self):
        for src in self.graph:
            seen = {}
            for dest, weight in self.graph[src]:
                if dest in seen:
                    seen[dest] += self.weights[(src, dest, weight)]
                else:
                    seen[dest] = self.weights[(src, dest, weight)]
            self.graph[src] = [(dest, weight) for dest, weight in seen.items()]

    def eliminate_two_cycles(self):
        for src in list(self.graph.keys()):
            for dest, weight in self.graph[src]:
                for back_dest, back_weight in self.graph[dest]:
                    if back_dest == src:
                        if weight > back_weight:
                            self.remove_edge(src, dest)
                            self.remove_edge(dest, src)
                        elif weight == back_weight:
                            self.remove_edge(src, dest)
                            self.remove_edge(dest, src)

def compute_vertex_rankings(graph, weights, n):
    """
    Compute rankings for the vertices in a DAG with parallel edges.
    :param graph: Adjacency list of the DAG with (neighbor, edge_id) pairs.
    :param weights: Dictionary of edge weights with keys (u, v, edge_id).
    :param n: Total number of vertices in the graph.
    :return: A list of rankings for the vertices.
    """
    # Step 1: Calculate in-degrees
    in_degree = [0] * n
    for u in graph:
        for v, _ in graph[u]:  # Ignore edge_id for in-degree calculation
            in_degree[v] += 1

    # Step 2: Perform topological sort using a min-heap
    from heapq import heappop, heappush
    min_heap = []
    for i in range(n):
        if in_degree[i] == 0:
            heappush(min_heap, i)

    topological_order = []
    while min_heap:
        current = heappop(min_heap)
        topological_order.append(current)
        for neighbor, _ in graph[current]:  # Ignore edge_id for topological sort
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                heappush(min_heap, neighbor)

    # Step 3: Calculate outgoing and incoming edge weight sums for all vertices
    outgoing_weights = {v: 0 for v in range(n)}
    incoming_weights = {v: 0 for v in range(n)}

    for u in graph:
        for v, edge_id in graph[u]:
            edge_key = (u, v, edge_id)
            outgoing_weights[u] += weights.get(edge_key, 0)
            incoming_weights[v] += weights.get(edge_key, 0)

    # Step 4: Assign rankings
    rankings = [-1] * n
    current_rank = 0
    for vertex in topological_order:
        rankings[vertex] = current_rank
        current_rank += 1

    # Break ties for vertices with the same ranking based on outgoing and incoming edge weights
    tied_vertices = sorted(
        [(rankings[v], -(outgoing_weights[v] - incoming_weights[v]) /
          (outgoing_weights[v] + incoming_weights[v] if outgoing_weights[v] + incoming_weights[v] > 0 else 1), v)
         for v in range(n)],
        key=lambda x: (x[0], x[1])  # Sort by rank first, then by normalized weight difference
    )

    scores = [0] * n
    for final_rank, (_, _, vertex) in enumerate(tied_vertices):
        scores[vertex] = n - final_rank - 1

    return scores


def graph_to_adjacency_matrix(graph, weights, n):
    """
    Convert the graph to an adjacency matrix with weights.
    :param graph: Adjacency list of the DAG with (neighbor, edge_id) pairs.
    :param weights: Dictionary of edge weights with keys (u, v, edge_id).
    :param n: Total number of vertices in the graph.
    :return: An adjacency matrix (n x n) with weights as a PyTorch tensor.
    """
    adjacency_matrix = torch.zeros((n, n), dtype=torch.float32)

    for u in graph:
        for v, edge_id in graph[u]:
            edge_key = (u, v, edge_id)
            adjacency_matrix[u, v] = weights.get(edge_key, 0)

    return adjacency_matrix
def reorder_floats(x):
    n = len(x)
    random_floats = np.random.uniform(0,  2*n/3, n)
    y = np.zeros(n)
    for idx, val in enumerate(np.argsort(x)):
        y[val] = sorted(random_floats)[idx]
    return y

def calculate_upset_loss(adjacency_matrix, scores, style='ratio', margin=0.01):
    """
    Calculate the upset loss for the graph rankings using adjacency matrix and scores.

    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Torch FloatTensor ranking scores (n x 1).
    :param style: Type of upset loss ('naive', 'simple', 'ratio', or 'margin').
    :param margin: Margin for margin loss (default: 0.01).
    :return: Torch FloatTensor upset loss value.
    """
    epsilon = 1e-8  # For numerical stability

    # Ensure scores are 2D
    if scores.ndim == 1:
        scores = scores.view(-1, 1)

    # Skew-symmetric pairwise comparison matrix (M)
    M1 = adjacency_matrix - adjacency_matrix.T

    # Normalize scores to [0, 1] range
    normalized_scores = scores

    # Pairwise score differences (T)
    T1 = normalized_scores - normalized_scores.T

    # Edge mask: Only consider meaningful edges (where M != 0)
    edge_mask = M1 != 0

    if style == 'ratio':
        min_upset = float('inf')  # Initialize with a large value

        for _ in range(40):
            # Generate reordered scores using reorder_floats
            if _==0:
                reordered_scores=scores
            else:
                reordered_scores = torch.FloatTensor(reorder_floats(scores.flatten().tolist()))
            reordered_scores = reordered_scores.view(-1, 1)

            # Compute T2 for normalized scores
            T2 = reordered_scores + reordered_scores.T + epsilon
            T = torch.div(T1, T2)
            M2 = adjacency_matrix + adjacency_matrix.T + epsilon
            M3 = torch.div(M1, M2)  # Normalize the adjacency matrix

            # Compute ratio-based upset loss for this iteration
            powers = torch.pow((M3 - T)[edge_mask], 2)
            upset_loss = torch.sum(powers) / torch.sum(edge_mask)

            # Track the minimum upset loss
            min_upset = min(min_upset, upset_loss.item())

        return torch.tensor(min_upset)

    elif style == 'naive':
        upset = torch.sum(torch.sign(T1[edge_mask]) != torch.sign(M1[edge_mask])) / torch.sum(edge_mask)

    elif style == 'simple':
        upset = torch.mean((torch.sign(T1[edge_mask]) - torch.sign(M1[edge_mask]))**2)

    elif style == 'margin':
        upset = torch.mean(torch.nn.functional.relu(-M1[edge_mask] * (T1[edge_mask] - margin)))

    else:
        raise ValueError(f"Unsupported style: {style}")

    return upset
def minimize_ratio_upset_loss(adjacency_matrix, scores, epsilon=1e-2, max_time=120):
    """
    Perform optimization to minimize the ratio upset loss, ensuring naive and simple losses do not worsen.
    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Initial scores for optimization.
    :param epsilon: Small value for numerical stability (default: 1e-2).
    :param max_time: Maximum time for optimization in seconds.
    :return: Tuple (optimal_scores, minimized_loss)
    """
    from scipy.optimize import minimize

    # Compute initial losses
    initial_scores = scores.clone().detach().view(-1).numpy()
    initial_losses = {
        "naive": calculate_upset_loss(adjacency_matrix, scores.view(-1, 1), style="naive"),
        "simple": calculate_upset_loss(adjacency_matrix, scores.view(-1, 1), style="simple"),
    }

    # Objective function
    def objective_function(updated_scores, adjacency_matrix, initial_losses):
        updated_scores = torch.tensor(updated_scores, dtype=torch.float32).view(-1, 1)
        ratio_loss = calculate_upset_loss(adjacency_matrix, updated_scores, style="ratio")
        naive_loss = calculate_upset_loss(adjacency_matrix, updated_scores, style="naive")
        simple_loss = calculate_upset_loss(adjacency_matrix, updated_scores, style="simple")

        penalty = 0
        if naive_loss > initial_losses["naive"]:
            penalty += naive_loss - initial_losses["naive"]
        if simple_loss > initial_losses["simple"]:
            penalty += simple_loss - initial_losses["simple"]

        return ratio_loss + 100 * penalty

    # Timer callback
    class TimerCallback:
        def __init__(self, max_time, objective_function, adjacency_matrix, initial_losses):
            self.start_time = time.time()
            self.max_time = max_time
            self.iterations = 0  # Track iterations
            self.objective_function = objective_function
            self.adjacency_matrix = adjacency_matrix
            self.initial_losses = initial_losses
            self.min_loss = float("inf")  # Track minimum loss

        def __call__(self, xk, *args, **kwargs):
            self.iterations += 1

            # Compute the objective function value
            current_loss = self.objective_function(
                xk, self.adjacency_matrix, self.initial_losses
            )

            # Update and print the minimum loss found so far
            self.min_loss = min(self.min_loss, current_loss)
            print(f"Iteration {self.iterations}: Minimum loss so far: {self.min_loss:.6f}")

            # Stop optimization if the time limit is exceeded
            if time.time() - self.start_time > self.max_time:
                print("Time limit exceeded, stopping optimization.")
                raise StopIteration  # Signal COBYLA to stop

    # Create the callback instance
    callback = TimerCallback(
        max_time=max_time,
        objective_function=objective_function,
        adjacency_matrix=adjacency_matrix,
        initial_losses=initial_losses,
    )

    try:
        # Run optimization with COBYLA
        result = minimize(
            fun=objective_function,
            x0=initial_scores,
            method="COBYLA",
            args=(adjacency_matrix, initial_losses),  # Pass required arguments
            options={"maxiter": 500, "disp": False},
            callback=callback,  # Logs the minimum loss after each iteration
        )
    except StopIteration:
        print("Optimization stopped early due to time limit.")

    # Extract results
    optimal_scores = result.x
    minimized_loss = result.fun

    print(f"Optimization completed after {callback.iterations} iterations.")
    print(f"Final minimum loss: {callback.min_loss:.6f}")
    return optimal_scores, minimized_loss


def compute_ratio_upset_loss(adjacency_matrix, scores, epsilon=1e-8):
    """
    Compute the ratio upset loss for the graph rankings using adjacency matrix and scores.

    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Torch FloatTensor ranking scores (n x 1).
    :param epsilon: Small value for numerical stability (default: 1e-8).
    :return: Torch FloatTensor ratio upset loss value.
    """
    if scores.ndim == 1:
        scores = scores.view(-1, 1)

    M1 = adjacency_matrix - adjacency_matrix.T
    T1 = scores - scores.T
    edge_mask = M1 != 0

    T2 = scores + scores.T + epsilon
    T = torch.div(T1, T2)

    M2 = adjacency_matrix + adjacency_matrix.T + epsilon
    M3 = torch.div(M1, M2)

    powers = torch.pow((M3 - T)[edge_mask], 2)
    upset_loss = torch.sum(powers) / torch.sum(edge_mask)

    return upset_loss


def plot_ratio_loss(adjacency_matrix, scores, index, lower_bound, upper_bound, steps=100, output_folder="loss_plots"):
    """
    Plot and save the compute_ratio_upset_loss function for scores[index] between lower_bound and upper_bound.

    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Torch FloatTensor ranking scores (n x 1).
    :param index: Index of the score to vary.
    :param lower_bound: Lower bound for the score value.
    :param upper_bound: Upper bound for the score value.
    :param steps: Number of steps for sampling the range.
    :param output_folder: Directory to save the plots.
    """
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    x_values = torch.linspace(lower_bound, upper_bound, steps)
    y_values = []

    original_score = scores[index].item()

    for x in x_values:
        scores[index] = x
        loss = compute_ratio_upset_loss(adjacency_matrix, scores)
        y_values.append(loss.item())

    # Restore the original score
    scores[index] = original_score

    # Plot the graph
    plt.figure(figsize=(8, 6))
    plt.plot(x_values.numpy(), y_values, label=f"Loss vs. scores[{index}]")
    plt.xlabel(f"scores[{index}] value")
    plt.ylabel("Ratio Upset Loss")
    plt.title(f"Compute Ratio Upset Loss for Varying scores[{index}]")
    plt.legend()
    plt.grid(True)

    # Save the plot with a unique name
    plot_path = os.path.join(output_folder, f"loss_plot_index_{index}.png")
    counter = 1
    while os.path.exists(plot_path):
        plot_path = os.path.join(output_folder, f"loss_plot_index_{index}_{counter}.png")
        counter += 1

    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved: {plot_path}")


def trinary_search_optimize(adjacency_matrix, scores, index, lower_bound, upper_bound, epsilon=1e-8, steps=100):
    """
    Perform ternary search to find the optimal score value for a given index that minimizes ratio upset loss.

    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Torch FloatTensor ranking scores (n x 1).
    :param index: Index of the score to optimize.
    :param lower_bound: Lower bound for the score value.
    :param upper_bound: Upper bound for the score value.
    :param epsilon: Small value for numerical stability.
    :param steps: Number of steps for ternary search.
    :return: Optimal score value for the given index.
    """
    for _ in range(steps):
        mid1 = lower_bound + (upper_bound - lower_bound) / 3.0
        mid2 = upper_bound - (upper_bound - lower_bound) / 3.0

        scores[index] = mid1
        loss1 = compute_ratio_upset_loss(adjacency_matrix, scores)

        scores[index] = mid2
        loss2 = compute_ratio_upset_loss(adjacency_matrix, scores)

        if loss1 < loss2:
            upper_bound = mid2
        elif loss1 > loss2:
            lower_bound = mid1
        else:
            lower_bound = mid1
            upper_bound = mid2

        # Break if the range is small enough
        if upper_bound - lower_bound < epsilon:
            break

    # After the loop, check the losses at lower_bound, upper_bound, and midpoint
    mid_point = (lower_bound + upper_bound) / 2.0
    scores[index] = mid_point
    mid_point_loss = compute_ratio_upset_loss(adjacency_matrix, scores)

    scores[index] = lower_bound
    lower_bound_loss = compute_ratio_upset_loss(adjacency_matrix, scores)

    scores[index] = upper_bound
    upper_bound_loss = compute_ratio_upset_loss(adjacency_matrix, scores)

    # Find the minimum loss and corresponding score
    min_loss = min(mid_point_loss, lower_bound_loss, upper_bound_loss)
    if min_loss == lower_bound_loss:
        optimal_score = lower_bound
    elif min_loss == upper_bound_loss:
        optimal_score = upper_bound
    else:
        optimal_score = mid_point

    scores[index] = optimal_score
    return scores[index]


def minimize_ratio_loss(adjacency_matrix, scores):
    """
    Minimize the ratio upset loss by iteratively optimizing each score using binary search.

    :param adjacency_matrix: Torch FloatTensor adjacency matrix (n x n).
    :param scores: Torch FloatTensor ranking scores (n x 1).
    :return: Optimized scores.
    """
    scores = scores.clone()  # Create a copy to avoid modifying the original

    for _ in range(40):
        sorted_indices = torch.argsort(scores.squeeze())
        print(_)
        for i in range(len(sorted_indices)):


            index = sorted_indices[i]

            if i == 0:
                lower_bound = 0
                upper_bound = scores[sorted_indices[i + 1]].item()
            elif i == len(sorted_indices) - 1:
                lower_bound = scores[sorted_indices[i - 1]].item()
                upper_bound = scores.max().item() + 10  # Extend beyond max for the last element
            else:
                lower_bound = scores[sorted_indices[i - 1]].item()
                upper_bound = scores[sorted_indices[i + 1]].item()

            # Perform binary search optimization for the current score
            scores[index] = trinary_search_optimize(adjacency_matrix, scores, index, lower_bound, upper_bound)
          #  print("i am here")
    return scores

# Example usage


def generate_ero_graph(n: int, p: float, eta: float, style: str = 'uniform'):
    """
    Generates an Erdős-Rényi Outliers (ERO) model graph in a format compatible with initialize_graph() and mwfas().

    Args:
        n (int): Number of nodes.
        p (float): Edge probability (sparsity).
        eta (float): Noise level (between 0 and 1).
        style (str): How to generate ground-truth scores ('uniform' or 'gamma').

    Returns:
        list: List of edges (start, end, weight, edge_id) for MWFAS.
        np.array: Ground-truth ranking of nodes.
    """
    # Generate node scores
    if style == 'uniform':
        scores = np.random.rand(n, 1)
        R_noise = np.random.rand(n, n) * 2 - 1
    elif style == 'gamma':
        scores = np.random.gamma(shape=0.5, scale=1, size=(n, 1))
        R_noise = np.random.rand(n, n) * 4 - 2  # Gamma noise

    # Compute ground-truth ranking
    labels = rankdata(-scores.flatten(), 'min')

    # Generate pairwise comparisons matrix
    R_GT = scores - scores.T  # True pairwise differences
    R_choice = np.random.rand(n, n)
    R = np.zeros((n, n))
    R[R_choice <= p] = R_noise[R_choice <= p]  # Assign noisy comparisons
    R[R_choice <= p * (1 - eta)] = R_GT[R_choice <= p * (1 - eta)]  # Assign correct comparisons

    # Ensure antisymmetry
    lower_ind = np.tril_indices(n)
    diag_ind = np.diag_indices(n)
    R[lower_ind] = -R.T[lower_ind]
    R[diag_ind] = 0
    R[R < 0] = 0  # Ensure positive weights

    # Convert matrix to edge list format for mwfas (edge_id is the last element)
    edges = []
    edge_id = 0
    R_coo = sp.csr_matrix(R).tocoo()
    for u, v, w in zip(R_coo.row, R_coo.col, R_coo.data):
        edges.append((int(u), int(v), float(w), edge_id))
        edge_id += 1

    return edges, labels


def kendall_tau_loss(true_ranking, predicted_ranking):
    """
    Computes Kendall tau loss, which measures ranking disagreement.

    Args:
        true_ranking (np.array): Ground-truth ranking.
        predicted_ranking (np.array): Computed ranking.

    Returns:
        float: Kendall tau loss (fraction of discordant pairs).
    """
    n = len(true_ranking)
    true_order = np.argsort(true_ranking)
    predicted_order = np.argsort(predicted_ranking)
    discordant_pairs = sum(
        (true_order[i] > true_order[j]) != (predicted_order[i] > predicted_order[j])
        for i in range(n)
        for j in range(i + 1, n)
    )
    total_pairs = n * (n - 1) / 2
    return discordant_pairs / total_pairs
