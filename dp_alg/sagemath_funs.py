import subprocess

import networkx as nx
from sage.all import Graph
from sage.graphs.graph_decompositions.tree_decomposition import label_nice_tree_decomposition


# define an enum for node type
class NodeType:
    LEAF = 'leaf'
    INTRODUCE = 'introduce'
    FORGET = 'forget'
    JOIN = 'join'
    ROOT = 'root'

class TreeNode:
    """
    Represents a node in the nice tree decomposition.
    """

    def __init__(self, node_type, bag=None, children=None, vertex=None):
        """
        Initializes a TreeNode.

        :param node_type: Type of the node ('leaf', 'introduce', 'forget', 'join')
        :param bag: Set of vertices in the current bag
        :param children: List of child TreeNode instances
        :param vertex: The vertex being introduced or forgotten (for 'introduce'/'forget' nodes)
        """
        self.type = node_type
        self.bag = set(bag) if bag else set()
        self.children = children if children else []
        self.vertex = vertex  # Relevant for 'introduce' and 'forget' nodes
        self.Vt = None


def compute_treewidth_from_PACE(graph):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    gr_content = f"p tw {num_nodes} {num_edges}\n" + "\n".join(f"{u+1} {v+1}" for u, v in graph.edges())
    result = subprocess.run(['./PACE2017-TrackA-master/tw-exact'], input=gr_content.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            cwd="./tree_decomp/PACE2017-TrackA-master")

    all_lines = result.stdout.decode()

    first_line = all_lines.splitlines()[0]
    treewidth = int(first_line.split()[3])
    return treewidth - 1


def nice_tree_decomp(nx_graph, tw_lower=None, tw_upper=None):
    # starting_time = time.process_time()
    G_sage = Graph(nx_graph)
    # print(f"Time taken to convert to Sage graph: {time.process_time() - starting_time:.6f} seconds")

    # starting_time = time.process_time()
    nice_TD = G_sage.treewidth(certificate=True, nice=True, kmin=tw_lower, k=tw_upper)
    # print(f"Time taken to compute nice tree decomposition: {time.process_time() - starting_time:.6f} seconds")

    # starting_time = time.process_time()
    root = sorted(nice_TD.vertices())[0]
    # print(f"Time taken to find root: {time.process_time() - starting_time:.6f} seconds")

    # starting_time = time.process_time()
    label_TD = label_nice_tree_decomposition(nice_TD, root, directed=True)
    # print(f"Time taken to label nice tree decomposition: {time.process_time() - starting_time:.6f} seconds")

    # starting_time = time.process_time()
    nx_nice_tree = label_TD.networkx_graph()
    # print(f"Time taken to convert to NetworkX graph: {time.process_time() - starting_time:.6f} seconds")
    return nx_nice_tree


def conv_nx_nt_to_treenode(nx_tree: nx.DiGraph) -> TreeNode:
    def get_node_type(current_bag, child_bags):
        if not child_bags:
            return NodeType.LEAF, None
        elif len(child_bags) == 1:
            child_bag = next(iter(child_bags))
            diff = current_bag - child_bag
            if len(diff) == 1:
                return NodeType.INTRODUCE, diff.pop()
            diff = child_bag - current_bag
            if len(diff) == 1:
                return NodeType.FORGET, diff.pop()
            return None
        else:
            return NodeType.JOIN, None


    root_id = next(iter(nx_tree.nodes()))

    def build_tree(node_id):
        t, X_t = node_id
        children = list(nx_tree.successors(node_id))
        child_nodes = [build_tree(child) for child in children]
        child_bags = [child.bag for child in child_nodes]
        node_type, vertex = get_node_type(set(X_t), child_bags)
        return TreeNode(node_type, bag=X_t, children=child_nodes, vertex=vertex)

    root_node = build_tree(root_id)
    return root_node
