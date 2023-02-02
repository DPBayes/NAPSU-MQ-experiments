import graphviz

class TreeNode:
    def __init__(self, variables, parent):
        self.variables = variables
        self.parent = parent
        self.children = []
        self.reset()
        
    def reset(self):
        self.potential = None
        self.messages = []
        self.result = None

class EmptyFactor:
    def __init__(self, scope):
        self.scope = set(scope)
        self.origin = None

    def product(self, other):
        return EmptyFactor(self.scope.union(other.scope))

    def marginalise(self, variable):
        return EmptyFactor(self.scope.difference({variable}))

    def list_product(factors):
        factors = list(factors)
        product = factors[0]
        for i in range(1, len(factors)):
            product = product.product(factors[i])
        return product

class JunctionTree:
    """Implementation of a junction tree.

    This method does not check that the given nodes and edges actually define 
    a valid junction tree.
    """
    def __init__(self, nodes, edges, cliques):
        """Create the junction tree explicitly.

        Args:
            nodes (set): The nodes of the junction tree.
            edges (dict): Edges as a dict with pair of nodes as keys and sets of nodes (separators) as values.
            cliques (list(tuple)): List of the factor scopes of the underlying Markov network.
        """
        self.nodes = nodes 
        self.edges = edges
        self.cliques = cliques
        self.init_factor_assignments()
        self.check_nodes_edges()
        self.init_node_orders()

    def init_factor_assignments(self):
        factor_assignment = {}
        for scope in self.cliques:
            for node in self.nodes:
                if set(scope).issubset(node):
                    if node not in factor_assignment.keys():
                        factor_assignment[node] = []
                    factor_assignment[node].append(scope)
                    break
        self.factors_in_node = factor_assignment
        self.node_for_factor = {}
        for node, factors in self.factors_in_node.items():
            for factor in factors:
                self.node_for_factor[factor] = node

    def init_node_orders(self):
        self.root_variables = list(self.nodes)[0]
        self.downward_order = self.compute_downward_order(self.root_variables)
        self.root_node = self.downward_order[0]
        self.upward_order = self.downward_order[::-1]

    def check_nodes_edges(self):
        for e1, e2 in self.edges.keys():
            edge = (e1, e2)
            if e1 not in self.nodes:
                raise ValueError("{} from {} not in nodes".format(e1, edge))
            if e2 not in self.nodes:
                raise ValueError("{} from {} not in nodes".format(e2, edge))
            if (e2, e1) in self.edges:
                raise ValueError("Duplicate edge {}".format(edge))
            if self.edges[edge] != set(e1).intersection(set(e2)):
                raise ValueError("Separator {} is not the intersection of {} and {}".format(set(e1).intersection(set(e2)), e1, e2))

    def check_asyclicity(self):
        marked_nodes = set()
        node_stack = [list(self.nodes)[0]]
        while len(node_stack) > 0:
            node = node_stack.pop()
            if node in marked_nodes:
                raise ValueError("Detected cycle at node {}".format(node))
            marked_nodes.add(node)
            neighbours = self.get_neighbours(node)
            for neighbour in neighbours:
                if neighbour not in marked_nodes:
                    node_stack.append(neighbour)


    def from_variable_elimination(feature_sets, elimination_order):
        """Create a junction tree from a variable elimination run.

        Args:
            feature_sets (list(tuple)): The factor scopes of the variable elimination.
            elimination_order (list): The elimination order.

        Returns:
            JunctionTree: The resulting JunctionTree.
        """
        factors = [EmptyFactor(feature_set) for feature_set in feature_sets]
        nodes = set()
        edges = {}

        for variable in elimination_order:
            factors = JunctionTree.eliminate_var(factors, variable, nodes, edges)

        jt = JunctionTree(nodes, edges, feature_sets)
        return jt

    def eliminate_var(factors, variable, nodes, edges):
        factors_in_prod = [factor for factor in factors if variable in factor.scope]
        factors_not_in_prod = [factor for factor in factors if variable not in factor.scope]

        prod_factor = EmptyFactor.list_product(factors_in_prod)
        prod_scope = tuple(prod_factor.scope)
        nodes.add(prod_scope)

        for factor in factors_in_prod:
            if factor.origin is not None:
                edges[(factor.origin, prod_scope)] = set(prod_scope).intersection(set(factor.origin))

        summed_factor = prod_factor.marginalise(variable)
        summed_factor.origin = prod_scope
        factors_not_in_prod.append(summed_factor)
        return factors_not_in_prod

    def visualize(self):
        """Visualize the junction tree with Graphviz.

        Returns:
            Graphviz graph: The resulting Graphviz object.
        """
        g = graphviz.Graph()
        for (e1, e2), sep in self.edges.items():
            g.edge(str(e1), str(e2), label=str(sep))
        return g

    def get_neighbours(self, node):
        return [other for other in self.nodes if (node, other) in self.edges or (other, node) in self.edges]

    def add_edge(self, edge):
        e1, e2 = edge
        self.edges[edge] = set(e1).intersection(set(e2))

    def remove_redundant_nodes(self):
        """Prune the juction tree to remove redundant nodes.
        """
        while(True):
            res = self.find_redundant_node()
            if res is not None:
                node, large_neighbour, neighbours = res
                self.nodes.remove(node)
                self.edges = {(e1, e2): sep for (e1, e2), sep in self.edges.items() if e1 != node and e2 != node}
                for neighbour in neighbours:
                    if neighbour != large_neighbour:
                        self.add_edge((large_neighbour, neighbour))
                
            else:
                self.check_nodes_edges()
                self.init_factor_assignments()
                self.init_node_orders()
                return

    def find_redundant_node(self):
        for node in self.nodes:
            neighbours = self.get_neighbours(node)
            for neighbour in neighbours:
                if set(node).issubset(set(neighbour)):
                    return node, neighbour, neighbours

        return None

    def compute_downward_order(self, root_node):
        order = []
        marked_nodes = set()
        root = TreeNode(root_node, None)
        node_stack = [root]
        while len(node_stack) > 0:
            node = node_stack.pop()
            order.append(node)
            marked_nodes.add(node.variables)
            for neighbour in self.get_neighbours(node.variables):
                if neighbour not in marked_nodes:
                    new_node = TreeNode(neighbour, node)
                    node_stack.append(new_node)
                    node.children.append(new_node)

        return order
