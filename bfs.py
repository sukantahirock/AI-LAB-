class Node:
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)

def depth_limited_search(node, goal, depth, visited):
    visited.append(node.data)  # Mark the node as visited
    if depth == 0:
        return None, visited
    if node.data == goal:
        return node, visited
    if not node.children:
        return None, visited
    for child in node.children:
        result, visited = depth_limited_search(child, goal, depth - 1, visited)
        if result is not None:
            return result, visited
    return None, visited

def iterative_deepening_search(root, goal, max_depth):
    for depth in range(max_depth + 1):
        result, visited = depth_limited_search(root, goal, depth, [])
        if result is not None:
            return result, visited
    return None, []

# Creating the tree
root = Node("0")
root.add_child(Node("1"))
root.add_child(Node("2"))
root.children[0].add_child(Node("3"))
root.children[0].add_child(Node("4"))
root.children[1].add_child(Node("5"))
root.children[1].add_child(Node("6"))

# Perform iterative deepening search
goal_node, visited_nodes = iterative_deepening_search(root, "5", 3)

if goal_node:
    print("Goal node found with data:", goal_node.data)
    print("Visited nodes:", visited_nodes)
else:
    print("Goal node not found.")
